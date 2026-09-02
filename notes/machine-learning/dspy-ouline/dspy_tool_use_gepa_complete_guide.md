# DSPyによるツール利用最適化：最初から実行する完全版

> JupyterLabでセル1から順番にコピー＆ペーストして実行するためのガイドです。  
> メンターから与えられた3つのMCPツールを使い、ReAct Agentの作成、ツール履歴の整理、Judgeによる評価、Baseline測定、GEPAによるプロンプト最適化、最適化後の比較まで行います。

## 0. 全体の流れ

```text
質問
  ↓
ReAct Agent
  ↓
3つのMCPツールを選択・実行
  ↓
trajectoryからツール履歴を取得
  ↓
Judge LMが採点とfeedbackを生成
  ↓
GEPAがAgentのプロンプトを改善
  ↓
最適化前後のスコアを比較
```

今回使用する3つのツールです。

| ツール | 用途 |
|---|---|
| `calculate_expression` | 1つの数式を計算する |
| `analyze_numbers` | 数値配列の平均や中央値などを求める |
| `convert_units` | 長さや重さなどの単位を変換する |

## 実行前の注意

- 社内MCPサーバーを使う場合は、接続できるネットワーク上で実行してください。
- Azure OpenAIのdeployment名は、Azure上で実際に作成されている名前を指定してください。
- APIキー、Azure endpoint、MCP URL、MCP APIキーをGitHubへ公開しないでください。
- すでにNotebook内で同名の変数を作っていても、このガイドを最初から順番に実行すれば上書きされます。

---

# Step 1：ライブラリとLMを設定する

## セル1：importとAzure OpenAI設定

```python
import os
import json
import httpx
import dspy


# メンターから指定された値を入力する。
# GitHubへ載せる前に、必ず秘密情報を削除する。
AZURE_OPENAI_API_KEY = ""
AZURE_OPENAI_ENDPOINT = ""
AZURE_OPENAI_API_VERSION = "2025-03-01-preview"
AZURE_OPENAI_DEPLOYMENT = "gpt-4o"


# 質問に回答し、ツールを操作するStudent LM。
lm = dspy.LM(
    f"azure/{AZURE_OPENAI_DEPLOYMENT}",
    api_key=AZURE_OPENAI_API_KEY,
    api_base=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION,
)

dspy.configure(lm=lm)
```

`lm`は、実際に質問を読み、どのツールを使うか判断するStudent LMです。

`dspy.configure(lm=lm)`により、特別な指定がないDSPy ModuleはこのLMを使います。

---

# Step 2：MCPサーバーを呼び出す関数を作る

## セル2：MCP接続設定と共通関数

```python
# メンターから指定された値を入力する。
MCP_URL = os.getenv(
    "MCP_URL",
    "",
)

MCP_API_KEY = os.getenv(
    "MCP_API_KEY",
    "",
)


def call_mcp_tool(name: str, arguments: dict) -> dict:
    """MCPサーバー上のツールを呼び出し、結果を辞書で返す。"""

    response = httpx.post(
        MCP_URL,
        headers={
            "Authorization": f"Bearer {MCP_API_KEY}",
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        },
        json={
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": name,
                "arguments": arguments,
            },
            "id": 1,
        },
        timeout=30,
    )

    response.raise_for_status()

    payload = response.json()

    # MCPのresult.content[0].textに入ったJSON文字列を辞書へ戻す。
    content = payload["result"]["content"][0]["text"]
    return json.loads(content)
```

この関数は、3つのツールに共通する通信処理をまとめています。

1. MCPサーバーへJSON-RPC形式でリクエストを送ります。
2. HTTPエラーがあれば`raise_for_status()`で例外を発生させます。
3. MCPから返されたツール結果をPythonの辞書へ変換します。

---

# Step 3：3つのツールを定義する

## セル3：数式計算ツール

```python
def calculate_expression(
    expression: str,
    precision: int = 6,
) -> dict:
    """
    1つの数式を計算する。
    四則演算、累乗、平方根などの計算に使用する。
    """

    return call_mcp_tool(
        "calculate_expression",
        {
            "expression": expression,
            "precision": precision,
        },
    )
```

## セル4：統計分析ツール

```python
def analyze_numbers(
    values: list[float],
    operations: list[str] | None = None,
    second_values: list[float] | None = None,
    outlier_method: str = "none",
) -> dict:
    """
    数値配列を統計分析する。
    平均、中央値、標準偏差、分散などを求める場合に使用する。
    """

    arguments = {
        "values": values,
        "outlier_method": outlier_method,
    }

    if operations is not None:
        arguments["operations"] = operations

    if second_values is not None:
        arguments["second_values"] = second_values

    return call_mcp_tool(
        "analyze_numbers",
        arguments,
    )
```

## セル5：単位変換ツール

```python
def convert_units(
    value: float,
    from_unit: str,
    to_unit: str,
    category: str | None = None,
    precision: int = 6,
) -> dict:
    """
    数値の単位を変換する。
    長さ、重さ、温度、時間、データサイズなどの変換に使用する。
    """

    arguments = {
        "value": value,
        "from_unit": from_unit,
        "to_unit": to_unit,
        "precision": precision,
    }

    if category is not None:
        arguments["category"] = category

    return call_mcp_tool(
        "convert_units",
        arguments,
    )
```

DSPyは、ツールの関数名、引数名、型ヒント、docstringをLMへ提示します。そのため、docstringには「そのツールを何に使うか」を日本語で書いています。

---

# Step 4：Agentの入出力を定義する

## セル6：`ToolQA`

```python
class ToolQA(dspy.Signature):
    """
    ユーザーの質問へ正確に回答してください。
    必要な場合は利用可能なツールから適切なものを選び、
    正しい引数を渡してください。
    ツールの実行結果を使って最終回答を作成してください。
    """

    question: str = dspy.InputField(
        desc="ユーザーが入力した質問"
    )

    answer: str = dspy.OutputField(
        desc="ユーザーの要求を満たす最終回答"
    )
```

`ToolQA`は計算をするクラスではなく、Agentの仕事を定義する仕様書です。

- `question`は入力です。
- `answer`は出力です。
- classのdocstringはAgentへの指示です。
- GEPAは、主にこのような指示文を改善します。

---

# Step 5：trajectoryを整理する

## 5.1 trajectoryとは

ReActは、最終回答だけでなく、途中の判断やツール実行を`trajectory`へ記録します。

```text
thought_0       ：0番目の判断
tool_name_0     ：0番目に選んだツール
tool_args_0     ：0番目のツールへ渡した引数
observation_0   ：0番目のツール結果
tool_name_1     ：次に選んだツール
```

例えば、`10 kmは何mですか？`では、おおむね次の履歴になります。

```python
{
    "thought_0": "単位を変換する必要がある",
    "tool_name_0": "convert_units",
    "tool_args_0": {
        "value": 10,
        "from_unit": "km",
        "to_unit": "m",
    },
    "observation_0": {"result": 10000},
    "tool_name_1": "finish",
}
```

`finish`はツールではなく、ReActが処理を終了する合図です。

## 5.2 trajectoryを整理する補助関数

## セル7：`extract_tool_history()`

```python
def extract_tool_history(trajectory):
    """
    ReActのtrajectoryからツール呼び出しと実行結果を取り出す。
    finishは終了合図なので、ツール呼び出しには含めない。
    """

    trajectory = trajectory or {}
    tool_calls = []
    tool_results = []

    for key, tool_name in trajectory.items():
        if not key.startswith("tool_name_"):
            continue

        step = key.removeprefix("tool_name_")
        tool_name = str(tool_name)

        if tool_name == "finish":
            continue

        tool_args = trajectory.get(
            f"tool_args_{step}",
            {},
        )

        observation = trajectory.get(
            f"observation_{step}"
        )

        tool_calls.append(
            {
                "step": step,
                "tool": tool_name,
                "arguments": tool_args,
            }
        )

        tool_results.append(
            {
                "step": step,
                "tool": tool_name,
                "result": observation,
            }
        )

    return (
        json.dumps(
            tool_calls,
            ensure_ascii=False,
            default=str,
        ),
        json.dumps(
            tool_results,
            ensure_ascii=False,
            default=str,
        ),
    )
```

この関数は、次の処理を行います。

1. `tool_name_0`、`tool_name_1`などのキーを探します。
2. 同じ番号の`tool_args`と`observation`を取り出します。
3. `finish`を除外します。
4. Judgeが読みやすいJSON文字列へ変換します。

`ensure_ascii=False`は、日本語を`\uXXXX`形式にせず、そのまま読める形で残す指定です。

---

# Step 6：ReAct Agentを作る

## セル8：`MCPToolAgent`

```python
class MCPToolAgent(dspy.Module):
    """3つのMCPツールを利用するReAct Agent。"""

    def __init__(self):
        """ReActと利用可能なツールを設定する。"""

        super().__init__()

        self.agent = dspy.ReAct(
            ToolQA,
            tools=[
                calculate_expression,
                analyze_numbers,
                convert_units,
            ],
            max_iters=5,
        )

    def forward(self, question: str):
        """質問を処理し、回答とツール履歴を返す。"""

        result = self.agent(
            question=question
        )

        tool_calls, tool_results = extract_tool_history(
            result.trajectory
        )

        return dspy.Prediction(
            answer=result.answer,
            tool_calls=tool_calls,
            tool_results=tool_results,
            trajectory=result.trajectory,
        )


agent = MCPToolAgent()
```

`dspy.ReAct`は、次の処理を繰り返すAgentです。

```text
考える
  ↓
ツールを選ぶ
  ↓
引数を決める
  ↓
ツール結果を見る
  ↓
次の行動を決める
```

`max_iters=5`は、最大5回まで行動できるという意味です。必ず5回ツールを呼ぶわけではありません。

## セル9：Agentの動作確認

```python
prediction = agent(
    question="10 kmは何mですか？"
)

print("回答:")
print(prediction.answer)

print("\ntool_calls:")
print(prediction.tool_calls)

print("\ntool_results:")
print(prediction.tool_results)

print("\ntrajectory:")
for key, value in prediction.trajectory.items():
    print(f"{key}: {value}")
```

`convert_units`が呼ばれ、最終回答が`10000 m`になれば、Agentは動作しています。

---

# Step 7：データセットを作る

今回は、次の14問を使います。

| 種類 | 問題数 |
|---|---:|
| 1ツール問題 | 6問 |
| 変換→計算 | 4問 |
| 変換→統計 | 4問 |

## セル10：`trainset`

```python
AVAILABLE_TOOLS = """
calculate_expression:
1つの数式を計算するツール。

analyze_numbers:
数値配列の平均、中央値、標準偏差などを分析するツール。

convert_units:
長さ、重さ、温度などの単位を変換するツール。
"""


def make_example(question):
    """質問とJudge用の情報を1件のExampleにまとめる。"""

    return dspy.Example(
        question=question,
        user_query=question,
        available_tools=AVAILABLE_TOOLS,
    ).with_inputs("question")


base_examples = [
    # 1ツール：数値計算
    make_example("25 * 16を計算してください"),
    make_example("2の10乗を求めてください"),

    # 1ツール：統計
    make_example("1, 2, 3, 4, 5の平均を求めてください"),
    make_example("2, 4, 6, 8, 10の中央値を求めてください"),

    # 1ツール：単位変換
    make_example("10 kmは何mですか？"),
    make_example("2 kgは何gですか？"),
]


multi_tool_examples = [
    # 2ツール：変換→計算
    make_example("2 kmをmに直し、350 mを足してください"),
    make_example("1.5 kgをgに直し、そこから250 gを引いてください"),
    make_example("3.4 kmをmに直し、600 mを足してください"),
    make_example("2.25 kgをgに直し、750 gを足してください"),

    # 2ツール：変換→統計
    make_example("1.2 km、800 m、1500 mの平均をmで求めてください"),
    make_example("500 cm、7 m、900 cmの中央値をcmで求めてください"),
    make_example("0.75 kg、1200 g、950 gの平均をgで求めてください"),
    make_example("1.8 km、2400 m、3 kmの中央値をmで求めてください"),
]


trainset = base_examples + multi_tool_examples

print("1ツール問題:", len(base_examples))
print("複数ツール問題:", len(multi_tool_examples))
print("合計:", len(trainset))
```

次のように表示されれば成功です。

```text
1ツール問題: 6
複数ツール問題: 8
合計: 14
```

`.with_inputs("question")`は、Agentへ渡す入力が`question`であることをDSPyへ伝えます。

`user_query`と`available_tools`は、後からJudgeが採点するときに使います。

---

# Step 8：複数ツール問題を確認する

## セル11：変換→計算を確認する

```python
example = trainset[6]

prediction = agent(
    question=example.question
)

print("質問:")
print(example.question)

print("\n回答:")
print(prediction.answer)

print("\ntool_calls:")
print(prediction.tool_calls)

print("\ntool_results:")
print(prediction.tool_results)
```

理想的な順番です。

```text
convert_units → calculate_expression
```

処理内容は、`2 km → 2000 m → 2000 + 350 → 2350 m`です。

## セル12：変換→統計を確認する

```python
example = trainset[10]

prediction = agent(
    question=example.question
)

print("質問:")
print(example.question)

print("\n回答:")
print(prediction.answer)

print("\ntool_calls:")
print(prediction.tool_calls)

print("\ntool_results:")
print(prediction.tool_results)
```

理想的な順番です。

```text
convert_units → analyze_numbers
```

処理内容は、`1.2 km → 1200 m → [1200, 800, 1500]の平均 → 約1166.67 m`です。

この時点で正しい順番にならなくても、その実行履歴とJudgeのfeedbackをGEPAが改善に使用します。

---

# Step 9：Judgeを定義する

## セル13：`ToolUseJudge`

```python
class ToolUseJudge(dspy.Signature):
    """
    AI Agentによるツール利用を厳格に評価してください。

    以下の観点を総合的に評価します。

    1. 選択したツールがユーザー要求に適しているか
    2. 外部ツールを使用する必要があったか
    3. ツールに渡した引数が適切か
    4. 最終回答がユーザー要求を満たしているか
    5. 不要または重複したツール呼び出しがないか

    複数の処理が必要な質問では、ツールの実行順序と、
    前のツール結果を次のツールへ正しく渡したかも確認してください。

    代替ツールでも同じ目的を安全かつ正確に達成できる場合は、
    必ずしも減点しないでください。

    feedbackは必ず日本語で記述してください。
    """

    user_query: str = dspy.InputField(
        desc="ユーザーが入力した質問"
    )

    available_tools: str = dspy.InputField(
        desc="Agentが利用可能だったツールの名前と説明"
    )

    tool_calls: str = dspy.InputField(
        desc="実際に呼び出したツール、引数、実行順序"
    )

    tool_results: str = dspy.InputField(
        desc="各ツールの実行結果"
    )

    final_answer: str = dspy.InputField(
        desc="Agentが生成した最終回答"
    )

    tool_selection_score: float = dspy.OutputField(
        desc="ツール選択の妥当性。0.0から1.0"
    )

    tool_necessity_score: float = dspy.OutputField(
        desc="ツール利用または非利用の妥当性。0.0から1.0"
    )

    argument_score: float = dspy.OutputField(
        desc="ツール引数と実行順序の妥当性。0.0から1.0"
    )

    task_success_score: float = dspy.OutputField(
        desc="ユーザー要求の達成度。0.0から1.0"
    )

    feedback: str = dspy.OutputField(
        desc="問題点とAgentの指示を改善するための日本語の助言"
    )
```

Judgeは、次の4項目を0.0から1.0で採点します。

| 出力 | 採点内容 |
|---|---|
| `tool_selection_score` | 適切なツールを選んだか |
| `tool_necessity_score` | ツール利用または非利用が妥当か |
| `argument_score` | 引数と実行順序が正しいか |
| `task_success_score` | 最終回答が要求を満たしたか |

`feedback`は、GEPAがプロンプトの改善点を考えるために使います。

---

# Step 10：Judge LMを設定する

## セル14：`judge_lm`と`judge`

```python
# 最低限動かすため、最初はStudent LMと同じdeploymentを使う。
# メンターからJudge専用deploymentを指定された場合は、右辺だけ変更する。
JUDGE_DEPLOYMENT = AZURE_OPENAI_DEPLOYMENT


judge_lm = dspy.LM(
    f"azure/{JUDGE_DEPLOYMENT}",
    api_key=AZURE_OPENAI_API_KEY,
    api_base=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION,
    temperature=0.0,
)


judge = dspy.ChainOfThought(
    ToolUseJudge
)
```

`judge_lm`は、Agentのツール利用と最終回答を採点するLMです。

ここでは`dspy.configure(lm=judge_lm)`を実行しません。全体設定を変更すると、AgentまでJudge LMへ切り替わるためです。

---

# Step 11：Judgeを呼び出す関数を作る

## セル15：`run_judge()`

```python
def run_judge(example, prediction):
    """質問とAgentの実行結果をJudge LMで採点する。"""

    with dspy.context(lm=judge_lm):
        return judge(
            user_query=example.user_query,
            available_tools=example.available_tools,
            tool_calls=prediction.tool_calls,
            tool_results=prediction.tool_results,
            final_answer=prediction.answer,
        )
```

`with dspy.context(lm=judge_lm)`の中だけJudge LMを使用します。この処理が終わると、通常のAgentは再びStudent LMを使います。

---

# Step 12：GEPA用metricを作る

## セル16：`tool_use_metric()`

```python
def tool_use_metric(
    example,
    prediction,
    trace=None,
    pred_name=None,
    pred_trace=None,
):
    """Judgeの4項目を重み付きで集約し、GEPAへ返す。"""

    judgment = run_judge(
        example,
        prediction,
    )

    tool_selection = float(
        judgment.tool_selection_score
    )

    tool_necessity = float(
        judgment.tool_necessity_score
    )

    argument = float(
        judgment.argument_score
    )

    task_success = float(
        judgment.task_success_score
    )

    score = (
        0.40 * tool_selection
        + 0.20 * tool_necessity
        + 0.15 * argument
        + 0.25 * task_success
    )

    return dspy.Prediction(
        score=score,
        feedback=judgment.feedback,
    )
```

総合scoreの重みです。

```text
ツール選択：40%
必要性　　：20%
引数・順序：15%
最終回答　：25%
合計　　　：100%
```

GEPAには、数値の`score`と文章の`feedback`を両方返します。

- `score`は、どのプロンプト候補が良いか比較するために使います。
- `feedback`は、プロンプトのどこを直すべきか考えるために使います。

---

# Step 13：Judgeとmetricを1問で確認する

## セル17：採点の動作確認

```python
example = trainset[6]

prediction = agent(
    question=example.question
)

judgment = run_judge(
    example,
    prediction,
)

print("tool_selection:", judgment.tool_selection_score)
print("tool_necessity:", judgment.tool_necessity_score)
print("argument:", judgment.argument_score)
print("task_success:", judgment.task_success_score)
print("feedback:", judgment.feedback)

metric_result = tool_use_metric(
    example,
    prediction,
)

print("score:", metric_result.score)
```

4つのスコア、feedback、総合scoreが表示されれば正常です。

feedbackが日本語になっていることも確認してください。

---

# Step 14：Baselineを測る

## 14.1 Baselineとは

Baselineは、GEPAで最適化する前の`agent`の点数です。「性能が悪いAgent」という意味ではなく、最適化後と比較するための基準です。

## 14.2 `score_only_metric`とは

`tool_use_metric()`は、次の2つを返します。

```text
score
feedback
```

GEPAには両方が必要ですが、通常の`dspy.Evaluate`では数値scoreだけを返す方が扱いやすいため、`score_only_metric()`でscoreだけを取り出します。

## セル18：Baseline評価

```python
def score_only_metric(
    example,
    prediction,
    trace=None,
):
    """通常評価用にtool_use_metricの総合scoreだけを返す。"""

    result = tool_use_metric(
        example,
        prediction,
        trace=trace,
    )

    return float(result.score)


evaluator = dspy.Evaluate(
    devset=trainset,
    metric=score_only_metric,
    num_threads=1,
    display_progress=True,
    display_table=True,
)


baseline_result = evaluator(
    agent
)

print(
    "Baseline:",
    baseline_result.score,
)
```

ここでは14問すべてを使って、最適化前のAgentを評価します。

例えば`98.71`と表示された場合、14問についてJudgeが出した総合scoreの平均が98.71%だったという意味です。

---

# Step 15：Reflection LMとGEPAを作る

Reflection LMは、Judgeのfeedbackを読み、Agentのプロンプトをどのように改善するか考えるLMです。

## セル19：`reflection_lm`と`optimizer`

```python
# 最低限動かすため、Judge LMと同じ設定を使う。
reflection_lm = judge_lm


optimizer = dspy.GEPA(
    metric=tool_use_metric,
    max_metric_calls=80,
    reflection_lm=reflection_lm,
    num_threads=4,
)
```

各引数の意味です。

| 引数 | 意味 |
|---|---|
| `metric` | Agentを採点し、feedbackを返す関数 |
| `max_metric_calls=80` | metricを呼び出せる最大回数 |
| `reflection_lm` | feedbackから改善指示を考えるLM |
| `num_threads=4` | 最大4件を並行処理する |

前回のようにmetric callが約766回にならないよう、今回は`max_metric_calls=80`を指定します。

`auto="light"`と`max_metric_calls`は同時に指定できないため、`auto="light"`は書きません。

レート制限エラーが出る場合だけ、`num_threads=1`へ変更してください。

---

# Step 16：GEPAで最適化する

## セル20：`compile()`

```python
optimized_agent = optimizer.compile(
    student=agent,
    trainset=trainset,
)
```

GEPAでは、おおまかに次の処理が繰り返されます。

1. 現在のAgentを`trainset`で実行します。
2. Judgeがscoreとfeedbackを返します。
3. Reflection LMが低得点の実行履歴とfeedbackを読みます。
4. Agentのプロンプトの改善案を作ります。
5. 改善候補を再度評価します。
6. 良い候補を`optimized_agent`として返します。

モデルの重みを更新する学習ではありません。Agentへ与える指示文を自動的に改善しています。

---

# Step 17：最適化後を評価する

## セル21：BaselineとGEPA後を比較する

```python
optimized_result = evaluator(
    optimized_agent
)

print(
    "Baseline:",
    baseline_result.score,
)

print(
    "GEPA後:",
    optimized_result.score,
)

print(
    "改善幅:",
    optimized_result.score
    - baseline_result.score,
)
```

実行例です。

```text
Baseline: 98.71
GEPA後: 99.71
改善幅: 1.0
```

この場合、14問の学習用データ上では、GEPA後のAgentが1.0ポイント高くなったと解釈できます。

ただし、同じ`trainset`を最適化と評価の両方に使っているため、これは最低限の動作確認です。未知の問題にも有効だと示すには、後から別のtestsetで評価します。

---

# Step 18：最適化後の複数ツール利用を確認する

## セル22：追加した8問の結果を表示する

```python
for example in trainset[6:]:
    prediction = optimized_agent(
        question=example.question
    )

    print("質問:")
    print(example.question)

    print("回答:")
    print(prediction.answer)

    print("tool_calls:")
    print(prediction.tool_calls)

    print("-" * 50)
```

`trainset[6:]`は、7問目以降、つまり追加した8問を意味します。

確認する点は次の2つです。

- 変換→計算問題で、`convert_units`の後に`calculate_expression`を使ったか。
- 変換→統計問題で、`convert_units`の後に`analyze_numbers`を使ったか。

このセルで何も表示されない場合は、次を実行してください。

```python
print(len(trainset))
```

`14`と表示される必要があります。`6`の場合は、セル10の14問版`trainset`をもう一度実行してください。

---

# Step 19：3つのLMの役割

| 変数 | 役割 |
|---|---|
| `lm` | Student LM。質問を解き、MCPツールを呼び出す |
| `judge_lm` | ツール利用、引数、最終回答を採点する |
| `reflection_lm` | feedbackから改善したプロンプトを考える |

今回は最低限動かすため、3つとも同じAzure deploymentを使える設定にしています。ただし、役割は異なります。

---

# Step 20：よくあるエラー

## `NameError`が出る

未実行のセルがあります。Kernelを再起動した場合は、セル1から順番に再実行してください。

## Azure deploymentのエラーが出る

次の値が、Azure上の実際のdeployment名と一致しているか確認してください。

```python
AZURE_OPENAI_DEPLOYMENT = "gpt-4o"
```

公開されているモデル名ではなく、Azure上で作成されたdeployment名が必要です。

## MCP接続エラーが出る

次を確認してください。

- `MCP_URL`が正しいか。
- `MCP_API_KEY`が正しいか。
- 社内ネットワークからMCPサーバーへ接続しているか。

## `prediction.tool_calls`が存在しない

生の`dspy.ReAct`ではなく、ラッパーを使って作った次のAgentを使用してください。

```python
agent = MCPToolAgent()
```

## `tool_calls`が空になる

生の履歴を確認します。

```python
print(prediction.trajectory)
```

LMがツールを使わずに回答した場合、`tool_calls`は空になります。これはJudgeとGEPAが評価する対象です。

## GEPAが長時間終わらない

セル19で次の設定になっているか確認してください。

```python
optimizer = dspy.GEPA(
    metric=tool_use_metric,
    max_metric_calls=80,
    reflection_lm=reflection_lm,
    num_threads=4,
)
```

`auto="light"`が残っている場合は削除してください。

## レート制限エラーが出る

`num_threads=1`へ変更して、optimizerを作り直します。

---

# Step 21：結果の解釈

単一ツール問題だけでは、Baselineが最初から約99点になり、GEPAが改善できる余地がほとんどありませんでした。

複数ツール問題では、次の能力も必要です。

```text
1つ目のツールを選ぶ
  ↓
結果を読み取る
  ↓
2つ目のツールを選ぶ
  ↓
前の結果を正しい引数として渡す
```

そのため、GEPAがプロンプトを改善した効果を確認しやすくなります。

発表では、次のように説明できます。

> 単一ツール問題だけではBaselineが非常に高く、最適化効果を確認しにくかった。そこで、単位変換の結果を数値計算または統計分析へ渡す複数ツール問題を追加した。これにより、ツール選択だけでなく、実行順序とツール間の結果の受け渡しも評価対象とした。GEPAによる最適化後、学習用データ上の評価スコアが改善した。

点数が改善した場合でも、Judgeの出力には多少の揺らぎがあります。そのため、点数と一緒に`tool_calls`も確認してください。

---

# Step 22：実行順チェックリスト

- [ ] セル1でStudent LMを設定した。
- [ ] セル2でMCP接続関数を作った。
- [ ] セル3から5で3つのツールを作った。
- [ ] セル6で`ToolQA`を定義した。
- [ ] セル7でtrajectory整理関数を作った。
- [ ] セル8で`agent`を作った。
- [ ] セル9でAgent単体を確認した。
- [ ] セル10で14問の`trainset`を作った。
- [ ] セル11と12で複数ツール問題を確認した。
- [ ] セル13で`ToolUseJudge`を定義した。
- [ ] セル14で`judge_lm`を作った。
- [ ] セル15で`run_judge()`を作った。
- [ ] セル16で`tool_use_metric()`を作った。
- [ ] セル17でJudgeの採点を確認した。
- [ ] セル18でBaselineを測った。
- [ ] セル19で`max_metric_calls=80`のGEPAを作った。
- [ ] セル20で最適化した。
- [ ] セル21で最適化前後を比較した。
- [ ] セル22で複数ツールの実行順序を確認した。

---

# GitHubへ載せる前の確認

次の情報は必ず削除または伏せてください。

- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_ENDPOINT`
- `MCP_URL`
- `MCP_API_KEY`
- 公開許可を得ていない社内ログ

質問データや評価結果を公開してよいかも、メンターへ確認してください。

---

# 参考資料

- [DSPy公式：ReAct](https://dspy.ai/api/modules/ReAct/)
- [DSPy公式：Tools with ReAct](https://dspy.ai/getting-started/react-and-tools/)
- [DSPy公式：GEPA](https://dspy.ai/api/optimizers/GEPA/overview/)
- [DSPy公式：GEPA in depth](https://dspy.ai/diving-deeper/gepa-in-depth/)
