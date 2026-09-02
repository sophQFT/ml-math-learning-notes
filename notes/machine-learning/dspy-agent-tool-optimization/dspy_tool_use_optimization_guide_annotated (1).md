# DSPyによるツール利用最適化：最小実行ガイド

> メンターから与えられた3つのMCPツールの後ろへ、順番にコピーして実行するためのガイドです。  
> まず一連の処理を動かすことを優先し、発展的な分析コードは入れていません。

## 0. 今回行うこと

次の順番で動かします。

```text
3つのMCPツール（メンター配布済み）
        ↓
ReAct Agentを作る
        ↓
Agentのツール履歴を取り出す
        ↓
Judge LMがツール利用を採点する
        ↓
GEPAでAgentの指示を最適化する
        ↓
最適化前後のスコアを比較する
```

このMarkdownでは、次の3つがすでに動くことを前提にします。

```python
calculate_expression(...)
analyze_numbers(...)
convert_units(...)
```

メンター配布部分は変更せず、以降のコードをその後ろへ追加してください。

---

## 1. Agentの仕事を定義する

### セル1：`ToolQA`

```python
class ToolQA(dspy.Signature):
    """
    ユーザーの質問へ正確に回答してください。
    必要な場合は利用可能なツールから適切なものを選び、
    正しい引数を渡してください。
    不要または重複したツール呼び出しは避け、
    ツールの結果を使って最終回答を作成してください。
    """

    question: str = dspy.InputField(
        desc="ユーザーが入力した質問"
    )

    answer: str = dspy.OutputField(
        desc="ユーザーの要求を満たす最終回答"
    )
```

`ToolQA`は計算を行うクラスではなく、Agentへ「何を入力し、何を出力させるか」を伝える仕様書です。

- `question`：Agentへの入力です。
- `answer`：Agentの最終回答です。
- docstring：Agentへ与える最初の指示です。GEPAはこのような指示文を改善します。

---

## 2. ReAct Agentを作る

画像の`MCPToolAgent`では、`forward()`の途中が`...`になっていました。その部分では、ReActの`trajectory`からツール名、引数、結果を取り出します。

### 2.1 trajectoryとは

ReActを実行すると、最終回答に加えて途中の行動が記録されます。

```python
{
    "thought_0": "単位変換ツールを使う",
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

各キーの意味です。

| キー | 意味 |
|---|---|
| `thought_0` | 0番目の判断 |
| `tool_name_0` | 0番目に選んだツール |
| `tool_args_0` | 0番目のツールへ渡した引数 |
| `observation_0` | 0番目のツールの実行結果 |
| `finish` | ReActを終了する合図 |

Judgeへ渡したいのは、主に次の2つです。

- `tool_calls`：どのツールを、どの引数で呼んだか。
- `tool_results`：ツールから何が返ったか。

### セル2：trajectoryを整理する関数

```python
import json


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

この関数が行うことは3つだけです。

1. `tool_name_0`、`tool_name_1`などを探します。
2. 同じ番号の`tool_args_0`と`observation_0`を取り出します。
3. Judgeが読みやすいJSON文字列に変換します。

`finish`はMCPツールではなく終了合図なので除外します。`ensure_ascii=False`は、日本語を読みやすい状態で残す指定です。

### セル3：`MCPToolAgent`

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

画像にある`dspy.Module`を使ったAgentです。

- `self.agent`：実際にツールを選択・実行するReActです。
- `max_iters=5`：最大5回まで行動できます。必ず5回ツールを使う意味ではありません。
- `forward()`：`agent(question=...)`を実行したときに呼ばれる本体です。
- `dspy.Prediction`：回答とJudge用の履歴をまとめた戻り値です。

---

## 3. Agentだけを動かす

### セル4：動作確認

```python
prediction = agent(
    question=(
        "10, 20, 30, 40, 50の平均と標準偏差を"
        "計算してください"
    )
)

print("answer:")
print(prediction.answer)

print("\ntool_calls:")
print(prediction.tool_calls)

print("\ntool_results:")
print(prediction.tool_results)
```

ここでは、次を確認します。

- エラーなく実行できる。
- `analyze_numbers`が呼ばれている。
- `tool_calls`と`tool_results`が空ではない。
- 最終回答に平均と標準偏差が書かれている。

この時点では、まだ最適化していません。

---

## 4. GEPAで使う質問データを作る

Judgeは、Agentが利用できた3つのツールを知る必要があります。最小構成では、説明を文字列で用意します。

### セル5：データセット

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


trainset = [
    make_example("25 * 16を計算してください"),
    make_example("2の10乗を求めてください"),
    make_example("1, 2, 3, 4, 5の平均を求めてください"),
    make_example("2, 4, 6, 8, 10の中央値を求めてください"),
    make_example("10 kmは何mですか？"),
    make_example("2 kgは何gですか？"),
]
```

`.with_inputs("question")`は、Agentへ渡す入力が`question`だけであることを指定しています。`user_query`と`available_tools`は、後からJudgeが使います。

最初はこの6問で動作確認します。大量のデータ追加は、GEPAまで正常に動いてから行います。

---

## 5. メンター画像の`ToolUseJudge`を作る

### セル6：JudgeのSignature

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

    代替ツールでも同じ目的を安全かつ正確に達成できる場合は、
    必ずしも減点しないでください。
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
        desc="ツール引数の妥当性。0.0から1.0"
    )

    task_success_score: float = dspy.OutputField(
        desc="ユーザー要求の達成度。0.0から1.0"
    )

    feedback: str = dspy.OutputField(
        desc="問題点と、Agentの指示を改善するための具体的な助言"
    )
```

Judgeは、次の4項目を0.0～1.0で採点します。

| 出力 | 採点内容 |
|---|---|
| `tool_selection_score` | 適切なツールを選んだか |
| `tool_necessity_score` | ツールを使う必要があったか |
| `argument_score` | 引数が正しいか |
| `task_success_score` | 最終回答が要求を満たしたか |

`feedback`は、GEPAがAgentの指示を改善するために使う文章です。

---

## 6. Judge LMを作る

このセルは、`ToolUseJudge`の直後に置きます。

### セル7：`judge_lm`と`judge`

```python
# Azure上で作成されているJudge用deployment名を指定する。
# 名前が異なる場合は、メンターに指定された名前へ変更する。
JUDGE_DEPLOYMENT = "gpt-5.4-mini"


judge_lm = dspy.LM(
    f"azure/{JUDGE_DEPLOYMENT}",
    api_key=AZURE_OPENAI_API_KEY,
    api_base=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION,
    temperature=0.0,
)


judge = dspy.ChainOfThought(ToolUseJudge)
```

画像の`judge_lm`に対応するコードです。`JUDGE_DEPLOYMENT`は公開モデル名ではなく、実際にAzure上に作成されているdeployment名でなければ動きません。

`dspy.configure(lm=judge_lm)`は実行しません。全体設定をJudgeへ変えると、AgentまでJudge LMで動くためです。

---

## 7. Judgeを呼び出す関数を作る

### セル8：`run_judge()`

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

`with dspy.context(lm=judge_lm)`の中だけ、Judge用LMを使います。処理が終われば、通常のAgentは最初に設定したStudent LMを使います。

入力の対応は次のとおりです。

| Judgeへの入力 | 取得元 |
|---|---|
| 元の質問 | `example.user_query` |
| 利用可能ツール | `example.available_tools` |
| 実際のツール呼び出し | `prediction.tool_calls` |
| ツール結果 | `prediction.tool_results` |
| 最終回答 | `prediction.answer` |

---

## 8. メンター画像の評価関数を作る

### セル9：`tool_use_metric()`

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

重みはメンター画像の値をそのまま使っています。

```text
ツール選択：40%
必要性　　：20%
引数　　　：15%
最終回答　：25%
合計　　　：100%
```

例えば4スコアが`1.0、0.8、0.6、0.9`なら、総合点は次のとおりです。

```text
0.40×1.0 + 0.20×0.8 + 0.15×0.6 + 0.25×0.9
= 0.875
```

GEPAには数値の`score`だけでなく、文章の`feedback`も返します。数値で候補の良し悪しを比較し、文章で改善点を伝えるためです。

---

## 9. Judgeとmetricだけを試す

GEPAを動かす前に、1問だけ採点できるか確認します。

### セル10：採点の動作確認

```python
example = trainset[0]
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

4つのスコア、feedback、総合scoreが表示されれば次へ進みます。

---

## 10. Baselineを測る

Baselineは、GEPAで最適化する前の`agent`です。GEPA後と比較するため、先に点数を記録します。

### セル11：最適化前の評価

```python
def score_only_metric(
    example,
    prediction,
    trace=None,
):
    """通常評価用に総合scoreだけを返す。"""

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


baseline_result = evaluator(agent)

print("Baseline:", baseline_result.score)
```

`Baseline`は「性能が悪いAgent」という意味ではなく、最適化前の比較基準です。

ここでは最低限動かすため、同じ`trainset`で評価しています。最終発表用の厳密な実験では、後から別のtestsetを用意します。

---

## 11. Reflection LMとGEPAを作る

Reflection LMは、Judgeのfeedbackを読み、新しい指示を提案するLMです。

### セル12：`reflection_lm`と`optimizer`

```python
reflection_lm = dspy.LM(
    f"azure/{JUDGE_DEPLOYMENT}",
    api_key=AZURE_OPENAI_API_KEY,
    api_base=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION,
    temperature=0.0,
)


optimizer = dspy.GEPA(
    metric=tool_use_metric,
    auto="light",
    reflection_lm=reflection_lm,
    num_threads=4,
)
```

- `metric`：候補Agentを採点する関数です。
- `auto="light"`：小さな探索予算で試します。
- `reflection_lm`：feedbackから改善案を考えます。
- `num_threads=4`：最大4件を並行処理します。

エラーやレート制限が出る場合だけ、`num_threads=1`へ下げてください。

---

## 12. GEPAで最適化する

### セル13：`compile()`

```python
optimized_agent = optimizer.compile(
    student=agent,
    trainset=trainset,
)
```

これはメンター画像にある最小形です。

内部では、おおまかに次の処理が行われます。

1. `agent`を`trainset`で実行します。
2. Judgeがscoreとfeedbackを返します。
3. Reflection LMがfeedbackを読みます。
4. 新しい指示を提案します。
5. より良い候補を`optimized_agent`として返します。

モデルそのものの重みを学習しているわけではなく、主にAgentへ与える指示文を改善しています。

---

## 13. 最適化後を確認する

### セル14：1問実行

```python
optimized_prediction = optimized_agent(
    question="10 kmは何mですか？"
)

print(optimized_prediction.answer)
print(optimized_prediction.tool_calls)
print(optimized_prediction.tool_results)
```

### セル15：スコアを比較

```python
optimized_result = evaluator(
    optimized_agent
)

print("Baseline:", baseline_result.score)
print("GEPA後:", optimized_result.score)
```

`GEPA後`がBaselineより高ければ、今回のmetricに基づいてツール利用が改善したと判断できます。

ただし、質問数が6問だけなので、この結果だけで一般的な性能向上を証明したことにはなりません。まず動作確認を終えた後で、必要ならデータを増やします。

---

## 14. 3つのLMの違い

| 変数 | 役割 |
|---|---|
| `lm` | Student LM。質問を解き、MCPツールを呼ぶ |
| `judge_lm` | Agentのツール利用を採点する |
| `reflection_lm` | feedbackから改善した指示を考える |

`judge_lm`と`reflection_lm`は同じAzure deploymentを使っていますが、役割が異なるため変数を分けています。

---

## 15. エラーが出た場合

### `BadRequestError`やdeploymentエラー

```python
JUDGE_DEPLOYMENT = "gpt-5.4-mini"
```

が、Azure上の実際のdeployment名と一致しているか確認します。分からなければメンターへ確認してください。

### `prediction.tool_calls`がない

生の`dspy.ReAct`ではなく、次のAgentを使っているか確認します。

```python
agent = MCPToolAgent()
```

### `tool_calls`が`[]`になる

まず生の履歴を確認します。

```python
print(prediction.trajectory)
```

ツールを使う必要のない質問なら、空でも正常です。計算問題なのに空の場合は、ReActがツールを使わず回答した可能性があります。

### GEPAが遅い、途中で失敗する

まず並列数を1へ下げます。

```python
optimizer = dspy.GEPA(
    metric=tool_use_metric,
    auto="light",
    reflection_lm=reflection_lm,
    num_threads=1,
)
```

---

## 16. 実行順チェックリスト

- [ ] メンター配布の3ツールが単体で動く。
- [ ] セル1で`ToolQA`を定義した。
- [ ] セル2で`extract_tool_history()`を定義した。
- [ ] セル3で`agent`を作った。
- [ ] セル4でAgent単体が動いた。
- [ ] セル5で`trainset`を作った。
- [ ] セル6で`ToolUseJudge`を定義した。
- [ ] セル7で`judge_lm`と`judge`を作った。
- [ ] セル8で`run_judge()`を定義した。
- [ ] セル9で`tool_use_metric()`を定義した。
- [ ] セル10でJudgeの採点を確認した。
- [ ] セル11でBaselineを記録した。
- [ ] セル12でGEPAを作った。
- [ ] セル13で`compile()`した。
- [ ] セル15で最適化前後を比較した。

---

## 17. GitHubへ載せるときの注意

次の値は公開しないでください。

- Azure OpenAI APIキー
- Azure endpoint
- MCP APIキー
- 社内MCPサーバーのURLやIPアドレス
- 公開許可を得ていない社内ログ

GitHubへ載せるコードでは、値を空文字列や環境変数に置き換えます。

---

## 18. 今回の要点

今回の最小構成は、次の一文にまとめられます。

```text
ReActのツール履歴をJudgeが採点し、
scoreとfeedbackをGEPAへ渡して、
Agentの指示を改善する。
```

最初はセル1からセル15までを順番に実行し、動かすことを優先してください。詳細なデータ分割、応答時間、Pandas集計などは、一連の処理が動いた後に追加すれば十分です。

