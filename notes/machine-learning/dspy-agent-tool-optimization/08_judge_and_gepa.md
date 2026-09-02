# LLM審査員 + GEPA で最適化する

メンターの方針に沿った完全版。
**3つのツールの直後から、
上から順にコピペするだけ**で
発表まで到達できる。

---

## 0. 全体像

### なぜ Module が必要なのか

これが今回の方針の核心。

`run_judge` はこう書かれている。

```python
tool_calls=prediction.tool_calls,
tool_results=prediction.tool_results,
final_answer=prediction.answer,
```

つまり審査員は
**「どのツールを呼んだか」
「その結果は何か」
「最終回答は何か」**
の3つを読む。

ところが素の `dspy.ReAct` が返すのは
`answer` と `trajectory`（辞書）だけ。
`tool_calls` も `tool_results` も無い。

だから **ReAct を包んで、
審査員が読める形に変換する層**が要る。
それが `dspy.Module`。

```
ユーザーの質問
  ↓
MCPToolAgent（包む層）
  ↓
  dspy.ReAct → trajectory（辞書）
  ↓
  読みやすい文字列に変換
  ↓
Prediction(
  answer, tool_calls, tool_results)
  ↓
ToolUseJudge が採点
  ↓
tool_use_metric がスコアと助言を返す
  ↓
GEPA が指示文を書き換える
```

### 前回との違い

| 前回 | 今回 |
|---|---|
| ルールで採点 | **LLMが採点** |
| 正解ラベルが必須 | 正解なしでも採点可 |
| BootstrapFewShot | **GEPA** |
| ReActをそのまま | **Moduleで包む** |
| お手本を追加 | **指示文を書き換える** |

GEPA は
**実行トレースをLMに振り返らせ、
何が失敗したかを特定させて、
指示文を提案させる**最適化器。

`feedback` という自然言語の助言を
使えるのが最大の特徴で、
今回の judge はまさに
その助言を出すために作られている。

---

## STEP 1 — ツールを日本語docstringにする

メンターの3つのツールの
**docstringだけ**日本語に書き換える。
処理の中身は一切変えない。

### なぜ日本語にするのか

DSPy はツールの docstring を
**加工せずそのまま**
LMへの説明文として渡す。

質問が日本語なのに説明が英語だと、
LMは言語をまたいで対応づける
必要が出る。日本語に揃えたほうが
ツール選択が安定しやすい。

**これは発表の実験ネタにもなる。**
「英語版」と「日本語版」で
スコアを比べれば1つの結果になる。

```python
def calculate_expression(
    expression: str,
    precision: int = 6,
) -> dict:
    """単一の数式を計算する。

    四則演算や平方根など、
    1本の数式を計算するときに使う。
    複数の数値の平均や中央値には使わない
    （その場合は analyze_numbers）。

    Args:
        expression: 計算する数式。
            例: "128 * 1.08"、"sqrt(144)"
        precision: 小数点以下の桁数。
    """
    return call_mcp_tool(
        "calculate_expression",
        {
            "expression": expression,
            "precision": precision,
        },
    )


def analyze_numbers(
    values: list[float],
    operations: list[str] | None = None,
    second_values: list[float] | None = None,
    outlier_method: str = "none",
) -> dict:
    """数値の配列に対する統計量を計算する。

    平均・中央値・標準偏差・分散・
    最大値・最小値・合計などを求めるときに使う。
    単位の変換には使わない
    （その場合は convert_units）。

    Args:
        values: 対象の数値の配列。
            例: [1, 2, 3, 4, 5]
        operations: 求めたい統計量の名前の配列。
            例: ["mean", "median"]
        second_values: 2つの配列を比較する
            ときの、もう一方の配列。
        outlier_method: 外れ値の扱い。
            既定は "none"。
    """
    args = {
        "values": values,
        "outlier_method": outlier_method,
    }
    if operations is not None:
        args["operations"] = operations
    if second_values is not None:
        args["second_values"] = second_values
    return call_mcp_tool(
        "analyze_numbers", args)


def convert_units(
    value: float,
    from_unit: str,
    to_unit: str,
    category: str | None = None,
    precision: int = 6,
) -> dict:
    """単位を変換する。

    長さ・質量・温度・時間・データサイズの
    単位を変換するときに使う。
    計算そのものには使わない
    （その場合は calculate_expression）。

    Args:
        value: 変換したい数値。
        from_unit: 変換前の単位。
            例: "km"、"kg"、"celsius"
        to_unit: 変換後の単位。
            例: "m"、"g"、"fahrenheit"
        category: 単位の種類。
            例: "length"、"mass"。省略可。
        precision: 小数点以下の桁数。
    """
    args = {
        "value": value,
        "from_unit": from_unit,
        "to_unit": to_unit,
        "precision": precision,
    }
    if category is not None:
        args["category"] = category
    return call_mcp_tool(
        "convert_units", args)
```

> **`operations` の許容値は
> メンターに確認すること。**
> `"mean"` なのか `"average"` なのかで
> 引数エラーになる。
> 分かったら上のdocstringの
> 例のところを直す。

---

## STEP 2 — Signature を2つ作る

### 2-1. エージェント用

```python
import dspy


class ToolQA(dspy.Signature):
    """利用可能なツールを使って、
    ユーザーの質問に正確に答えてください。

    計算にはcalculate_expression、
    数値の配列の統計量にはanalyze_numbers、
    単位の変換にはconvert_unitsを使います。
    ツールが不要な質問には、
    ツールを呼ばずに直接答えてください。
    """

    question: str = dspy.InputField(
        desc="ユーザーの質問")
    answer: str = dspy.OutputField(
        desc="日本語の簡潔な回答。"
             "数値は回答の最後に置く。")
```

**この docstring が
GEPA に書き換えられる本体。**

あえて素っ気なく書いておき、
最適化後にどう変わったかを
比べるのが発表の山場になる。

### 2-2. 審査員用

メンターの指定どおり。

```python
class ToolUseJudge(dspy.Signature):
    """
    AI Agentによるツール利用を
    厳格に評価してください。

    以下の観点を総合的に評価します。

    1. 選択したツールがユーザー要求に
       適しているか
    2. 外部ツールを使用する必要があったか
    3. ツールに渡した引数が適切か
    4. 最終回答がユーザー要求を
       満たしているか
    5. 不要または重複したツール呼び出しが
       ないか

    代替ツールでも同じ目的を安全かつ
    正確に達成できる場合は、
    必ずしも減点しないでください。
    """

    user_query: str = dspy.InputField(
        desc="ユーザーが入力した質問"
    )
    available_tools: str = dspy.InputField(
        desc="Agentが利用可能だった"
             "ツールの名前と説明"
    )
    tool_calls: str = dspy.InputField(
        desc="実際に呼び出したツール、"
             "引数、実行順序"
    )
    tool_results: str = dspy.InputField(
        desc="各ツールの実行結果"
    )
    final_answer: str = dspy.InputField(
        desc="Agentが生成した最終回答"
    )

    tool_selection_score: float = \
        dspy.OutputField(
            desc="ツール選択の妥当性。"
                 "0.0から1.0"
        )
    tool_necessity_score: float = \
        dspy.OutputField(
            desc="ツール利用または非利用の"
                 "妥当性。0.0から1.0"
        )
    argument_score: float = \
        dspy.OutputField(
            desc="ツール引数の妥当性。"
                 "0.0から1.0"
        )
    task_success_score: float = \
        dspy.OutputField(
            desc="ユーザー要求の達成度。"
                 "0.0から1.0"
        )
    feedback: str = dspy.OutputField(
        desc="問題点と、Agentの指示を"
             "改善するための具体的な助言"
    )
```

### 4つの観点の意味

発表で説明できるようにしておく。

- **tool_selection**
  正しいツールを選べたか
- **tool_necessity**
  そもそも呼ぶべきだったか。
  **呼ばない判断も評価対象**
- **argument**
  引数が正しかったか
- **task_success**
  ユーザーの要求を満たしたか

`feedback` が
**GEPA に渡る自然言語の助言**。
これが今回の設計の要。

---

## STEP 3 — Agent を Module で包む

**方針で言われている
「Agentの定義を変更する」部分。**

```python
def format_trajectory(trajectory):
    """ReActの軌跡を、審査員が読める
    文字列2つに変換する。"""
    calls = []
    results = []
    i = 0

    while f"tool_name_{i}" in trajectory:
        name = trajectory[f"tool_name_{i}"]
        args = trajectory.get(
            f"tool_args_{i}", {})
        obs = trajectory.get(
            f"observation_{i}", "")

        # finish は終了用の組み込みなので除く
        if name != "finish":
            calls.append(
                f"{len(calls)+1}. {name} "
                f"(引数: {args})")
            results.append(
                f"{len(results)+1}. {name} "
                f"の結果: {obs}")
        i += 1

    if len(calls) == 0:
        calls = ["（ツールを1つも"
                 "呼び出していない）"]
        results = ["（実行結果なし）"]

    return "\n".join(calls), \
           "\n".join(results)


class MCPToolAgent(dspy.Module):
    """ReActを包んで、審査員が読める
    形式で結果を返すエージェント。"""

    def __init__(self, max_iters: int = 5):
        super().__init__()
        self.agent = dspy.ReAct(
            ToolQA,
            tools=[
                calculate_expression,
                analyze_numbers,
                convert_units,
            ],
            max_iters=max_iters,
        )

    def forward(self, user_query: str):
        result = self.agent(
            question=user_query)

        trajectory = getattr(
            result, "trajectory", {}) or {}
        tool_calls, tool_results = \
            format_trajectory(trajectory)

        return dspy.Prediction(
            answer=result.answer,
            tool_calls=tool_calls,
            tool_results=tool_results,
            trajectory=trajectory,
        )


program = MCPToolAgent()
```

### ここで押さえる2点

**① 引数の名前が `user_query`**

`run_judge` が
`example.user_query` を使うので、
データセットの入力名も
`user_query` に揃える。

`forward` の引数名と
データセットの入力名は
**必ず一致させる**。
ずれると `TypeError` になる。

**② `dspy.Prediction` で返す**

普通の辞書ではなく
`dspy.Prediction` で返すこと。
DSPy の最適化器はこれを前提にしている。

### 動作確認

```python
pred = program(
    user_query="25 * 16 を計算してください")

print("回答:", pred.answer)
print("--- tool_calls ---")
print(pred.tool_calls)
print("--- tool_results ---")
print(pred.tool_results)
```

`tool_calls` に
`1. calculate_expression (引数: ...)`
のような行が出ればOK。

**ここが通らないうちは先に進まない。**

---

## STEP 4 — データセット

審査員は `available_tools` も読むので、
ツールの説明文をまとめた文字列を
用意して、全部の例に持たせる。

```python
AVAILABLE_TOOLS_TEXT = """
- calculate_expression:
  単一の数式を計算する。
  例: "128 * 1.08"、"sqrt(144)"
- analyze_numbers:
  数値の配列の統計量を計算する。
  平均・中央値・標準偏差・最大値など。
- convert_units:
  単位を変換する。
  長さ・質量・温度・時間・データサイズ。
""".strip()


def ex(user_query, expected_answer,
       expected_tools):
    """例を1つ作る便利関数"""
    return dspy.Example(
        user_query=user_query,
        available_tools=AVAILABLE_TOOLS_TEXT,
        expected_answer=expected_answer,
        expected_tools=expected_tools,
    ).with_inputs("user_query")


# 学習用（GEPAに見せる）
trainset = [
    ex("25 * 16 を計算してください",
       "400", ["calculate_expression"]),
    ex("100 / 4 を計算してください",
       "25", ["calculate_expression"]),
    ex("1, 2, 3, 4, 5 の平均を"
       "求めてください",
       "3", ["analyze_numbers"]),
    ex("2, 4, 6, 8, 10 の中央値を"
       "求めてください",
       "6", ["analyze_numbers"]),
    ex("1 km は何 m ですか？",
       "1000", ["convert_units"]),
    ex("2 kg は何 g ですか？",
       "2000", ["convert_units"]),
    ex("5 km を3回走りました。"
       "合計は何 m ですか？",
       "15000",
       ["convert_units",
        "calculate_expression"]),
    ex("日本の首都はどこですか？",
       "東京", []),
]

# 評価用（GEPAには見せない）
testset = [
    ex("128 * 1.08 を計算してください",
       "138.24", ["calculate_expression"]),
    ex("(45 + 55) * 3 を計算してください",
       "300", ["calculate_expression"]),
    ex("1, 1, 2, 2, 100 の平均を"
       "求めてください",
       "21.2", ["analyze_numbers"]),
    ex("3, 7, 1, 9, 5 の最大値を"
       "求めてください",
       "9", ["analyze_numbers"]),
    ex("5000 m は何 km ですか？",
       "5", ["convert_units"]),
    ex("500 g は何 kg ですか？",
       "0.5", ["convert_units"]),
    ex("2 kg と 500 g の合計は"
       "何 g ですか？",
       "2500",
       ["convert_units",
        "calculate_expression"]),
    ex("虹は一般に何色と言われますか？",
       "7", []),
]

print("学習用", len(trainset), "件")
print("評価用", len(testset), "件")
```

### なぜ `expected_answer` を持たせるのか

**審査員はこれを見ない。**
`ToolUseJudge` の入力に
正解の欄が無いため。

それでも持たせるのは、
**あとで審査員の採点が
正しいか検算するため**。

LLM審査員は甘く付ける傾向があるので、
ルールベースの正誤と突き合わせると
「審査員は妥当か」を検証できる。

**これは発表で強い論点になる。**

### なぜ `expected_tools` を持たせるのか

同じく審査員は見ない。
こちらも検算用で、
「審査員が高得点を付けた回答で、
本当に正しいツールを
呼んでいたか」を確認できる。

---

## STEP 5 — judge_lm をどこに置くか

**質問の答え：ここ。**
Signature を定義したあと、
metric を定義する前。

```python
# 審査員が使うモデル。
# メンター指示のモデル名に変えること。
JUDGE_DEPLOYMENT = AZURE_OPENAI_DEPLOYMENT

judge_lm = dspy.LM(
    f"azure/{JUDGE_DEPLOYMENT}",
    api_key=AZURE_OPENAI_API_KEY,
    api_base=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION,
    temperature=0.0,
)

judge = dspy.ChainOfThought(ToolUseJudge)


def run_judge(example, prediction):
    with dspy.context(lm=judge_lm):
        return judge(
            user_query=example.user_query,
            available_tools=(
                example.available_tools),
            tool_calls=prediction.tool_calls,
            tool_results=(
                prediction.tool_results),
            final_answer=prediction.answer,
        )
```

### judge_lm の役割を理解する

DSPy には
**「今どのLMを使うか」という設定**が
1つある。
`dspy.configure(lm=lm)` で設定したもの。

エージェント本体はこれを使う。

しかし審査員には
**別のモデルを使いたい**。
そこで `dspy.context` を使う。

```python
with dspy.context(lm=judge_lm):
    # このブロックの中だけ judge_lm を使う
    judgment = judge(...)
```

ブロックを抜ければ
元のLMに戻る。

### なぜ審査員を別モデルにするのか

**採点者と被採点者が同じだと、
自分の答えに甘い点を付けやすい。**

メンターが別モデルを指定しているのは
このため。発表で必ず触れること。

`temperature=0.0` も重要で、
同じ回答に毎回同じ点が付かないと
比較できなくなる。

---

## STEP 6 — metric

```python
def to_float(value, default=0.0):
    """審査員の出力を安全に数値にする"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def tool_use_metric(
    example,
    prediction,
    trace=None,
    pred_name=None,
    pred_trace=None,
):
    judgment = run_judge(example, prediction)

    tool_selection = to_float(
        judgment.tool_selection_score)
    tool_necessity = to_float(
        judgment.tool_necessity_score)
    argument = to_float(
        judgment.argument_score)
    task_success = to_float(
        judgment.task_success_score)

    score = (
        0.40 * tool_selection
        + 0.20 * tool_necessity
        + 0.15 * argument
        + 0.25 * task_success
    )
    score = max(0.0, min(1.0, score))

    return dspy.Prediction(
        score=score,
        feedback=judgment.feedback,
    )
```

### 引数が5つある理由

GEPA は metric を呼ぶとき
`pred_name` と `pred_trace` も渡す。

- `pred_name`
  今どの部品を直そうとしているか
  （`agent.react` など）
- `pred_trace`
  その部品の実行記録

受け取るだけで、使わなくてよい。
**書いておかないとエラーになる。**

### 返り値が Prediction の理由

GEPA は
**スコアと助言の両方**を必要とする。

`score` で良し悪しを判断し、
`feedback` を読んで
「じゃあ指示文をこう直そう」と
提案する。

普通の数値を返す最適化器とは
ここが違う。

### 重みの意味

```
0.40 ツール選択    ← 最重視
0.25 タスク達成
0.20 ツールの要否
0.15 引数の妥当性
```

テーマが「ツール利用最適化」なので、
**回答の正しさよりツール選択を
重く置いている**。

この重み配分は
発表で必ず説明すること。
「何を重視するかを決めることが、
何を最適化するかを決めている」
という話につながる。

---

## STEP 7 — metric の動作確認

**必ずやること。**
LM呼び出しは2回だけなので安い。

```python
pred = program(
    user_query="25 * 16 を計算してください")

result = tool_use_metric(trainset[0], pred)

print("スコア:", result.score)
print("助言 :", result.feedback)
```

### 見るべきポイント

- スコアが `0.0` や `1.0` に
  張り付いていないか
- `feedback` が具体的か
  （「良い」だけなら審査員の
  docstring を調整する）

### 審査員の内訳も見る

```python
j = run_judge(trainset[0], pred)
print("推論   :", j.reasoning)
print("ツール選択:", j.tool_selection_score)
print("要否   :", j.tool_necessity_score)
print("引数   :", j.argument_score)
print("達成度 :", j.task_success_score)
print("助言   :", j.feedback)
```

`dspy.ChainOfThought` を使っているので
`reasoning`（採点理由）も出る。

**この出力をスライドに貼ると、
審査員が何を見ているかが
一目で伝わる。**

---

## STEP 8 — 評価関数

`dspy.Evaluate` は
数値を返す metric を前提にしている。
今回の metric は `Prediction` を返すので、
自分で回す関数を作る。

そのほうが
**4つの観点の内訳が取れて
発表に使いやすい。**

```python
import re


def clean(text):
    text = str(text).strip().replace(",", "")
    return text.translate(str.maketrans(
        "０１２３４５６７８９．",
        "0123456789."))


def is_correct(example, prediction):
    """ルールベースの正誤（審査員の検算用）"""
    expected = clean(example.expected_answer)
    actual = clean(
        getattr(prediction, "answer", ""))
    if actual == "":
        return False
    try:
        expected_num = float(expected)
    except ValueError:
        return expected in actual
    numbers = re.findall(
        r"-?\d+\.?\d*", actual)
    if len(numbers) == 0:
        return False
    return abs(float(numbers[-1])
               - expected_num) < 0.001


def used_tools(prediction):
    trajectory = getattr(
        prediction, "trajectory", {}) or {}
    tools = []
    i = 0
    while f"tool_name_{i}" in trajectory:
        name = trajectory[f"tool_name_{i}"]
        if name != "finish":
            tools.append(name)
        i += 1
    return tools


def evaluate(program, dataset, name):
    rows = []

    for i, example in enumerate(dataset, 1):
        try:
            pred = program(
                user_query=example.user_query)
        except Exception as e:
            print(f"  [{i}] エラー: {e}")
            pred = dspy.Prediction(
                answer="",
                tool_calls="（失敗）",
                tool_results="（失敗）",
                trajectory={},
            )

        j = run_judge(example, pred)

        row = {
            "query": example.user_query,
            "answer": getattr(
                pred, "answer", ""),
            "correct": is_correct(
                example, pred),
            "tools": used_tools(pred),
            "expected_tools":
                example.expected_tools,
            "selection": to_float(
                j.tool_selection_score),
            "necessity": to_float(
                j.tool_necessity_score),
            "argument": to_float(
                j.argument_score),
            "success": to_float(
                j.task_success_score),
            "feedback": j.feedback,
        }
        row["total"] = (
            0.40 * row["selection"]
            + 0.20 * row["necessity"]
            + 0.15 * row["argument"]
            + 0.25 * row["success"]
        )
        rows.append(row)

        mark = "○" if row["correct"] else "×"
        print(f"  [{i:2d}] {mark} "
              f"score={row['total']:.2f} "
              f"{row['tools']}")

    n = len(rows)

    def avg(key):
        return sum(r[key] for r in rows) / n

    summary = {
        "name": name,
        "total": avg("total"),
        "selection": avg("selection"),
        "necessity": avg("necessity"),
        "argument": avg("argument"),
        "success": avg("success"),
        "accuracy": sum(
            r["correct"] for r in rows) / n,
        "avg_calls": sum(
            len(r["tools"])
            for r in rows) / n,
        "rows": rows,
    }

    print(f"\n=== {name} ===")
    print(f"  総合スコア     : "
          f"{summary['total']:.3f}")
    print(f"  ツール選択     : "
          f"{summary['selection']:.3f}")
    print(f"  ツール要否     : "
          f"{summary['necessity']:.3f}")
    print(f"  引数の妥当性   : "
          f"{summary['argument']:.3f}")
    print(f"  タスク達成度   : "
          f"{summary['success']:.3f}")
    print(f"  実際の正解率   : "
          f"{summary['accuracy']:.3f}")
    print(f"  平均呼び出し数 : "
          f"{summary['avg_calls']:.2f}")
    return summary
```

---

## STEP 9 — 最適化「前」を測る

**これが無いと発表が成立しない。**

```python
before = evaluate(program, testset, "最適化前")
```

8件 × （エージェント + 審査員）で
LM呼び出しは30回程度。

数字を紙にメモしておくこと。

---

## STEP 10 — 失敗と助言を読む

```python
for r in before["rows"]:
    if r["total"] < 0.9 or not r["correct"]:
        print("質問:", r["query"])
        print("回答:", r["answer"])
        print("期待ツール:",
              r["expected_tools"])
        print("実際ツール:", r["tools"])
        print("助言:", r["feedback"])
        print("-" * 30)
```

**この `feedback` が
そのままGEPAに渡るもの。**

助言が具体的でなければ、
GEPA も具体的な改善ができない。
「もっと注意してください」のような
曖昧な助言しか出ていなければ、
`ToolUseJudge` の docstring に
一文足して調整する。

### 審査員の妥当性を確認する

```python
for r in before["rows"]:
    judge_ok = r["success"] >= 0.8
    if judge_ok != r["correct"]:
        print("食い違い:", r["query"])
        print("  審査員の達成度:",
              r["success"])
        print("  実際の正誤   :",
              r["correct"])
        print("  回答:", r["answer"])
```

**食い違いが出たら、それは発表ネタ。**

「LLM審査員は答えが間違っていても
高得点を付けることがあった」
という報告は、
LLM-as-a-judge の限界を示す
価値のある観察になる。

---

## STEP 11 — GEPA で最適化

### 実行前に必ず確認

GEPA は
**metricを呼ぶたびに審査員LMも呼ぶ。**

1回の評価あたり
「エージェント（ツール数回のHTTP）
＋ 審査員1回」がかかる。

`auto="light"` でも
数百回の実行になりうる。

- [ ] MCPサーバに数百リクエスト
      投げてよいかメンターに確認した
- [ ] Azure OpenAI の課金上限を確認した

**必ず聞いてから実行すること。**

### 実行

```python
reflection_lm = dspy.LM(
    f"azure/{JUDGE_DEPLOYMENT}",
    api_key=AZURE_OPENAI_API_KEY,
    api_base=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION,
    temperature=1.0,   # 提案は多様なほうがよい
    max_tokens=8000,
)

optimizer = dspy.GEPA(
    metric=tool_use_metric,
    auto="light",
    reflection_lm=reflection_lm,
    num_threads=4,
    track_stats=True,
)

optimized_agent = optimizer.compile(
    student=MCPToolAgent(),
    trainset=trainset,
    valset=trainset,
)

print("最適化 完了")
```

### reflection_lm とは

**指示文を書き直す担当のLM。**
役割が3つに分かれている。

| LM | 役割 | temperature |
|---|---|---|
| メインLM | ツールを使って答える | 0.0 |
| judge_lm | 採点と助言 | 0.0 |
| reflection_lm | 指示文を書き直す | 1.0 |

reflection_lm だけ
`temperature` を上げるのは、
**いろいろな案を出してほしい**から。

採点は毎回同じであってほしいので
judge は 0.0。

### 保存

```python
optimized_agent.save("optimized_gepa.json")
```

### 時間がかかりすぎるとき

`auto="light"` でも長い場合は
呼び出し回数を直接指定する。

```python
optimizer = dspy.GEPA(
    metric=tool_use_metric,
    max_metric_calls=60,   # ← 上限を固定
    reflection_lm=reflection_lm,
    num_threads=2,
    track_stats=True,
)
```

`auto` と `max_metric_calls` は
**どちらか一方だけ**指定する。

---

## STEP 12 — 最適化「後」を測る

**同じ testset で測ること。**

```python
after = evaluate(
    optimized_agent, testset, "最適化後")
```

---

## STEP 13 — 指示文の変化を見る

**発表の山場。**

```python
print("########## 最適化前 ##########")
for n, p in program.named_predictors():
    print(f"\n===== {n} =====")
    print(p.signature.instructions)

print("\n\n########## 最適化後 ##########")
for n, p in optimized_agent.named_predictors():
    print(f"\n===== {n} =====")
    print(p.signature.instructions)
```

部品は2つある。

- `agent.react`
  次にどのツールを呼ぶか決める
- `agent.extract.predict`
  最終回答を作る

### 必ず確認すること

DSPy は**ツールの説明文を
`agent.react` の指示文の中に
埋め込んでいる**。

GEPA がその指示文を書き換えるので、
**3つのツールの説明が
残っているか目で見る。**

```python
instr = optimized_agent.agent.react \
    .signature.instructions
for name in ["calculate_expression",
             "analyze_numbers",
             "convert_units"]:
    print(name, name in instr)
```

3つとも `True` ならOK。

**この確認をしたと発表で言うと、
仕組みを理解している証拠になる。**

---

## STEP 14 — 発表用の表

```python
keys = [
    ("total", "総合スコア"),
    ("selection", "ツール選択"),
    ("necessity", "ツール要否"),
    ("argument", "引数の妥当性"),
    ("success", "タスク達成度"),
    ("accuracy", "実際の正解率"),
    ("avg_calls", "平均呼び出し数"),
]

print("| 指標 | 最適化前 | 最適化後 |")
print("|---|---|---|")
for key, label in keys:
    print(f"| {label} | "
          f"{before[key]:.3f} | "
          f"{after[key]:.3f} |")
```

### ツールの選び方が変わった問題

```python
for b, a in zip(before["rows"],
                after["rows"]):
    if b["tools"] != a["tools"]:
        print("質問:", b["query"])
        print("  前:", b["tools"])
        print("  後:", a["tools"])
```

1〜2件スライドに載せると、
何が起きたかが一目で伝わる。

---

## 数字が改善しなかったら

**それでも発表は成立する。**

言い方の例:

> 3つのツールは役割が明確に
> 分かれており、最適化前の時点で
> ツール選択スコアが高かった。
> そのため改善の余地が小さかった。
>
> これは、ツール設計が適切であれば
> 自動最適化に頼らずとも
> 十分な性能が出ることを示している。
>
> 最適化が効くのは、ツールの役割が
> 曖昧な場合やツール数が多い場合だと
> 考えられる。

**立派な考察。**
実運用への示唆になっている。

---

## 10分発表の構成

### 1. タイトル（0:30）

### 2. 背景と課題（1:30）

- AIエージェントの品質は
  ツールをいつ・どう使うかで決まる
- その制御はプロンプトにある
- 改善は手作業で、属人的

### 3. DSPy とは（1:00）

> プロンプトを手で書く代わりに、
> 入出力の契約をコードで宣言し、
> プロンプトは最適化アルゴリズムに
> 書かせるフレームワーク

Signature / Module / Optimizer

### 4. 実装（1:30）

- Azure OpenAI + MCP の3ツール
- ReAct を `dspy.Module` で包み、
  審査員が読める形に変換
- 図: 質問 → Module → ReAct →
  MCP → 審査員 → GEPA

### 5. 評価の設計（2:30）★売り

**一番時間を使う。**

- LLM-as-a-judge を採用
- 4つの観点で採点
  （選択・要否・引数・達成）
- 重み 0.40 / 0.20 / 0.15 / 0.25
  → **ツール選択を最重視**
- 審査員は別モデル・temperature 0
  → 自己採点の甘さを避けるため
- 審査員の妥当性をルールベースの
  正誤と突き合わせて検証した

STEP 7 の審査員出力を貼る。

### 6. 最適化（1:00）

- GEPA を採用した理由
  = 自然言語の助言を使える唯一の
  最適化器で、審査員の feedback を
  そのまま活かせる
- 3つのLMの役割分担
  （本体 / 審査員 / 書き直し）

### 7. 結果（2:00）

- STEP 14 の表
- 指示文の変化
- ツール選択が変わった問題

### 8. 考察とまとめ（1:00）

**効きそうな場面**
- ツール数が多い
- ツールの役割が曖昧

**課題**
- 審査員LMのコストが
  最適化コストを倍増させる
- 審査員が甘い点を付ける場合がある
- 何を重視するかの重み設計が
  結果を決めてしまう

**まとめの一文**

> DSPy を使うと、人間の仕事は
> 「プロンプトを書くこと」から
> 「何を良しとするかを定義すること」
> に移る。
>
> 本実習で最も時間を使ったのは
> 評価指標の設計だった。

---

## 質疑で効く一言

**① 学習と評価を分けた**
> GEPA に見せたデータで評価すると
> 数字が嘘になるので、
> 8件ずつに分けました。

**② 審査員を検証した**
> LLM審査員が甘い可能性があるので、
> ルールベースの正誤と
> 突き合わせて確認しました。

**③ 審査員を別モデルにした**
> 採点者と被採点者が同じだと
> 自己採点が甘くなるためです。

**④ ツールを呼ばない判断も測った**
> tool_necessity_score で
> 評価しています。実運用では
> 呼びすぎがコストになるので。

---

## 困ったときの対処

**`TypeError: forward() got an
unexpected keyword argument`**
→ `forward` の引数名と
　`.with_inputs()` の名前が違う。
　両方 `user_query` に揃える

**`AttributeError: tool_calls`**
→ `MCPToolAgent` を通さず
　素の ReAct を渡している

**`ValueError: could not convert
string to float`**
→ 審査員が数値以外を返した。
　`to_float` を使っているか確認

**metricのスコアが全部同じ**
→ 審査員のdocstringが
　曖昧すぎる。観点を具体化する

**GEPAが終わらない**
→ `max_metric_calls=60` を指定し、
　`num_threads=2` に下げる

**GEPAでエラーが出る**
→ metric の引数が5つあるか
　（`pred_name`, `pred_trace`）確認

**回答が英語になる**
→ `ToolQA` の `answer` の desc に
　「日本語で」と書く

**15分詰まったらメンターに聞く。**
