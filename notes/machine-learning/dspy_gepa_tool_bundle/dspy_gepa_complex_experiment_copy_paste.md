# DSPy + GEPA 複雑な複数ツール実験：初心者向けコピペ実行ガイド

> この資料は、メンターから渡された `calculate_expression`、`analyze_numbers`、`convert_units` の3関数を定義した**直後**から使います。  
> Jupyter Notebookへ「セル1」から順にコピーし、説明を読みながら進めてください。

## この資料で解決したいこと

これまでの実験では、次の結果が得られました。

- 1ツール問題は簡単で、最適化前から正しいツールを使えたため、GEPAによる改善が見えなかった。
- 2ツール問題では、「1.2 km、800 m、1500 mの平均」の問題だけ、使用ツールが `calculate_expression` のみから `convert_units → calculate_expression` へ変わり、総合スコアが約1ポイント上がった。
- ただし、期待した手順は `convert_units → analyze_numbers` なので、これは**部分的な改善**である。

この結果は発表に使えます。重要なのは「改善した」と言い切ることではなく、次のことが分かった点です。

> 簡単な問題では最初から性能が高く、最適化効果を測りにくい。複数のツール結果を次のツールへ渡す問題にすると、エージェントの弱点とGEPAの変化が観察しやすくなる。

そこで新しい実験では、2〜4回のツール呼び出しが必要な問題を使います。

```text
例
kgをgへ変換する
    ↓
複数の値の中央値を求める
    ↓
中央値から100を引く

期待する順序
convert_units → convert_units → analyze_numbers → calculate_expression
```

新しい問題にすれば必ず改善するわけではありません。実験では、改善が出なかった結果も意味があります。この資料では、**改善が出るまで条件を変えるのではなく、最初に決めた条件で最適化前後を公平に比較する**方法を採用します。

---

# 0. 最初に知っておく言葉

## LLM

文章を受け取り、文章を生成するモデルです。今回のAzure OpenAIモデルがこれに当たります。

## ツール

LLMの外で処理を行うPython関数です。今回の3つは次の役割です。

| ツール | 役割 | 例 |
|---|---|---|
| `calculate_expression` | 1つの数式を計算 | `(2400 + 375) / 3` |
| `analyze_numbers` | 数値列の統計を計算 | 平均、中央値、標準偏差 |
| `convert_units` | 単位を変換 | km→m、kg→g |

## AIエージェント

質問を読み、どのツールをどの順番で使うかを判断し、ツール結果から回答を作るプログラムです。

## DSPy

LLMへの指示、入出力、評価方法、最適化方法をPythonの部品として組み立てるためのフレームワークです。

## ReAct

おおまかに次を繰り返すツール利用エージェントです。

```text
考える → ツールを選ぶ → 実行結果を見る → 次の行動を考える
```

DSPyの現在の安定版ReActでは、途中の行動が `trajectory` に保存され、1回の反復で1つのツールを呼びます。`finish` は処理を終了するための内部ツールです。

## metric

エージェントの実行結果を採点する関数です。GEPAは、この点数が高くなるように指示文を探します。

## Judge LM

エージェントとは別に、ツール選択、引数、回答を採点するLLMです。

## GEPA

実行結果、点数、文章のフィードバックを読み、DSPyプログラムの指示文を改善するOptimizerです。Azure OpenAIモデル自体の重みを学習し直すものではありません。

---

# 1. 今回の実験設計

## 1.1 最適化前後で比較するもの

```text
最適化前のagent
      と
GEPA後のoptimized_agent
```

同じ未知問題を両方へ渡し、次の3指標を比較します。

| 指標 | 見ていること |
|---|---|
| 平均総合点 | Judgeを含む総合評価 |
| ツール順序完全一致率 | 期待したツール列と完全に同じだった問題の割合 |
| タスク達成度 | 最終回答が正しいか |

## 1.2 4種類のデータ

| データ | 用途 | GEPAへ見せるか |
|---|---|---|
| `trainset` | 改善案を考えるための練習問題 | 見せる |
| `valset` | 改善候補を比較する問題 | 見せる |
| `testset` | 最後の発表用評価 | 見せない |
| `controlset` | 1ツール性能が悪化していないか確認 | 見せない |

`testset`をGEPAへ渡さない理由は、最後に初見問題で性能を測るためです。

## 1.3 今回の改善点

以前のJudgeは、正解のツール順序を明示的に知らず、代替手段にも比較的寛容でした。今回の制御実験では、各問題へ次の正解情報を付けます。

- `expected_tools`：期待するツール順序
- `reference_plan`：期待する処理手順
- `reference_answer`：期待する最終回答

ただし、`.with_inputs("question")`を使うので、エージェントが見るのは質問だけです。正解情報はJudgeとmetricだけが使います。

---

# 2. GitHubへ上げる前の注意

公開リポジトリには、次を絶対に含めないでください。

- Azure OpenAI APIキー
- MCP APIキー
- Azure endpoint
- MCPサーバーのURLや社内IPアドレス
- 社内画面のURLが写った写真
- 公開許可のない実行ログや質問データ

この資料は、メンター配布部分のキーやURLを含まない形で作っています。Notebookを公開する場合は、公開用コピーから秘密情報を削除してください。

---

# 3. Jupyter Notebookでの実行方法

1. メンター配布コードを上から実行する。
2. `calculate_expression`、`analyze_numbers`、`convert_units`が定義されたことを確認する。
3. 以下の「セル1」から順番に新しいCodeセルへコピーする。
4. 各セルを `Shift + Enter` で実行する。
5. エラーが出たら、そのセルより上を実行済みか確認する。

Jupyter上部が `Markdown` になっているとPythonコードを実行できません。Pythonを貼るセルは `Code` にしてください。

---

# 4. セル0：環境を確認する

```python
print("DSPy version:", dspy.__version__)
print("GEPAあり:", hasattr(dspy, "GEPA"))
print("計算ツールあり:", callable(calculate_expression))
print("統計ツールあり:", callable(analyze_numbers))
print("変換ツールあり:", callable(convert_units))
```

## 何をしているか

- `print(...)`は、括弧内の値を画面へ表示します。
- `dspy.__version__`は、インストールされているDSPyの版です。
- `hasattr(dspy, "GEPA")`は、`dspy`の中に`GEPA`があるか確認します。
- `callable(...)`は、その名前が関数として呼べるか確認します。

すべてのツールが`True`で、`GEPAあり`も`True`なら次へ進みます。

ReActの履歴形式はDSPyの版によって変わる可能性があります。今回のコードは`prediction.trajectory`を返す安定版ReActを前提にしているため、発表資料へDSPyのバージョンを記録してください。

---

# 5. セル1：importと共通設定

```python

import csv
import json
from statistics import mean

import dspy


# ReActが行動できる最大回数。
# 4個の外部ツール + finish を使えるよう、余裕を持って8にする。
MAX_ITERS = 8

# Judgeへ見せるツール説明。
AVAILABLE_TOOLS_TEXT = """
calculate_expression:
1つの数式を計算する。例: 128 * 1.08、(2000 + 350) / 3。

analyze_numbers:
数値配列を統計分析する。平均、中央値、標準偏差、分散などを求める。

convert_units:
長さ、重さ、温度、時間、データサイズなどの単位を変換する。
""".strip()

```

## 1行ずつ解説

### `import csv`

CSVファイルを保存するための標準ライブラリを読み込みます。CSVはExcelでも開きやすい表形式のファイルです。

### `import json`

Pythonのリストや辞書を、Judge LMが読みやすい文字列へ変換するために使います。

### `from statistics import mean`

複数の点数の平均を計算する`mean()`だけを読み込みます。

### `import dspy`

DSPyの`Signature`、`ReAct`、`Module`、`GEPA`などを使えるようにします。メンターのコードで既にimport済みでも、もう一度書いて問題ありません。

### `MAX_ITERS = 8`

`=`は、右側の値を左側の名前へ保存する記号です。

`MAX_ITERS`は、ReActが最大何回行動できるかです。今回の最長問題は外部ツール4回に加えて終了判断が必要なので、余裕を持って8にしています。8回必ずツールを呼ぶ意味ではありません。

### 三重引用符 `""" ... """`

複数行の文字列を書けます。`AVAILABLE_TOOLS_TEXT`はJudgeへ説明する3ツールの一覧です。

### `.strip()`

文字列の先頭と末尾にある余分な改行や空白を取り除きます。

---

# 6. セル2：Agentの入出力を定義する

```python
class ToolQA(dspy.Signature):
    """
    ユーザーの質問に正確に答えてください。
    必要な場合は、利用可能なツールを使ってください。
    ツールの実行結果を確認し、最終回答を作ってください。
    """

    question: str = dspy.InputField(
        desc="ユーザーが入力した質問"
    )

    answer: str = dspy.OutputField(
        desc="単位を含む、分かりやすい最終回答"
    )

```

## 何をしているか

`ToolQA`は計算を実行するクラスではありません。エージェントへ渡す入力と、エージェントから受け取る出力を決める**設計図**です。

### `class ToolQA(dspy.Signature):`

- `class`は、関連する設定をひとまとめにするための書き方です。
- `ToolQA`は自分で付けたクラス名です。
- `(dspy.Signature)`は、DSPyのSignatureを基に作るという意味です。
- 行末の`:`の後は、4文字分程度インデントします。

### docstring

クラス直下の三重引用符の文章が、最適化前の基本指示です。あえて細かいツール順序をすべて書いていません。最初から完璧な手順を人間が指示すると、GEPAが改善する余地がなくなるからです。

### `question: str = dspy.InputField(...)`

- `question`は入力の名前です。
- `: str`は文字列であることを表します。
- `InputField`はLMへ渡す入力です。
- `desc`は、この項目の説明です。

### `answer: str = dspy.OutputField(...)`

エージェントから受け取る最終回答です。単位を含む文章を返せるよう`str`にしています。

---

# 7. セル3：ReActの履歴を整理する

```python
def extract_tool_history(trajectory):
    """
    ReActのtrajectoryから、外部ツールの名前・引数・結果を取り出す。
    finishはReActの終了合図なので除外する。
    """

    trajectory = trajectory or {}

    tool_calls = []
    tool_results = []
    tool_names = []

    # tool_name_0、tool_name_1 ... の番号を集める。
    step_numbers = []

    for key in trajectory:
        if key.startswith("tool_name_"):
            step_text = key.removeprefix("tool_name_")

            try:
                step_numbers.append(int(step_text))
            except ValueError:
                # 想定外のキーは無視する。
                continue

    # 0、1、2 ... の順番に並べて履歴を取り出す。
    for step in sorted(step_numbers):
        tool_name = str(
            trajectory.get(f"tool_name_{step}", "")
        )

        if tool_name in ("", "finish", "submit"):
            continue

        tool_args = trajectory.get(
            f"tool_args_{step}",
            {},
        )

        observation = trajectory.get(
            f"observation_{step}"
        )

        tool_names.append(tool_name)

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

    return tool_calls, tool_results, tool_names

```

## `trajectory`とは

ReActが次のように動いたとします。

```text
1回目: convert_unitsを使う
2回目: analyze_numbersを使う
3回目: finishで終了する
```

`trajectory`には、概ね次の情報が入ります。

```python
{
    "tool_name_0": "convert_units",
    "tool_args_0": { ... },
    "observation_0": { ... },
    "tool_name_1": "analyze_numbers",
    "tool_args_1": { ... },
    "observation_1": { ... },
    "tool_name_2": "finish",
}
```

## 関数を少しずつ読む

### `def extract_tool_history(trajectory):`

`def`は関数を作るキーワードです。括弧の中の`trajectory`が、この関数へ渡す値です。

### `trajectory = trajectory or {}`

`trajectory`が空や`None`なら、空の辞書`{}`を使います。履歴がない場合でもエラーを起こしにくくするためです。

### `tool_calls = []`

`[]`は空のリストです。使ったツール情報を順番に追加します。

### `for key in trajectory:`

辞書のキーを1つずつ取り出す繰り返しです。

### `key.startswith("tool_name_")`

文字列が`tool_name_`で始まるか確認します。

### `.removeprefix("tool_name_")`

`tool_name_2`から`tool_name_`を取り除き、`2`だけにします。

### `try`と`except`

数字へ変換できる場合は`try`内を実行し、できない場合は`except`へ移動します。想定外のキーが混ざっても全体を止めないためです。

### `sorted(step_numbers)`

ステップ番号を小さい順に並べます。これにより、実行順序を保ってツール履歴を作れます。

### `trajectory.get(キー, 初期値)`

辞書から値を安全に取り出します。キーがなければ初期値を返すため、通常の`trajectory[キー]`よりエラーになりにくい書き方です。

### `if tool_name in ("", "finish", "submit"):`

`finish`と`submit`は終了操作で、MCPの外部ツールではありません。評価対象から除外します。

### `.append(...)`

リストの末尾へ値を追加します。

### `return tool_calls, tool_results, tool_names`

3つの結果を、関数を呼び出した場所へ返します。

---

# 8. セル4：GEPAで扱えるAgentを作る

```python
class MCPToolAgent(dspy.Module):
    """メンター配布の3ツールを利用するReAct Agent。"""

    def __init__(self):
        super().__init__()

        self.agent = dspy.ReAct(
            ToolQA,
            tools=[
                calculate_expression,
                analyze_numbers,
                convert_units,
            ],
            max_iters=MAX_ITERS,
        )

    def forward(self, question: str):
        result = self.agent(
            question=question
        )

        trajectory = getattr(
            result,
            "trajectory",
            {},
        )

        tool_calls, tool_results, tool_names = (
            extract_tool_history(trajectory)
        )

        return dspy.Prediction(
            answer=str(result.answer),
            tool_calls=json.dumps(
                tool_calls,
                ensure_ascii=False,
                default=str,
            ),
            tool_results=json.dumps(
                tool_results,
                ensure_ascii=False,
                default=str,
            ),
            tool_names=tool_names,
            trajectory=trajectory,
        )


# GEPAで最適化する前のAgent。
agent = MCPToolAgent()

```

## なぜ`dspy.Module`で包むのか

生のReActは最終回答と`trajectory`を返します。しかしJudgeには、整理した`tool_calls`や`tool_names`も渡したいです。そこで、ReActの実行と履歴整理を1つのModuleへまとめます。

### `class MCPToolAgent(dspy.Module):`

DSPyが最適化対象として扱えるプログラムを作ります。

### `def __init__(self):`

クラスから実物を作るときに最初に呼ばれる設定部分です。

### `self`

今作っているMCPToolAgent自身を表します。`self.agent`と書くと、クラスの別の関数からもReActを使えます。

### `super().__init__()`

親である`dspy.Module`の初期設定を実行します。定型的な行だと考えて構いません。

### `self.agent = dspy.ReAct(...)`

3つのPython関数を利用可能なツールとしてReActへ渡します。

### `def forward(self, question: str):`

Moduleを次のように呼んだときに実行される本体です。

```python
prediction = agent(question="質問")
```

### `getattr(result, "trajectory", {})`

`result`に`trajectory`があれば取り出し、なければ空の辞書を返します。

### `dspy.Prediction(...)`

最終回答だけでなく、ツール履歴も1つの結果へまとめます。

### `json.dumps(...)`

Pythonのリスト・辞書をJSON形式の文字列へ変換します。

- `ensure_ascii=False`：日本語をそのまま表示する。
- `default=str`：通常JSONへ変換できない値も文字列にする。

### `agent = MCPToolAgent()`

設計図から、実際に呼べる最適化前エージェントを作ります。

---

# 9. セル5：まずAgentを1問だけ動かす

```python
prediction = agent(
    question=(
        "1.2 km、800 m、1500 mの平均をmで求め、"
        "その値に250 mを足してください。"
    )
)

print("回答:")
print(prediction.answer)

print("\n使ったツール順序:")
print(prediction.tool_names)

print("\nツール呼び出し:")
print(prediction.tool_calls)

print("\nツール結果:")
print(prediction.tool_results)

print("\n生のtrajectory:")
print(prediction.trajectory)
```

## 期待する順序

```text
convert_units → analyze_numbers → calculate_expression
```

最適化前にこの順序にならなくても、ここでは修正しません。失敗した実行とJudgeの助言をGEPAへ渡すためです。

`prediction.tool_names`が空なら、エージェントは外部ツールを使わずに答えています。

---

# 10. セル6：1問分のデータを作る関数

```python
# ------------------------------------------------------------
# データセット
# ------------------------------------------------------------

def make_example(
    question,
    expected_tools,
    reference_answer,
    reference_plan,
    category,
):
    """1問分の入力と正解情報をdspy.Exampleへまとめる。"""

    return dspy.Example(
        question=question,
        user_query=question,
        available_tools=AVAILABLE_TOOLS_TEXT,
        expected_tools=expected_tools,
        reference_answer=reference_answer,
        reference_plan=reference_plan,
        category=category,
    ).with_inputs("question")


```

## 1行ずつ解説

### `make_example(...)`

毎回同じ項目を書く代わりに、1問分のExampleを簡単に作る関数です。

### `dspy.Example(...)`

質問と正解情報をひとまとめにします。

### `expected_tools`

今回の制御実験で期待する外部ツールの順序です。例えば次のリストです。

```python
[
    "convert_units",
    "analyze_numbers",
    "calculate_expression",
]
```

リストは順序を持つため、どのツールを何番目に使うべきか表せます。同じツールを2回使う場合は、同じ名前を2回書きます。

### `.with_inputs("question")`

エージェントへの入力は`question`だけ、と指定します。

`reference_answer`や`expected_tools`はExample内にありますが、エージェントには渡りません。Judgeとmetricだけが採点時に参照します。これは正解の漏洩を防ぐ重要な指定です。

---

# 11. セル7：trainsetを作る

```python
# GEPAが改善案を考えるために使う練習問題。
trainset = [
    make_example(
        question=(
            "1.2 km、800 m、1500 mの平均をmで求め、"
            "その値に250 mを足してください。"
        ),
        expected_tools=[
            "convert_units",
            "analyze_numbers",
            "calculate_expression",
        ],
        reference_answer="約1416.67 m",
        reference_plan=(
            "1.2 kmを1200 mへ変換する。"
            "[1200, 800, 1500]の平均1166.666...を求める。"
            "最後に250を足して1416.666... mとする。"
        ),
        category="3ツール: 変換→統計→計算",
    ),
    make_example(
        question=(
            "2.5 kg、750 g、1.25 kgの中央値をgで求め、"
            "そこから100 gを引いてください。"
        ),
        expected_tools=[
            "convert_units",
            "convert_units",
            "analyze_numbers",
            "calculate_expression",
        ],
        reference_answer="1150 g",
        reference_plan=(
            "2.5 kgを2500 g、1.25 kgを1250 gへ変換する。"
            "[2500, 750, 1250]の中央値1250を求め、"
            "100を引いて1150 gとする。"
        ),
        category="4ツール: 変換×2→統計→計算",
    ),
    make_example(
        question=(
            "1.5 kmと900 mの平均をmで求め、"
            "その平均値をkmへ変換してください。"
        ),
        expected_tools=[
            "convert_units",
            "analyze_numbers",
            "convert_units",
        ],
        reference_answer="1.2 km",
        reference_plan=(
            "1.5 kmを1500 mへ変換する。"
            "[1500, 900]の平均1200 mを求め、"
            "1200 mを1.2 kmへ変換する。"
        ),
        category="3ツール: 変換→統計→変換",
    ),
    make_example(
        question=(
            "20分、0.5時間、1800秒の平均を分で求めてください。"
        ),
        expected_tools=[
            "convert_units",
            "convert_units",
            "analyze_numbers",
        ],
        reference_answer="約26.67分",
        reference_plan=(
            "0.5時間を30分、1800秒を30分へ変換する。"
            "[20, 30, 30]の平均26.666...分を求める。"
        ),
        category="3ツール: 変換×2→統計",
    ),
    make_example(
        question=(
            "750 cm、8 m、0.095 kmの平均をmで求め、"
            "その値に1.5 mを足してください。"
        ),
        expected_tools=[
            "convert_units",
            "convert_units",
            "analyze_numbers",
            "calculate_expression",
        ],
        reference_answer="約38.33 m",
        reference_plan=(
            "750 cmを7.5 m、0.095 kmを95 mへ変換する。"
            "[7.5, 8, 95]の平均36.833...を求め、"
            "1.5を足して38.333... mとする。"
        ),
        category="4ツール: 変換×2→統計→計算",
    ),
    make_example(
        question=(
            "1.8 kg、650 g、1.05 kgの中央値をgで求め、"
            "その値を5で割ってください。"
        ),
        expected_tools=[
            "convert_units",
            "convert_units",
            "analyze_numbers",
            "calculate_expression",
        ],
        reference_answer="210 g",
        reference_plan=(
            "1.8 kgを1800 g、1.05 kgを1050 gへ変換する。"
            "[1800, 650, 1050]の中央値1050を求め、"
            "5で割って210 gとする。"
        ),
        category="4ツール: 変換×2→統計→計算",
    ),
    make_example(
        question=(
            "2.4 kmをmへ変換し、375 mを足した後、"
            "3で割ってください。"
        ),
        expected_tools=[
            "convert_units",
            "calculate_expression",
        ],
        reference_answer="925 m",
        reference_plan=(
            "2.4 kmを2400 mへ変換する。"
            "(2400 + 375) / 3を計算し、925 mとする。"
        ),
        category="2ツール: 変換→計算",
    ),
    make_example(
        question=(
            "1.75 kgをgへ変換し、325 gを引いた後、"
            "残りをkgへ変換してください。"
        ),
        expected_tools=[
            "convert_units",
            "calculate_expression",
            "convert_units",
        ],
        reference_answer="1.425 kg",
        reference_plan=(
            "1.75 kgを1750 gへ変換する。"
            "1750 - 325を計算して1425 gを得る。"
            "1425 gを1.425 kgへ変換する。"
        ),
        category="3ツール: 変換→計算→変換",
    ),
]


```

## trainsetの役割

GEPAが失敗例とfeedbackを読み、改善された指示を考えるための問題です。

問題は次の難易度を含みます。

- 2ツール：変換→計算
- 3ツール：変換→統計→計算
- 3ツール：変換→統計→変換
- 4ツール：変換を2回→統計→計算

## `question=("..." "...")`について

Pythonでは、括弧の中に隣接して書いた文字列は連結されます。

```python
question=(
    "前半"
    "後半"
)
```

は、`"前半後半"`という1つの文字列になります。長い行を読みやすく分けるための書き方です。

## 時間変換でエラーが出る場合

MCPツールが`分`、`時間`、`秒`の表記を受け付けない場合があります。その場合は、まず`convert_units`を単体で呼び、使える単位文字列をメンターへ確認してください。実験全体が止まる場合は、時間問題を一時的に外し、長さ・重さの問題だけで実行できます。

---

# 12. セル8：valsetを作る

```python
# GEPAが候補の良し悪しを比較する検証問題。
# trainsetとは別問題にしている。
valset = [
    make_example(
        question=(
            "2.3 km、600 m、1.1 kmの平均をmで求め、"
            "その値に125 mを足してください。"
        ),
        expected_tools=[
            "convert_units",
            "convert_units",
            "analyze_numbers",
            "calculate_expression",
        ],
        reference_answer="約1458.33 m",
        reference_plan=(
            "2.3 kmを2300 m、1.1 kmを1100 mへ変換する。"
            "[2300, 600, 1100]の平均1333.333...を求め、"
            "125を足して1458.333... mとする。"
        ),
        category="4ツール: 変換×2→統計→計算",
    ),
    make_example(
        question=(
            "45分、1.25時間、3600秒の平均を分で求め、"
            "その平均を時間へ変換してください。"
        ),
        expected_tools=[
            "convert_units",
            "convert_units",
            "analyze_numbers",
            "convert_units",
        ],
        reference_answer="1時間",
        reference_plan=(
            "1.25時間を75分、3600秒を60分へ変換する。"
            "[45, 75, 60]の平均60分を求め、"
            "60分を1時間へ変換する。"
        ),
        category="4ツール: 変換×2→統計→変換",
    ),
    make_example(
        question=(
            "500 cm、7 m、900 cmの中央値をcmで求め、"
            "その値に50 cmを足してください。"
        ),
        expected_tools=[
            "convert_units",
            "analyze_numbers",
            "calculate_expression",
        ],
        reference_answer="750 cm",
        reference_plan=(
            "7 mを700 cmへ変換する。"
            "[500, 700, 900]の中央値700を求め、"
            "50を足して750 cmとする。"
        ),
        category="3ツール: 変換→統計→計算",
    ),
    make_example(
        question=(
            "0.75 kg、1200 g、950 gの平均をgで求め、"
            "その平均をkgへ変換してください。"
        ),
        expected_tools=[
            "convert_units",
            "analyze_numbers",
            "convert_units",
        ],
        reference_answer="約0.9667 kg",
        reference_plan=(
            "0.75 kgを750 gへ変換する。"
            "[750, 1200, 950]の平均966.666... gを求め、"
            "kgへ変換して0.966666... kgとする。"
        ),
        category="3ツール: 変換→統計→変換",
    ),
]


```

## valsetの役割

GEPAは複数の指示文候補を作ります。`valset`は、その候補のどれが良いか比較するための問題です。

`trainset`と同じ問題を使うと、練習問題だけに合う指示を選ぶ危険があります。そこで、形式は似ているが数値が異なる問題を用意しています。

---

# 13. セル9：testsetを作る

```python
# 最後の報告に使う未知問題。
# GEPAのcompileには渡さない。
testset = [
    make_example(
        question=(
            "3.4 km、1250 m、0.8 kmの平均をmで求め、"
            "その値に400 mを足してください。"
        ),
        expected_tools=[
            "convert_units",
            "convert_units",
            "analyze_numbers",
            "calculate_expression",
        ],
        reference_answer="約2216.67 m",
        reference_plan=(
            "3.4 kmを3400 m、0.8 kmを800 mへ変換する。"
            "[3400, 1250, 800]の平均1816.666...を求め、"
            "400を足して2216.666... mとする。"
        ),
        category="4ツール: 変換×2→統計→計算",
    ),
    make_example(
        question=(
            "1.6 kg、900 g、1.1 kgの中央値をgで求め、"
            "そこから75 gを引いてください。"
        ),
        expected_tools=[
            "convert_units",
            "convert_units",
            "analyze_numbers",
            "calculate_expression",
        ],
        reference_answer="1025 g",
        reference_plan=(
            "1.6 kgを1600 g、1.1 kgを1100 gへ変換する。"
            "[1600, 900, 1100]の中央値1100を求め、"
            "75を引いて1025 gとする。"
        ),
        category="4ツール: 変換×2→統計→計算",
    ),
    make_example(
        question=(
            "30分、0.75時間、2700秒の平均を分で求め、"
            "その平均を時間へ変換してください。"
        ),
        expected_tools=[
            "convert_units",
            "convert_units",
            "analyze_numbers",
            "convert_units",
        ],
        reference_answer="約0.6667時間",
        reference_plan=(
            "0.75時間を45分、2700秒を45分へ変換する。"
            "[30, 45, 45]の平均40分を求め、"
            "40分を0.666666...時間へ変換する。"
        ),
        category="4ツール: 変換×2→統計→変換",
    ),
    make_example(
        question=(
            "250 cm、4.5 m、0.006 kmの平均をmで求め、"
            "その値に0.5 mを足してください。"
        ),
        expected_tools=[
            "convert_units",
            "convert_units",
            "analyze_numbers",
            "calculate_expression",
        ],
        reference_answer="約4.8333 m",
        reference_plan=(
            "250 cmを2.5 m、0.006 kmを6 mへ変換する。"
            "[2.5, 4.5, 6]の平均4.333...を求め、"
            "0.5を足して4.833... mとする。"
        ),
        category="4ツール: 変換×2→統計→計算",
    ),
    make_example(
        question=(
            "2.2 kmをmへ変換し、480 mを足した後、"
            "4で割ってください。"
        ),
        expected_tools=[
            "convert_units",
            "calculate_expression",
        ],
        reference_answer="670 m",
        reference_plan=(
            "2.2 kmを2200 mへ変換する。"
            "(2200 + 480) / 4を計算して670 mとする。"
        ),
        category="2ツール: 変換→計算",
    ),
    make_example(
        question=(
            "2.4 kgをgへ変換し、600 gを引いた後、"
            "残りをkgへ変換してください。"
        ),
        expected_tools=[
            "convert_units",
            "calculate_expression",
            "convert_units",
        ],
        reference_answer="1.8 kg",
        reference_plan=(
            "2.4 kgを2400 gへ変換する。"
            "2400 - 600を計算して1800 gを得る。"
            "1800 gを1.8 kgへ変換する。"
        ),
        category="3ツール: 変換→計算→変換",
    ),
    make_example(
        question=(
            "1200 m、1.8 km、950 mの中央値をmで求め、"
            "その中央値をkmへ変換してください。"
        ),
        expected_tools=[
            "convert_units",
            "analyze_numbers",
            "convert_units",
        ],
        reference_answer="1.2 km",
        reference_plan=(
            "1.8 kmを1800 mへ変換する。"
            "[1200, 1800, 950]の中央値1200 mを求め、"
            "1200 mを1.2 kmへ変換する。"
        ),
        category="3ツール: 変換→統計→変換",
    ),
    make_example(
        question=(
            "15分、0.25時間、1200秒の平均を分で求め、"
            "その値に5分を足してください。"
        ),
        expected_tools=[
            "convert_units",
            "convert_units",
            "analyze_numbers",
            "calculate_expression",
        ],
        reference_answer="約21.67分",
        reference_plan=(
            "0.25時間を15分、1200秒を20分へ変換する。"
            "[15, 15, 20]の平均16.666...を求め、"
            "5を足して21.666...分とする。"
        ),
        category="4ツール: 変換×2→統計→計算",
    ),
]


```

## testsetの役割

最終発表で最も重要な未知問題です。

```python
optimized_agent = optimizer.compile(
    student=agent,
    trainset=trainset,
    valset=valset,
)
```

この`compile()`には`testset`を渡しません。GEPAが一度も見ていない問題で最適化前後を比べます。

実験結果を見た後でtestsetの問題を入れ替え、良い結果だけを採用すると、公平な比較になりません。最初の実行結果を保存してください。

---

# 14. セル10：1ツール対照問題を作る

```python
# 1ツール問題で性能が悪化していないか確認する対照問題。
controlset = [
    make_example(
        question="37 * 19を計算してください。",
        expected_tools=["calculate_expression"],
        reference_answer="703",
        reference_plan="37 * 19をcalculate_expressionで計算する。",
        category="対照: 1ツール計算",
    ),
    make_example(
        question="3、7、14の平均を求めてください。",
        expected_tools=["analyze_numbers"],
        reference_answer="8",
        reference_plan="[3, 7, 14]をanalyze_numbersへ渡し、平均8を求める。",
        category="対照: 1ツール統計",
    ),
    make_example(
        question="3.5 kmは何mですか。",
        expected_tools=["convert_units"],
        reference_answer="3500 m",
        reference_plan="3.5 kmをconvert_unitsで3500 mへ変換する。",
        category="対照: 1ツール変換",
    ),
]


```

## なぜcontrolsetが必要か

複雑問題の性能が上がっても、簡単な1ツール問題で余計なツールを使うようになったら、実運用では困ります。

そのため、次の3つも最適化前後で確認します。

- 計算だけ
- 統計だけ
- 単位変換だけ

これは、改善と引き換えに既存性能が悪化していないかを見る簡単な回帰テストです。

---

# 15. セル11：問題数と正解順序を確認する

```python
print("trainset:", len(trainset))
print("valset:", len(valset))
print("testset:", len(testset))
print("controlset:", len(controlset))

for number, example in enumerate(trainset, start=1):
    print(number, example.category)
    print("  質問:", example.question)
    print("  期待:", " -> ".join(example.expected_tools))
```

期待される問題数です。

```text
trainset: 8
valset: 4
testset: 8
controlset: 3
```

### `len(...)`

リストに何件入っているか返します。

### `enumerate(..., start=1)`

リストの要素に、1から始まる番号を付けて取り出します。

### `" -> ".join(example.expected_tools)`

ツール名のリストを、矢印でつないだ1つの文字列へ変換します。

---

# 16. セル12：Judgeの入出力を定義する

```python
# ------------------------------------------------------------
# Judge（採点用LM）
# ------------------------------------------------------------

class ToolUseJudge(dspy.Signature):
    """
    AI Agentのツール利用を、参照計画と参照回答に照らして評価してください。

    評価観点:
    1. 期待されたツールを適切な順序で選んだか
    2. ツールを使う必要がある問題で、必要なツールを使ったか
    3. 各ツールへ渡した引数と、前段の結果の受け渡しが正しいか
    4. 最終回答が参照回答と一致し、単位も適切か
    5. 不要なツール呼び出しや重複がないか

    0.0は不適切、1.0は完全に適切です。
    feedbackには、良かった点と、次に改善すべき行動を具体的に書いてください。
    """

    user_query: str = dspy.InputField(
        desc="ユーザーの質問"
    )
    available_tools: str = dspy.InputField(
        desc="利用可能なツールの名前と説明"
    )
    expected_tools: str = dspy.InputField(
        desc="この実験で期待するツールの順序"
    )
    reference_plan: str = dspy.InputField(
        desc="期待する処理手順"
    )
    reference_answer: str = dspy.InputField(
        desc="期待する最終回答"
    )
    tool_calls: str = dspy.InputField(
        desc="実際のツール名、引数、実行順序"
    )
    tool_results: str = dspy.InputField(
        desc="実際のツール実行結果"
    )
    final_answer: str = dspy.InputField(
        desc="Agentが生成した最終回答"
    )

    tool_selection_score: float = dspy.OutputField(
        desc="ツール選択と順序の妥当性。0.0から1.0"
    )
    tool_necessity_score: float = dspy.OutputField(
        desc="ツール利用の必要性判断。0.0から1.0"
    )
    argument_score: float = dspy.OutputField(
        desc="引数と結果受け渡しの妥当性。0.0から1.0"
    )
    task_success_score: float = dspy.OutputField(
        desc="最終回答の正確さと要求達成度。0.0から1.0"
    )
    feedback: str = dspy.OutputField(
        desc="Agentの指示を改善するための具体的な日本語の助言"
    )


```

## AgentとJudgeの違い

| プログラム | 仕事 |
|---|---|
| `ToolQA`を使うAgent | 質問を解き、ツールを呼ぶ |
| `ToolUseJudge`を使うJudge | Agentの実行を採点する |

Judgeへは、質問だけでなく次も渡します。

- 利用可能だったツール
- 期待するツール順序
- 参照手順
- 参照回答
- 実際のツール呼び出し
- 実際のツール結果
- 実際の最終回答

Judgeは4つの0.0〜1.0の点数と、文章のfeedbackを返します。

## `InputField`と`OutputField`

- `InputField`：Judgeへ見せる情報
- `OutputField`：Judgeから受け取りたい採点結果

### なぜfeedbackが必要か

数値スコアだけでは、GEPAは「何が悪かったか」を詳しく理解しにくいです。

```text
悪い例: score = 0.62

良い例:
期待はconvert_units → analyze_numbersだったが、
calculate_expressionだけを使った。
変換結果を統計ツールへ渡す手順を指示へ追加するべきである。
```

GEPAは、このような文章の助言を使って新しい指示文を提案します。

---

# 17. セル13：Judge用LMを作る

```python
# まずは、すでに動いているAgent用deploymentをJudgeにも使う。
# メンターから別のJudge用deployment名を指定された場合だけ変更する。
JUDGE_DEPLOYMENT = AZURE_OPENAI_DEPLOYMENT

judge_lm = dspy.LM(
    f"azure/{JUDGE_DEPLOYMENT}",
    api_key=AZURE_OPENAI_API_KEY,
    api_base=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION,
    temperature=0.0,
)

judge = dspy.ChainOfThought(ToolUseJudge)


```

## 注意

`JUDGE_DEPLOYMENT`は、Azure上のdeployment名です。公開されている一般的なモデル名と同じとは限りません。

このコードでは、すでにAgentで動作した`AZURE_OPENAI_DEPLOYMENT`をそのまま使います。メンターから別のJudge用deploymentを指定された場合だけ変更してください。

### `temperature=0.0`

採点のばらつきを減らす設定です。完全に同じになる保証はありませんが、自由度を低くします。

### `judge = dspy.ChainOfThought(ToolUseJudge)`

`ToolUseJudge`の設計図を使って、採点を行うDSPyモジュールを作ります。

---

# 18. セル14：点数と順序を扱う補助関数

```python
def clamp_score(value):
    """値をfloatへ変換し、0.0から1.0の範囲へ収める。"""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0

    return max(0.0, min(1.0, number))


def sequence_position_score(expected_tools, actual_tools):
    """
    期待順序と実際の順序を同じ位置どうしで比較し、0.0〜1.0で返す。

    例:
    期待 [convert, analyze, calculate]
    実際 [convert, calculate]
    なら、先頭の1個だけ一致するので 1 / 3。
    """

    if not expected_tools and not actual_tools:
        return 1.0

    denominator = max(
        len(expected_tools),
        len(actual_tools),
        1,
    )

    matched = 0

    for expected, actual in zip(
        expected_tools,
        actual_tools,
    ):
        if expected == actual:
            matched += 1

    return matched / denominator


def make_sequence_feedback(expected_tools, actual_tools):
    """期待順序と実際の順序をGEPAが読みやすい文章へ変換する。"""

    expected_text = " → ".join(expected_tools) or "ツールなし"
    actual_text = " → ".join(actual_tools) or "ツールなし"

    if expected_tools == actual_tools:
        return (
            "ツール順序は参照計画と一致しました。"
            f" 順序: {actual_text}"
        )

    return (
        "ツール順序が参照計画と一致していません。"
        f" 期待: {expected_text}。"
        f" 実際: {actual_text}。"
        "前段のツール結果を確認してから、"
        "次に必要なツールへその値を渡す指示を強めてください。"
    )


```

## `clamp_score`

Judgeが返した値を`float`へ変換し、0.0〜1.0に収めます。

```python
max(0.0, min(1.0, number))
```

内側の`min`で1.0より大きい値を1.0へ、外側の`max`で0.0より小さい値を0.0へします。

## `sequence_position_score`

期待したツールと実際のツールを、同じ位置どうしで比較します。

```text
期待: convert → analyze → calculate
実際: convert → calculate
```

1番目だけ一致するため、`1 / 3 = 0.333...`です。

### `zip(expected_tools, actual_tools)`

2つのリストから同じ位置の要素をペアで取り出します。

### `matched += 1`

`matched = matched + 1`と同じ意味です。一致数を1増やします。

## `make_sequence_feedback`

期待順序と実際順序を文章にします。Judgeのfeedbackに加えてこの機械的なfeedbackをGEPAへ渡すため、Judgeが多少寛容でも、順序違いを明確に伝えられます。

---

# 19. セル15：Judgeを呼び、総合点を作る

```python
def run_judge(example, prediction):
    """1問の正解情報とAgentの実行結果をJudge LMへ渡す。"""

    expected_tools_text = " → ".join(
        example.expected_tools
    )

    with dspy.context(lm=judge_lm):
        return judge(
            user_query=example.user_query,
            available_tools=example.available_tools,
            expected_tools=expected_tools_text,
            reference_plan=example.reference_plan,
            reference_answer=example.reference_answer,
            tool_calls=getattr(
                prediction,
                "tool_calls",
                "[]",
            ),
            tool_results=getattr(
                prediction,
                "tool_results",
                "[]",
            ),
            final_answer=str(
                getattr(prediction, "answer", "")
            ),
        )


def score_prediction(example, prediction):
    """
    Judgeの評価と客観的な順序比較をまとめ、総合点を作る。

    メンターの大分類の重み:
    - ツール選択 40%
    - ツール必要性 20%
    - 引数 15%
    - タスク成功 25%
    """

    actual_tools = list(
        getattr(prediction, "tool_names", [])
    )
    expected_tools = list(example.expected_tools)

    judge_result = run_judge(
        example,
        prediction,
    )

    objective_sequence = sequence_position_score(
        expected_tools,
        actual_tools,
    )

    judge_selection = clamp_score(
        judge_result.tool_selection_score
    )

    # ツール選択は、機械的な順序比較とJudge判断を半分ずつ使う。
    tool_selection = (
        0.50 * objective_sequence
        + 0.50 * judge_selection
    )

    tool_necessity = clamp_score(
        judge_result.tool_necessity_score
    )
    argument = clamp_score(
        judge_result.argument_score
    )
    task_success = clamp_score(
        judge_result.task_success_score
    )

    total_score = (
        0.40 * tool_selection
        + 0.20 * tool_necessity
        + 0.15 * argument
        + 0.25 * task_success
    )

    sequence_exact = (
        expected_tools == actual_tools
    )

    feedback = (
        make_sequence_feedback(
            expected_tools,
            actual_tools,
        )
        + "\nJudgeの助言: "
        + str(judge_result.feedback)
    )

    return {
        "score": clamp_score(total_score),
        "tool_selection": tool_selection,
        "tool_necessity": tool_necessity,
        "argument": argument,
        "task_success": task_success,
        "sequence_score": objective_sequence,
        "sequence_exact": sequence_exact,
        "expected_tools": expected_tools,
        "actual_tools": actual_tools,
        "feedback": feedback,
    }


```

## `run_judge`

1問の正解情報とAgentの実行結果をJudge LMへ渡します。

### `with dspy.context(lm=judge_lm):`

このインデント内だけJudge用LMを使います。全体設定をJudgeへ変更しないため、通常のAgentは元のLMを使い続けます。

## `score_prediction`

1問分の採点をまとめて行う中心関数です。

### ツール選択点

機械的な順序比較とJudgeの判断を半分ずつ使います。

```python
tool_selection = (
    0.50 * objective_sequence
    + 0.50 * judge_selection
)
```

これにより、次の両方を利用します。

- 決めた順序と一致したかという客観的評価
- 引数や代替手順も考慮するJudgeの評価

### 総合点

メンターの大分類の重みを維持しています。

```text
ツール選択  40%
必要性      20%
引数        15%
タスク達成  25%
合計       100%
```

### `sequence_exact`

期待リストと実際リストが完全に同じなら`True`です。発表では、この完全一致率を総合点と別に報告します。

### 辞書 `{ ... }`

複数の採点結果へ名前を付けてひとまとめにします。例えば`details["score"]`で総合点を取り出せます。

---

# 20. セル16：GEPA用metricを作る

```python
def tool_use_metric(
    example,
    prediction,
    trace=None,
    pred_name=None,
    pred_trace=None,
    program_trace=None,
):
    """GEPAへ総合scoreと文章feedbackを返す。"""

    details = score_prediction(
        example,
        prediction,
    )

    return dspy.Prediction(
        score=details["score"],
        feedback=details["feedback"],
    )


```

## metricの5〜6個の引数

現在のGEPAでは、基本的に次の情報をmetricへ渡します。

- `example`：正解情報を持つExample
- `prediction`：Agentの実行結果
- `trace`：プログラム全体の実行記録
- `pred_name`：現在feedbackを求めているpredictor名
- `pred_trace`：そのpredictor部分の記録
- `program_trace`：対応版では、採点時のプログラム実行記録

この実験では、前2つだけを直接使います。残りを省略せず書くのは、GEPAがキーワード引数として渡しても受け取れるようにするためです。

## 戻り値

```python
dspy.Prediction(
    score=...,
    feedback=...,
)
```

- `score`：GEPAが候補を比較する数値
- `feedback`：GEPAが次の指示案を考えるための文章

通常の`dspy.Evaluate`は主にscoreを集計し、GEPAはscoreとfeedbackの両方を利用します。

---

# 21. セル17：1問だけ採点する

```python
example = trainset[0]

prediction = agent(
    question=example.question
)

details = score_prediction(
    example,
    prediction,
)

print("質問:", example.question)
print("期待ツール:", details["expected_tools"])
print("実際ツール:", details["actual_tools"])
print("Agent回答:", prediction.answer)
print("総合点:", details["score"] * 100)
print("順序スコア:", details["sequence_score"])
print("順序完全一致:", details["sequence_exact"])
print("feedback:")
print(details["feedback"])
```

## このセルで止めて確認すること

- Judgeの各出力をfloatへ変換できた。
- 総合点が0〜100の範囲で表示された。
- feedbackが日本語で具体的に出た。
- 実際のツール順序が表示された。

ここでエラーが出る状態のままGEPAを実行しないでください。GEPAはmetricを何度も呼ぶため、同じエラーが繰り返されます。

---

# 22. セル18：複数問題を評価してCSVへ保存する

```python
# ------------------------------------------------------------
# 評価・CSV保存
# ------------------------------------------------------------

def evaluate_program(
    program,
    dataset,
    label,
    csv_path,
):
    """複数問題を実行し、平均点とツール順序一致率を返す。"""

    rows = []

    print("\n" + "=" * 78)
    print(label)
    print("=" * 78)

    for number, example in enumerate(
        dataset,
        start=1,
    ):
        prediction = program(
            question=example.question
        )

        details = score_prediction(
            example,
            prediction,
        )

        row = {
            "number": number,
            "category": example.category,
            "question": example.question,
            "reference_answer": example.reference_answer,
            "agent_answer": prediction.answer,
            "expected_tools": " -> ".join(
                details["expected_tools"]
            ),
            "actual_tools": " -> ".join(
                details["actual_tools"]
            ),
            "sequence_exact": details["sequence_exact"],
            "sequence_score": details["sequence_score"],
            "tool_selection": details["tool_selection"],
            "tool_necessity": details["tool_necessity"],
            "argument": details["argument"],
            "task_success": details["task_success"],
            "total_score": details["score"],
            "feedback": details["feedback"],
        }

        rows.append(row)

        print(f"\n[{number}] {example.category}")
        print("質問:", example.question)
        print("期待順序:", row["expected_tools"])
        print("実際順序:", row["actual_tools"] or "ツールなし")
        print("回答:", prediction.answer)
        print("総合点:", f"{details['score'] * 100:.1f}")
        print(
            "順序完全一致:",
            "OK" if details["sequence_exact"] else "NG",
        )

    if not rows:
        raise ValueError("datasetが空です。")

    summary = {
        "label": label,
        "problem_count": len(rows),
        "average_score": 100 * mean(
            row["total_score"]
            for row in rows
        ),
        "exact_sequence_rate": 100 * mean(
            1.0 if row["sequence_exact"] else 0.0
            for row in rows
        ),
        "task_success_rate": 100 * mean(
            row["task_success"]
            for row in rows
        ),
    }

    fieldnames = list(rows[0].keys())

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\n--- 集計 ---")
    print("問題数:", summary["problem_count"])
    print(
        "平均総合点:",
        f"{summary['average_score']:.1f}",
    )
    print(
        "ツール順序完全一致率:",
        f"{summary['exact_sequence_rate']:.1f}%",
    )
    print(
        "タスク達成度の平均:",
        f"{summary['task_success_rate']:.1f}",
    )
    print("CSV保存先:", csv_path)

    return summary, rows


```

## この関数が行うこと

1. データセットから問題を1問ずつ取り出す。
2. Agentを実行する。
3. Judgeと順序比較で採点する。
4. 問題ごとの結果を画面へ表示する。
5. 平均値を計算する。
6. CSVファイルへ保存する。

## 初心者向けの重要部分

### `rows = []`

問題ごとの結果を入れる空のリストです。

### `for number, example in enumerate(...)`

問題を1問ずつ処理します。

### `row = { ... }`

1問の質問、回答、ツール順序、点数を辞書へまとめます。

### `f"{details['score'] * 100:.1f}"`

`f`から始まる文字列は、波括弧内へ変数の値を埋め込めます。`:.1f`は小数点以下1桁で表示します。

### `mean(...)`

全問題の点数を平均します。

### `with open(...) as file:`

CSVファイルを開き、処理が終わったら自動的に閉じます。

### `encoding="utf-8-sig"`

日本語CSVをWindows版Excelで開いたときに文字化けしにくい指定です。

### `return summary, rows`

集計値と問題別結果の2つを返します。

---

# 23. セル19：比較用の関数を作る

```python
def compare_summaries(before, after):
    """最適化前後の3指標を見やすく表示する。"""

    print("\n" + "=" * 78)
    print("最適化前後の比較")
    print("=" * 78)

    items = [
        ("平均総合点", "average_score", "点"),
        ("ツール順序完全一致率", "exact_sequence_rate", "%"),
        ("タスク達成度の平均", "task_success_rate", "点"),
    ]

    for label, key, unit in items:
        before_value = before[key]
        after_value = after[key]
        difference = after_value - before_value

        print(
            f"{label}: "
            f"{before_value:.1f}{unit} -> "
            f"{after_value:.1f}{unit} "
            f"(差 {difference:+.1f}{unit})"
        )


def compare_each_problem(
    before_rows,
    after_rows,
    csv_path,
):
    """問題ごとの改善幅を計算し、大きい順にCSVへ保存する。"""

    comparison_rows = []

    for before_row, after_row in zip(
        before_rows,
        after_rows,
    ):
        comparison_rows.append(
            {
                "number": before_row["number"],
                "category": before_row["category"],
                "question": before_row["question"],
                "before_score": 100 * before_row["total_score"],
                "after_score": 100 * after_row["total_score"],
                "score_difference": 100 * (
                    after_row["total_score"]
                    - before_row["total_score"]
                ),
                "expected_tools": before_row["expected_tools"],
                "before_tools": before_row["actual_tools"],
                "after_tools": after_row["actual_tools"],
                "before_answer": before_row["agent_answer"],
                "after_answer": after_row["agent_answer"],
            }
        )

    comparison_rows.sort(
        key=lambda row: row["score_difference"],
        reverse=True,
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(comparison_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(comparison_rows)

    print("\n改善幅が大きい問題:")

    for row in comparison_rows[:3]:
        print("-", row["question"])
        print(
            "  score:",
            f"{row['before_score']:.1f}",
            "->",
            f"{row['after_score']:.1f}",
        )
        print("  before:", row["before_tools"] or "ツールなし")
        print("  after :", row["after_tools"] or "ツールなし")

    print("比較CSV保存先:", csv_path)

    return comparison_rows


```

## `compare_summaries`

最適化前後の平均値と差を表示します。

```text
平均総合点: 72.1点 -> 80.4点 (差 +8.3点)
```

実際の数値は実行結果によって変わります。

## `compare_each_problem`

各問題の点数差を計算し、改善幅が大きい順に並べます。発表へ載せる代表例を選びやすくするためです。

### `lambda`

短い関数をその場で作る記法です。

```python
key=lambda row: row["score_difference"]
```

は「各行の`score_difference`を並べ替えの基準にする」という意味です。

### `reverse=True`

大きい順に並べます。

---

# 24. セル20：最適化前の結果を保存する

```python
baseline_test_summary, baseline_test_rows = evaluate_program(
    program=agent,
    dataset=testset,
    label="最適化前: 複雑な未知問題",
    csv_path="baseline_test_results.csv",
)

baseline_control_summary, baseline_control_rows = evaluate_program(
    program=agent,
    dataset=controlset,
    label="最適化前: 1ツール対照問題",
    csv_path="baseline_control_results.csv",
)
```

## 重要

この結果を先に保存してからGEPAを動かしてください。

生成されるファイルです。

```text
baseline_test_results.csv
baseline_control_results.csv
```

Jupyterの左側にあるファイル一覧から確認できます。

---

# 25. セル21：Reflection LMとGEPAを作る

```python
# Reflection LMはJudgeと同じdeploymentを使う。
reflection_lm = dspy.LM(
    f"azure/{JUDGE_DEPLOYMENT}",
    api_key=AZURE_OPENAI_API_KEY,
    api_base=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION,
    temperature=0.0,
)


# autoとmax_metric_callsは同時に指定しない。
optimizer = dspy.GEPA(
    metric=tool_use_metric,
    max_metric_calls=80,
    reflection_lm=reflection_lm,
    num_threads=1,
    seed=0,
)


```

## Reflection LM

Judgeが書いたfeedbackを読み、より良い指示文を提案するLMです。この例ではJudgeと同じAzure deploymentを使います。

## `dspy.GEPA(...)`

- `metric=tool_use_metric`：候補Agentの採点方法
- `max_metric_calls=80`：metricを呼ぶ最大回数
- `reflection_lm=reflection_lm`：改善案を考えるLM
- `num_threads=1`：一度に1件ずつ処理
- `seed=0`：GEPA内の乱数条件を固定しやすくする

### `auto`と`max_metric_calls`

次の2つは同時に指定しません。

```python
# 同時に書かない
# auto="light"
# max_metric_calls=80
```

今回は実行量を明示するため`max_metric_calls=80`だけを使います。

### なぜ最初は`num_threads=1`か

初心者がログを追いやすく、Azureのレート制限やMCPサーバーの同時呼び出し問題を避けやすいためです。正常に動き、メンターから許可がある場合にだけ4へ増やします。

### 80回で改善案がほとんど作られなかった場合

最初の80回の結果を保存し、条件を記録したうえで、別実験として`120`を試せます。良い結果が出るまで無計画に繰り返さないでください。

---

# 26. セル22：GEPAを実行する

```python
# GEPAへ渡すのはtrainsetとvalsetだけ。
# testsetは最後の評価まで見せない。
optimized_agent = optimizer.compile(
    student=agent,
    trainset=trainset,
    valset=valset,
)


```

## `compile()`の意味

通常のPythonコンパイルとは少し意味が違います。ここでは、GEPAが複数の指示候補を評価し、良かった候補を持つ新しいModuleを返します。

- `student=agent`：改善したい元のAgent
- `trainset=trainset`：改善案を考える問題
- `valset=valset`：候補を比較する問題

元の`agent`は最適化前の比較用として残り、結果は`optimized_agent`へ入ります。

この処理は時間がかかります。途中でNotebookのカーネルを再起動しないでください。

---

# 27. セル23：最適化後を評価する

```python
# 最適化後を、同じ未知問題・同じ採点方法で評価する。
optimized_test_summary, optimized_test_rows = evaluate_program(
    program=optimized_agent,
    dataset=testset,
    label="GEPA後: 複雑な未知問題",
    csv_path="optimized_test_results.csv",
)

optimized_control_summary, optimized_control_rows = evaluate_program(
    program=optimized_agent,
    dataset=controlset,
    label="GEPA後: 1ツール対照問題",
    csv_path="optimized_control_results.csv",
)


```

最適化前と同じ`testset`、同じ`controlset`、同じ採点方法を使います。

生成されるファイルです。

```text
optimized_test_results.csv
optimized_control_results.csv
```

この時点まで、`testset`はGEPAのcompileへ渡していません。

---

# 28. セル24：最適化前後を比較する

```python
# 集計値と各問題の変化を表示・保存する。
compare_summaries(
    baseline_test_summary,
    optimized_test_summary,
)

comparison_rows = compare_each_problem(
    baseline_test_rows,
    optimized_test_rows,
    csv_path="before_after_comparison.csv",
)


# 1ツール性能が悪化していないかも確認する。
print("\n1ツール対照問題の比較:")
compare_summaries(
    baseline_control_summary,
    optimized_control_summary,
)


```

生成される`before_after_comparison.csv`には、問題ごとの次が入ります。

- 最適化前の点数
- 最適化後の点数
- 点数差
- 期待したツール順序
- 最適化前のツール順序
- 最適化後のツール順序
- 最適化前後の回答

発表では、このCSVから代表的な1〜2問を選びます。

---

# 29. セル25：GEPAが変えた指示を確認する

```python
# GEPAが変更した可能性のある指示を確認する。
print("\n最適化後のpredictor:")

for name, predictor in optimized_agent.named_predictors():
    print("\npredictor名:", name)

    signature = getattr(
        predictor,
        "signature",
        None,
    )

    if signature is not None:
        print("instructions:")
        print(signature.instructions)
```

## 何を見るか

GEPA後の指示に、例えば次のような考え方が追加されているか確認します。

- 異なる単位を先にそろえる。
- 統計量を求める質問では`analyze_numbers`を使う。
- 前のツール結果を次のツール引数へ渡す。
- 最終単位への変換を忘れない。
- 不要な計算ツールで統計を代用しない。

表示方法はDSPyのバージョンや内部構造により異なることがあります。何も表示されない場合でも、最適化後のツール履歴と点数比較は続けられます。

---

# 30. 発表用に残す結果

実験終了後、次の値をスライドへ入力します。

| 指標 | 最適化前 | GEPA後 | 差 |
|---|---:|---:|---:|
| 平均総合点 | `[入力]` | `[入力]` | `[入力]` |
| ツール順序完全一致率 | `[入力]%` | `[入力]%` | `[入力]pt` |
| タスク達成度 | `[入力]` | `[入力]` | `[入力]` |
| 1ツール対照・平均総合点 | `[入力]` | `[入力]` | `[入力]` |

## 代表例の選び方

`before_after_comparison.csv`の先頭付近から、次を満たす問題を1問選びます。

- 最適化前後でツール列が変わった。
- 変更理由を10〜20秒で説明できる。
- 最適化後の方が期待列へ近い。
- 回答も悪化していない。

例として、次の形式でスライドへ載せます。

```text
質問：3.4 km、1250 m、0.8 kmの平均をmで求め、400 mを足す

期待：convert → convert → analyze → calculate
前　：calculate
後　：convert → convert → analyze → calculate
点数：[前] → [後]
```

実際の結果を使い、上の例をそのまま結果として書かないでください。

---

# 31. 結果の読み方

## A. 総合点と完全一致率が両方上がった

最も説明しやすい改善です。

> GEPA後は、異なる単位をそろえ、統計処理を行い、その結果を計算へ渡す順序が増えた。未知問題でもツール順序完全一致率が上がった。

## B. 総合点だけ上がり、完全一致率が変わらない

引数や最終回答が改善した可能性があります。

> ツール列の完全一致数は変わらなかったが、引数や中間結果の受け渡し、最終回答が改善し、Judgeの総合評価が上がった。

## C. 一部の問題だけ改善した

これまでの「1.2 km、800 m、1500 m」の結果と同じタイプです。

> GEPAによって単位変換を先に行う行動は追加されたが、統計ツールの選択までは改善せず、部分的な改善にとどまった。

部分的改善を隠さず、「どこまで改善したか」を分解して説明します。

## D. 改善しなかった

実験失敗と決めつける必要はありません。

考えられる理由です。

- 最適化前から正答しており、改善余地が少ない。
- 80回の探索量では不足した。
- Judgeのfeedbackが具体的でない。
- trainsetが少なく、失敗パターンを十分に含まない。
- ツールの戻り値形式が複雑で、次のツールへ渡せない。
- Agent用LMの能力上限や、ReActの実行制約が影響した。

結論例です。

> 問題を複雑化して評価余地は作れたが、今回の条件では未知問題の明確な改善は確認できなかった。今後はデータ数、Judgeの安定性、探索予算を個別に検証する必要がある。

## E. 複雑問題は改善したが、1ツール問題が悪化した

> 複数ツール利用を強く促す指示により、単純問題でも不要なツールを呼ぶ副作用が見られた。

このため、`controlset`の結果も必ず報告します。

---

# 32. 今回の実験で言えること・言えないこと

## 言えること

- DSPy ReActのツール利用履歴を取り出して評価できる。
- Judgeの点数とfeedbackをGEPAへ返す流れを実装できる。
- 複数ツールの順序を含む制御問題で、最適化前後を比較できる。
- 1ツール問題と複数ツール問題では、改善余地が異なる。

## この実験だけでは言えないこと

- すべてのAIエージェントでGEPAが必ず改善する。
- 今回の8問の改善が、実際の東芝業務全体でも同じように再現する。
- LLM Judgeの点数が人間評価と完全に一致する。
- コスト、応答時間、安全性を含めて実運用可能である。

発表では、「適用可能性を確認した」と「実運用で有効と証明した」を区別してください。

---

# 33. よくあるエラー

## `NameError: calculate_expression is not defined`

メンター配布コードを先に実行していません。カーネル再起動後は、Notebookの上から実行し直します。

## `AttributeError: module 'dspy' has no attribute 'GEPA'`

DSPyのバージョンがGEPAを含んでいない可能性があります。勝手に環境を更新せず、`dspy.__version__`をメンターへ伝えてください。

## `GEPA metric must accept ... arguments`

`tool_use_metric`を、この資料の6引数を受け取れる形で定義し直します。

```python
def tool_use_metric(
    example,
    prediction,
    trace=None,
    pred_name=None,
    pred_trace=None,
    program_trace=None,
):
    ...
```

## `auto`と`max_metric_calls`のエラー

どちらか一方だけにします。この資料では`max_metric_calls=80`を残し、`auto`を書きません。

## `prediction.tool_names`がない

生の`dspy.ReAct`ではなく、次を使っているか確認します。

```python
agent = MCPToolAgent()
```

## `tool_names`が空

`print(prediction.trajectory)`で生の履歴を見ます。最初に`finish`を選んだ場合、LLMがツールを使わず回答しています。これは今回の評価では順序不一致として扱います。

## 時間単位だけMCPエラーになる

ツールが受け付ける単位文字列を確認します。解決しない場合は、時間問題を外し、長さ・重さの問題で同じ構造を維持します。

## MCPタイムアウト

- MCPサーバーへ接続できるか確認する。
- `num_threads=1`にする。
- 1つのツールを単体で呼べるか試す。
- 何度も無制限に再実行しない。

## CSVが見つからない

次を実行してNotebookの現在位置を確認できます。

```python
import os
print(os.getcwd())
```

CSVは通常、このフォルダへ保存されます。

---

# 34. 実行順チェックリスト

- [ ] メンターの3ツールを定義した。
- [ ] DSPyバージョンとGEPAの有無を確認した。
- [ ] `ToolQA`を定義した。
- [ ] `extract_tool_history()`を定義した。
- [ ] `MCPToolAgent`と`agent`を作った。
- [ ] 1問だけAgentを動かし、`tool_names`を確認した。
- [ ] train/val/test/controlの問題数を確認した。
- [ ] `ToolUseJudge`と`judge_lm`を作った。
- [ ] 1問だけJudgeで採点できた。
- [ ] 評価関数を定義した。
- [ ] 最適化前のCSVを保存した。
- [ ] GEPAを1回実行した。
- [ ] 最適化後のCSVを保存した。
- [ ] `before_after_comparison.csv`を作った。
- [ ] 代表例を1〜2問選んだ。
- [ ] 1ツール対照問題が悪化していないか確認した。
- [ ] APIキーや社内URLを公開ファイルから削除した。

---

# 35. 発表の中心メッセージ

10分発表では、コードを細かく説明するより、次の流れを伝えます。

```text
簡単な1ツール問題
→ 最適化前から正解し、差が出なかった

2ツール問題
→ 1問だけ部分的な改善が見えた

より複雑な2〜4ツール問題
→ 期待順序を明示して、未知問題で最適化前後を評価した
```

発表で最も大切な一文です。

> ツール利用最適化を評価するには、最終回答だけでなく、使ったツールの種類、順序、中間結果の受け渡しを観測する必要がある。

---

# 36. 参考資料

今回の構成は、ユーザーの既存ノートとDSPy公式資料を基に整理しています。

## 既存ノート

- `dspy_tool_use_optimization_guide_annotated (1).md`
- `dspy_gepa_tool_optimization_experiment.md`
- `dspy_gepa_tool_optimization_experiment (1).md`
- `dspy_tool_use_gepa_complete_guide.md`

## DSPy公式

- ReAct: https://dspy.ai/diving-deeper/react/
- Metrics and evaluation: https://dspy.ai/diving-deeper/metrics-and-evaluation/
- GEPA in depth: https://dspy.ai/diving-deeper/gepa-in-depth/
- GEPA API: https://dspy.ai/api/optimizers/GEPA/overview/

---

# 付録：コードだけ欲しい場合

同梱の次のファイルには、この資料のPythonコードを1つにまとめています。

```text
dspy_gepa_complex_experiment_code.py
```

Jupyterでは、コード全体を一度に実行するより、このMarkdownのセル単位で動作確認しながら進める方が安全です。
