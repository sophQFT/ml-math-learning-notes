# DSPyでツール利用を評価・最適化する実習ガイド

この資料は、PythonとDSPyの初学者が、実習中に上から順番にコードをコピーし、説明を読みながら進めるための手順書です。

今回のテーマは、次の一文で説明できます。

> **AIエージェントが質問に合ったツールを選べるかを評価し、DSPyで改善できるか確かめる。**

使えるツールは次の3つです。

| ツール | 用途 | 例 |
|---|---|---|
| `calculate_expression` | 1つの数式を計算する | `128 * 1.08` |
| `analyze_numbers` | 平均値・中央値などを計算する | `1, 2, 3 の平均` |
| `convert_units` | 単位を変換する | `1 km → m` |

---

## 最初に：GitHubへ上げるときの注意

> [!CAUTION]
> **実習先の許可がないコードや接続情報を、個人のGitHubへ上げないでください。**

特に、次の情報は載せないでください。

- APIキー
- MCPの認証キー
- 社内IPアドレス
- 社内URL
- Azure OpenAIのエンドポイント
- メンターから渡された非公開コード
- 実際の質問・回答データのうち、社外秘に当たるもの

このMarkdownでは、`class ToolQA(dspy.Signature):`より前の接続コードを扱いません。

---

# 0. この資料の使い方

## `class ToolQA`より前は「道具を準備する部分」

メンターから渡された部分は、ひとまず次のように理解すれば十分です。

```text
calculate_expression という電卓を使えるようにする
analyze_numbers という統計ツールを使えるようにする
convert_units という単位変換ツールを使えるようにする
```

その中では、Azure OpenAIやMCPサーバーとの通信が行われています。しかし今回、自分が主に考えるのは、その後の部分です。

```text
質問
  ↓
AIエージェントが使うツールを選ぶ
  ↓
選んだツールを実行する
  ↓
最終回答を出す
```

この資料では、次の順番で進めます。

1. エージェントを作る
2. 実際に使ったツールを確認する
3. 練習問題と試験問題を用意する
4. 回答とツール選択を採点する
5. 最適化前の性能を測る
6. DSPyで最適化する
7. 最適化後の性能を測る
8. 10分発表にまとめる

---

# 1. 先に知っておくPythonの最小知識

全部を暗記する必要はありません。コードを読むときに、この表へ戻って確認してください。

| 書き方 | 意味 | 例 |
|---|---|---|
| `# ...` | コメント。Pythonは実行しない | `# 計算問題` |
| `=` | 右側の値を左側の名前へ入れる | `x = 10` |
| `==` | 左右が同じか比べる | `x == 10` |
| `:` | この次から処理のまとまりが始まる | `if x == 10:` |
| インデント | 行の先頭を4文字ほど下げ、同じまとまりを表す | 関数の中など |
| `()` | 関数を呼ぶ、または引数を受け取る | `print(x)` |
| `[]` | 複数の値を並べたリスト | `tools = ["a", "b"]` |
| `{}` | 名前と値を組にした辞書 | `{"score": 100}` |
| `.` | オブジェクトが持つ値や機能へアクセスする | `pred.answer` |
| `def` | 関数を作る | `def add(a, b):` |
| `return` | 関数の結果を呼び出し元へ返す | `return a + b` |
| `if` | 条件が成立したときだけ実行する | `if x > 0:` |
| `for` | データを1つずつ繰り返し処理する | `for x in values:` |
| `True` | 条件が正しい | `1 == 1` |
| `False` | 条件が正しくない | `1 == 2` |
| `None` | 値がないことを表す | 数字を見つけられなかった場合など |

## インデントは特に重要

Pythonでは、行の先頭の空白に意味があります。

正しい例：

```python
def say_hello():
    print("こんにちは")
```

間違った例：

```python
def say_hello():
print("こんにちは")
```

関数の中にある行は、通常4文字分右へ下げます。

---

# 2. 今回の実験ルール

実際のサービスでは、簡単な計算をLLM自身に任せる設計もあります。

しかし今回は、ツール選択能力を分かりやすく評価するため、次のルールにします。

> **計算・統計・単位変換の問題では、問題が簡単でも対応するツールを1回使う。**

例えば、次の質問を考えます。

```text
128 * 1.08 を計算してください
```

LLMが自力で`138.24`と答えた場合、回答自体は正解です。しかし`calculate_expression`を使っていないため、今回の実験では次の判定にします。

```text
回答判定：OK
ツール判定：NG
総合判定：NG
```

測定する指標は次の3つです。

| 指標 | 意味 |
|---|---|
| 回答正解率 | 最終的な数値が正しい割合 |
| ツール選択正解率 | 正しいツールを1回使った割合 |
| 両方の成功率 | 回答とツール選択の両方が正しい割合 |

この3つを、DSPyによる最適化の前後で比較します。

---

# 3. Step 1：エージェントを作る

メンターから渡されたコードを先に実行し、次の3つの関数が使える状態にしてください。

```python
calculate_expression
analyze_numbers
convert_units
```

その後、次のコードをコピーします。`class ToolQA`は行頭から書き始めてください。

## コピーするコード

```python
class ToolQA(dspy.Signature):
    """
    質問に答えてください。

    1つの数式を計算する質問では、
    calculate_expressionを必ず1回使ってください。

    平均値や中央値を求める質問では、
    analyze_numbersを必ず1回使ってください。

    単位変換の質問では、
    convert_unitsを必ず1回使ってください。

    関係のないツールは使わないでください。
    """
    question: str = dspy.InputField()
    answer: str = dspy.OutputField(
        desc="最終的な答えの数値だけを返す"
    )


agent = dspy.ReAct(
    ToolQA,
    tools=[
        calculate_expression,
        analyze_numbers,
        convert_units,
    ],
    max_iters=4,
)
```

## 1行ずつ解説

### `class ToolQA(dspy.Signature):`

```python
class ToolQA(dspy.Signature):
```

- `class`は、関連する情報をまとめた「型」や「設計図」を作る書き方です。
- `ToolQA`は、自分で付けた名前です。ここでは「ツールを使ってQuestion Answeringする仕事」という意味です。
- `dspy.Signature`は、DSPyで入出力を定義するための土台です。
- 最後の`:`は、「ここからToolQAの中身が始まる」という意味です。

### 三重引用符の部分

```python
    """
    質問に答えてください。
    ...
    """
```

これは`docstring`と呼ばれる説明文です。この場合は、単なる人間向けのメモではなく、AIエージェントへの指示として使われます。

ここで次のルールを伝えています。

```text
数式計算    → calculate_expression
平均・中央値 → analyze_numbers
単位変換    → convert_units
```

### 入力欄

```python
    question: str = dspy.InputField()
```

- `question`は入力の名前です。
- `: str`は、「文字列として扱う予定」という型のヒントです。
- `dspy.InputField()`は、「これはAIへの入力欄です」とDSPyへ伝えます。

例：

```text
question = "1 km は何 m ですか"
```

### 出力欄

```python
    answer: str = dspy.OutputField(
        desc="最終的な答えの数値だけを返す"
    )
```

- `answer`は出力の名前です。
- `dspy.OutputField()`は、「これはAIに作ってほしい出力欄です」とDSPyへ伝えます。
- `desc=...`は、出力形式についての追加説明です。
- 今回は採点を簡単にするため、「答えの数値だけ」を返すよう指示します。

期待する出力：

```text
138.24
```

なるべく避けたい出力：

```text
128 * 1.08 の答えは 138.24 です
```

### `agent = ...`

```python
agent = dspy.ReAct(
```

- `agent`という名前の変数を作っています。
- 右側で作ったReActエージェントを、左側の`agent`へ保存します。
- 以後は、`agent(question="...")`と書けば質問できます。

### `ToolQA`

```python
    ToolQA,
```

ReActへ、先ほど作った仕事の定義を渡しています。

```text
入力はquestion
出力はanswer
ツール選択ルールはdocstringに書いてある
```

という情報を使います。

### `tools=[...]`

```python
    tools=[
        calculate_expression,
        analyze_numbers,
        convert_units,
    ],
```

- `[]`はPythonのリストです。
- 今回エージェントが選べる3つのツールを並べています。
- ここには関数名を書きます。`calculate_expression()`のように丸括弧を付けません。

丸括弧を付けない理由は、「今すぐ実行する」のではなく、「必要なときに実行できる道具として渡す」ためです。

### `max_iters=4`

```python
    max_iters=4,
```

- `max_iters`は、ReActが考えてツールを使う繰り返し回数の上限です。
- 今回は基本的に「ツールを1回使う → 終了」という流れです。
- 上限を4にして、エージェントが必要以上に繰り返さないようにします。

## 動作確認

```python
print("DSPy version:", dspy.__version__)
```

### 解説

- `print(...)`は画面へ表示する関数です。
- `dspy.__version__`には、インストールされているDSPyのバージョンが入っています。
- 実習中にエラーが出た場合、バージョンを記録しておくとメンターへ相談しやすくなります。

---

# 4. Step 2：本当にツールを使ったか確認する

最終回答だけを見ても、ツールを使ったかどうかは分かりません。

DSPyのReActは、実行中の行動を`trajectory`という履歴に保存します。そこから、実際に選んだツール名を取り出します。

## コピーするコード

```python
def used_tools(pred):
    """ReActが使った外部ツール名を取り出す。"""
    trajectory = getattr(pred, "trajectory", {})
    tools = []

    if not isinstance(trajectory, dict):
        return tools

    for key, value in trajectory.items():
        if key.startswith("tool_name_"):
            tool_name = str(value)

            if tool_name not in ["finish", "submit"]:
                tools.append(tool_name)

    return tools
```

## この関数の入力と出力

入力：

```text
ReActが返した実行結果 pred
```

出力例：

```python
["calculate_expression"]
```

ツールを使わなかった場合：

```python
[]
```

## 1行ずつ解説

### 関数を作る

```python
def used_tools(pred):
```

- `def`は関数を作る書き方です。
- 関数名は`used_tools`です。
- 丸括弧の中の`pred`は、この関数が受け取る値の名前です。
- `pred`は`prediction`の略で、エージェントが返した予測結果を表します。

### 関数の説明

```python
    """ReActが使った外部ツール名を取り出す。"""
```

これは関数の説明です。Pythonはこれをdocstringとして保存します。

### `trajectory`を取り出す

```python
    trajectory = getattr(pred, "trajectory", {})
```

`getattr`は、オブジェクトが持つ値を安全に取り出す関数です。

この行は、次の意味です。

```text
predの中にtrajectoryがあれば取り出す
なければ空の辞書 {} を使う
```

3つの引数は次の意味です。

```python
getattr(
    pred,          # 調べる対象
    "trajectory", # 取り出したい名前
    {},            # 見つからない場合の代わりの値
)
```

`{}`は空の辞書です。

### 空のリストを作る

```python
    tools = []
```

見つけたツール名を入れるため、最初に空のリストを作っています。

処理が進むと、例えば次のようになります。

```python
tools = ["calculate_expression"]
```

### `trajectory`が辞書か確認する

```python
    if not isinstance(trajectory, dict):
        return tools
```

- `isinstance(trajectory, dict)`は、「trajectoryは辞書か」を確認します。
- `not`は結果を反対にします。
- つまり、「trajectoryが辞書ではなければ」という条件です。
- その場合は処理を続けず、空の`tools`を返します。

これはエラーでプログラム全体が止まりにくくするための安全策です。

### 履歴を1項目ずつ見る

```python
    for key, value in trajectory.items():
```

`trajectory`は、例えば次のような辞書です。

```python
{
    "thought_0": "計算ツールを使う",
    "tool_name_0": "calculate_expression",
    "tool_args_0": {"expression": "128 * 1.08"},
    "observation_0": {"result": 138.24},
    "tool_name_1": "finish",
}
```

`.items()`を使うと、辞書の名前と値を1組ずつ取り出せます。

```text
key   = "tool_name_0"
value = "calculate_expression"
```

のように、順番に処理します。

### ツール名の行だけ選ぶ

```python
        if key.startswith("tool_name_"):
```

`.startswith(...)`は、「文字列が指定した文字から始まるか」を確認します。

例：

```python
"tool_name_0".startswith("tool_name_")  # True
"thought_0".startswith("tool_name_")    # False
```

そのため、履歴の中から`tool_name_0`や`tool_name_1`だけを選べます。

### ツール名を文字列にする

```python
            tool_name = str(value)
```

`str(...)`は、値を文字列へ変換します。

例えば、値が`calculate_expression`を表していれば、次のような文字列として扱います。

```python
"calculate_expression"
```

### `finish`を除外する

```python
            if tool_name not in ["finish", "submit"]:
```

- `in`は、リストの中に値があるかを確認します。
- `not in`は、リストの中に値がないかを確認します。

ReActは処理を終えるとき、`finish`という内部の終了用ツールを選びます。これは計算・統計・単位変換の外部ツールではありません。

そのため、`finish`は使用ツールの集計から除きます。環境差への安全策として`submit`も除いています。

### リストへ追加する

```python
                tools.append(tool_name)
```

`.append(...)`は、リストの最後へ値を追加します。

処理前：

```python
[]
```

処理後：

```python
["calculate_expression"]
```

### 結果を返す

```python
    return tools
```

見つけた外部ツール名のリストを、関数の呼び出し元へ返します。

---

# 5. Step 3：1問だけ実行して履歴を見る

## コピーするコード

```python
def show_one(program, question):
    pred = program(question=question)

    print("質問:", question)
    print("回答:", pred.answer)
    print("使用ツール:", used_tools(pred))
    print("\n行動履歴:")

    trajectory = getattr(pred, "trajectory", {})

    for key, value in trajectory.items():
        print(f"{key}: {value}")

    return pred
```

## 1行ずつ解説

### 関数の入力

```python
def show_one(program, question):
```

この関数は2つの値を受け取ります。

| 名前 | 入れるもの |
|---|---|
| `program` | `agent`または後で作る`optimized_agent` |
| `question` | 質問文 |

`program`という一般的な名前にしているため、同じ関数で最適化前と最適化後の両方を確認できます。

### エージェントへ質問する

```python
    pred = program(question=question)
```

例えば`program`に`agent`、`question`に質問文が入っている場合、実際には次と同じ意味です。

```python
pred = agent(
    question="128 * 1.08 を計算してください"
)
```

返ってきた結果を`pred`へ保存します。

### 結果を表示する

```python
    print("質問:", question)
    print("回答:", pred.answer)
    print("使用ツール:", used_tools(pred))
```

- `pred.answer`で最終回答を取り出します。
- `used_tools(pred)`で、先ほど作った関数を呼び、使用ツールを取り出します。

### `\n`の意味

```python
    print("\n行動履歴:")
```

`\n`は改行を表す特別な文字です。表示を見やすくするため、行動履歴の前に空行を入れています。

### 履歴を表示する

```python
    trajectory = getattr(pred, "trajectory", {})

    for key, value in trajectory.items():
        print(f"{key}: {value}")
```

`for`で履歴を1項目ずつ取り出して表示します。

`f"{key}: {value}"`はf文字列です。`{}`の中へ変数の値を埋め込みます。

例：

```text
key   = tool_name_0
value = calculate_expression
```

なら、表示は次になります。

```text
tool_name_0: calculate_expression
```

### 結果を返す

```python
    return pred
```

表示するだけでなく、後で詳しく調べられるように`pred`自体も返しています。

## 実行するコード

```python
result = show_one(
    agent,
    "128 * 1.08 を計算してください",
)
```

期待する表示例：

```text
質問: 128 * 1.08 を計算してください
回答: 138.24
使用ツール: ['calculate_expression']
```

履歴に次があれば、計算ツールを呼んでいます。

```text
tool_name_0: calculate_expression
tool_args_0: ...
observation_0: ...
tool_name_1: finish
```

最初から`finish`だけの場合、LLMがツールを使わずに回答した可能性があります。

次の3問も確認してください。

```python
show_one(agent, "128 * 1.08 を計算してください")
show_one(agent, "1, 2, 3, 4, 5 の平均値を求めてください")
show_one(agent, "10 km は何 m ですか")
```

---

# 6. Step 4：練習問題と試験問題を作る

DSPyで使う1問分のデータを`dspy.Example`で表します。

今回の1問には、次の3つを保存します。

```text
question      質問文
answer        正解の数値
expected_tool 正解のツール名
```

## 問題を作る関数

### コピーするコード

```python
def make_example(question, answer, expected_tool):
    return dspy.Example(
        question=question,
        answer=answer,
        expected_tool=expected_tool,
    ).with_inputs("question")
```

## 1行ずつ解説

### 関数を作る

```python
def make_example(question, answer, expected_tool):
```

3つの情報を受け取り、DSPy用の1問を作る関数です。

### `dspy.Example`を作る

```python
    return dspy.Example(
        question=question,
        answer=answer,
        expected_tool=expected_tool,
    ).with_inputs("question")
```

例えば次の呼び出しを考えます。

```python
make_example(
    "25 * 16 を計算してください",
    400.0,
    "calculate_expression",
)
```

作られるデータは、イメージとして次のようなものです。

```text
question      = "25 * 16 を計算してください"
answer        = 400.0
expected_tool = "calculate_expression"
```

### `.with_inputs("question")`

```python
.with_inputs("question")
```

これはDSPyへ、次のことを伝えます。

```text
AIエージェントへ見せる入力はquestionだけ
answerとexpected_toolは採点用に残す
```

つまり、正解の`answer`や`expected_tool`を先にAIへ教えているわけではありません。

必要なら次で確認できます。

```python
sample = make_example(
    "25 * 16 を計算してください",
    400.0,
    "calculate_expression",
)

print("入力:", sample.inputs())
print("正解ラベル:", sample.labels())
```

---

## 練習問題 `trainset`

`trainset`は、最適化に使う練習問題です。

### コピーするコード

```python
trainset = [
    # 計算
    make_example("25 * 16 を計算してください", 400.0, "calculate_expression"),
    make_example("100 / 4 を計算してください", 25.0, "calculate_expression"),
    make_example("2 ** 10 を計算してください", 1024.0, "calculate_expression"),

    # 統計
    make_example("1, 2, 3, 4, 5 の平均値を求めてください", 3.0, "analyze_numbers"),
    make_example("2, 4, 6, 8, 10 の中央値を求めてください", 6.0, "analyze_numbers"),
    make_example("1, 1, 2, 2, 100 の平均値を求めてください", 21.2, "analyze_numbers"),

    # 単位変換
    make_example("1 km は何 m ですか", 1000.0, "convert_units"),
    make_example("5000 m は何 km ですか", 5.0, "convert_units"),
    make_example("2 kg は何 g ですか", 2000.0, "convert_units"),
]
```

## 読み方

```python
trainset = [
```

- `trainset`という変数を作ります。
- `[]`の中に9問を並べています。

各行は次の順番です。

```python
make_example(
    質問文,
    正解の数値,
    正解のツール名,
)
```

例：

```python
make_example(
    "1 km は何 m ですか",
    1000.0,
    "convert_units",
)
```

- 質問：`1 km は何 m ですか`
- 正解：`1000.0`
- 期待ツール：`convert_units`

`# 計算`などの行はコメントであり、Pythonの動作には影響しません。人間が読みやすくするための見出しです。

---

## 試験問題 `testset`

`testset`は、最適化には使わず、最後の比較に使う初見問題です。

### コピーするコード

```python
testset = [
    # 計算
    make_example("128 * 1.08 を計算してください", 138.24, "calculate_expression"),
    make_example("144 / 12 を計算してください", 12.0, "calculate_expression"),
    make_example("(18 + 7) * 4 を計算してください", 100.0, "calculate_expression"),

    # 統計
    make_example("3, 7, 14 の平均値を求めてください", 8.0, "analyze_numbers"),
    make_example("1, 5, 9, 15 の中央値を求めてください", 7.0, "analyze_numbers"),
    make_example("2, 4, 6, 8 の平均値を求めてください", 5.0, "analyze_numbers"),

    # 単位変換
    make_example("3.5 km は何 m ですか", 3500.0, "convert_units"),
    make_example("7500 g は何 kg ですか", 7.5, "convert_units"),
    make_example("0.75 kg は何 g ですか", 750.0, "convert_units"),
]
```

## なぜ練習問題と試験問題を分けるのか

同じ問題で最適化と評価をすると、「その問題にだけ合わせた」のか、「新しい問題でも改善した」のか分かりません。

そのため、次のように分けます。

```text
trainset：お手本を集めるために使う

testset ：最適化前後の最終比較に使う
```

9問は本格的な性能評価には少ないですが、短期間の実習で一連の流れを示すPoCとしては使えます。発表では「小規模な検証である」と明記します。

---

# 7. Step 5：回答文から数値を取り出す

ここが初心者には最も難しく見えやすい部分です。

LLMは同じ答えでも、次のように返す可能性があります。

```text
138.24
答えは138.24です
138.24 m
```

採点するときは、どれも数値`138.24`として扱いたいので、回答文の中から数字を探します。

## コピーするコード

```python
import re


def to_number(value):
    """回答文の最後にある数値をfloatへ変換する。"""
    text = str(value).replace(",", "")

    numbers = re.findall(
        r"[-+]?\d+(?:\.\d+)?",
        text,
    )

    if len(numbers) == 0:
        return None

    last_number = numbers[-1]
    return float(last_number)
```

---

## まず、この関数全体がしていること

例として次を渡します。

```python
to_number("答えは1,000 mです")
```

処理の流れは次の通りです。

```text
1. どんな値でも文字列にする
2. カンマを削除する
3. 文字列の中から数値らしい部分を全部探す
4. 最後に見つかった数値を選ぶ
5. floatという数値型に変換する
```

結果：

```python
1000.0
```

---

## 1行ずつ解説

### `import re`

```python
import re
```

- `import`は、Pythonに用意されている追加機能を読み込む書き方です。
- `re`は、文字列から特定のパターンを探すための標準機能です。
- `re`は`regular expression`、日本語では「正規表現」の略です。

今回の目的は、回答文の中から数字を探すことです。

### 関数の定義

```python
def to_number(value):
```

- 関数名は`to_number`です。
- 「受け取った値を数値へ変える」という意味で付けています。
- `value`には、`pred.answer`などが入ります。

### docstring

```python
    """回答文の最後にある数値をfloatへ変換する。"""
```

この関数が何をするか、人間向けに説明しています。

### 文字列へ変換し、カンマを削除する

```python
    text = str(value).replace(",", "")
```

この1行は、2段階の処理をしています。

#### 1. `str(value)`

`str(...)`は、受け取った値を文字列へ変換します。

```python
str(138.24)
```

結果：

```text
"138.24"
```

もともと文字列でも、そのまま文字列として扱えます。

#### 2. `.replace(",", "")`

`.replace(置き換える前, 置き換えた後)`は、文字を置き換えます。

```python
"1,000".replace(",", "")
```

結果：

```text
"1000"
```

2つ目の引数が空文字`""`なので、カンマを削除する意味になります。

そのため、この行全体は次の意味です。

```text
valueを文字列にして、含まれるカンマを全部消し、textへ保存する
```

### 数字を探す

```python
    numbers = re.findall(
        r"[-+]?\d+(?:\.\d+)?",
        text,
    )
```

`re.findall(パターン, 文字列)`は、指定したパターンに一致する部分をすべて探し、リストで返します。

例：

```python
re.findall(
    r"[-+]?\d+(?:\.\d+)?",
    "答えは138.24です",
)
```

結果：

```python
["138.24"]
```

例：

```python
re.findall(
    r"[-+]?\d+(?:\.\d+)?",
    "3.5 km は 3500 m です",
)
```

結果：

```python
["3.5", "3500"]
```

### 正規表現を分解する

使っているパターンは次です。

```python
r"[-+]?\d+(?:\.\d+)?"
```

初めて見たときに読めなくて普通です。暗記する必要はありません。次の3部分に分けて考えます。

```text
[-+]?   \d+   (?:\.\d+)?
```

#### `r"..."`

最初の`r`はraw文字列を表します。

正規表現では`\d`のようなバックスラッシュをよく使うため、Pythonに「この中のバックスラッシュをなるべくそのまま扱ってください」と伝えます。

#### `[-+]?`

| 記号 | 意味 |
|---|---|
| `[...]` | この中のどれか1文字 |
| `-+` | マイナス記号またはプラス記号 |
| `?` | 直前のものが0回または1回 |

つまり、符号があってもなくてもよいという意味です。

一致する例：

```text
-12
+5
8
```

#### `\d+`

| 記号 | 意味 |
|---|---|
| `\d` | 0から9の数字1文字 |
| `+` | 直前のものが1回以上続く |

つまり、整数部分を表します。

一致する例：

```text
1
25
138
1000
```

#### `(?:\.\d+)?`

これは小数部分です。

中身を分けると次の通りです。

| 部分 | 意味 |
|---|---|
| `(?: ... )` | 複数の記号を1つのまとまりにする |
| `\.` | 小数点そのもの |
| `\d+` | 小数点の後ろに数字が1文字以上 |
| 最後の`?` | この小数部分全体はなくてもよい |

したがって、整数と小数の両方に対応します。

```text
25
25.0
138.24
-3.5
```

`(?: ... )`の`?:`は、`findall`の結果に小数部分だけが返るのを避けるために使っています。今は「まとまりを作るための記号」と理解すれば十分です。

### 数字がなかった場合

```python
    if len(numbers) == 0:
        return None
```

- `len(numbers)`は、リストに何個の要素があるかを返します。
- `== 0`は、「0個か」を確認します。
- 数字が1つも見つからなければ、`None`を返します。

例：

```python
to_number("計算できませんでした")
```

結果：

```python
None
```

`None`は「数値を取り出せなかった」という合図として使います。

### 最後の数値を選ぶ

```python
    last_number = numbers[-1]
```

`numbers`はリストです。

```python
numbers = ["3.5", "3500"]
```

Pythonでは、`[-1]`でリストの最後の要素を取り出せます。

```python
numbers[-1]
```

結果：

```text
"3500"
```

LLMが「3.5 km は 3500 m です」と答えた場合、最後の数値が最終回答であることが多いため、この方法を使います。

### `float`へ変換する

```python
    return float(last_number)
```

`float(...)`は、文字列を小数として扱える数値型へ変換します。

```python
float("138.24")
```

結果：

```python
138.24
```

`return`で、その数値を呼び出し元へ返します。

---

## 動作確認

```python
print(to_number("138.24"))
print(to_number("答えは138.24です"))
print(to_number("1,000 m"))
print(to_number("3.5 km は 3500 m です"))
print(to_number("数値がありません"))
```

期待する結果：

```text
138.24
138.24
1000.0
3500.0
None
```

## この方法の限界

この関数は「最後にある数値が答え」と仮定しています。

例えば次の回答では、最後の`3.5`を選んでしまいます。

```text
答えは3500 mです。入力は3.5 kmでした。
```

そのため、`ToolQA`で「最終的な答えの数値だけを返す」と指示しています。今回の小規模な実験では、この簡単な方法で進めます。

---

# 8. Step 6：回答が正しいか採点する

## コピーするコード

```python
def answer_is_correct(example, pred):
    expected = float(example.answer)
    actual = to_number(pred.answer)

    if actual is None:
        return False

    difference = abs(expected - actual)
    return difference < 0.000001
```

## この関数の役割

```text
正解データ example.answer
        と
AIの回答 pred.answer
```

を比べて、正しければ`True`、間違っていれば`False`を返します。

## 1行ずつ解説

### 関数の入力

```python
def answer_is_correct(example, pred):
```

- `example`には、問題と正解が入っています。
- `pred`には、AIエージェントの実行結果が入っています。

### 正解を数値へ変える

```python
    expected = float(example.answer)
```

`example.answer`は正解データです。それを`float`へ変換し、`expected`へ入れます。

`expected`は「期待する値」という意味です。

### AIの回答から数値を取り出す

```python
    actual = to_number(pred.answer)
```

先ほど作った`to_number`関数を使います。

`actual`は「実際に得られた値」という意味です。

### 数字がなければ不正解

```python
    if actual is None:
        return False
```

AIの回答から数値を見つけられなかった場合は、不正解として`False`を返します。

`is None`は、「値がNoneであるか」を確認する一般的な書き方です。

### 差を計算する

```python
    difference = abs(expected - actual)
```

- `expected - actual`で正解と回答の差を出します。
- `abs(...)`は絶対値を返します。

例：

```text
expected = 138.24
actual   = 138.2400001
```

差はとても小さいため、実質的には同じ答えと扱いたい場合があります。

### 誤差が小さいか確認する

```python
    return difference < 0.000001
```

差が`0.000001`未満なら`True`、それ以上なら`False`を返します。

コンピューターでは小数を完全に同じ形で表せない場合があるため、`expected == actual`という完全一致ではなく、小さな誤差を許しています。

## 確認例

```python
example = make_example(
    "128 * 1.08 を計算してください",
    138.24,
    "calculate_expression",
)

pred = dspy.Prediction(answer="答えは138.24です")

print(answer_is_correct(example, pred))
```

期待する結果：

```text
True
```

---

# 9. Step 7：ツール選択が正しいか採点する

## コピーするコード

```python
def tool_is_correct(example, pred):
    expected = [str(example.expected_tool)]
    actual = used_tools(pred)

    return actual == expected
```

## この関数の役割

例として、計算問題の正解ツールが次だとします。

```text
calculate_expression
```

実際の使用ツールが次なら成功です。

```python
["calculate_expression"]
```

以下は失敗です。

```python
[]
["analyze_numbers"]
["calculate_expression", "analyze_numbers"]
```

## 1行ずつ解説

### 正解ツールをリストにする

```python
    expected = [str(example.expected_tool)]
```

`example.expected_tool`には、例えば次の文字列が入っています。

```text
calculate_expression
```

`used_tools(pred)`はリストを返すため、比較しやすいように正解側もリストにします。

```python
["calculate_expression"]
```

### 実際のツールを取り出す

```python
    actual = used_tools(pred)
```

先ほど作った`used_tools`関数を使います。

### 完全に同じか比べる

```python
    return actual == expected
```

`==`は左右が同じかを比較します。

今回のルールでは、正しいツールをちょうど1回使った場合だけ`True`です。

---

# 10. Step 8：回答とツールをまとめて採点するmetric

## コピーするコード

```python
def metric(example, pred, trace=None):
    answer_ok = answer_is_correct(example, pred)
    tool_ok = tool_is_correct(example, pred)

    return answer_ok and tool_ok
```

## 1行ずつ解説

### 関数の形

```python
def metric(example, pred, trace=None):
```

DSPyの最適化では、採点関数へ主に次の値が渡されます。

- `example`：正解付きの問題
- `pred`：エージェントの予測結果
- `trace`：最適化中の詳しい実行情報

今回は自分で`trace`を使いません。しかしDSPyから渡される可能性があるため、`trace=None`を受け取れる形にしています。

`None`は「指定されていない場合の初期値」です。

### 2種類の採点

```python
    answer_ok = answer_is_correct(example, pred)
    tool_ok = tool_is_correct(example, pred)
```

- `answer_ok`には回答判定の`True`または`False`が入ります。
- `tool_ok`にはツール判定の`True`または`False`が入ります。

### `and`

```python
    return answer_ok and tool_ok
```

`and`は、左右が両方とも`True`の場合だけ`True`になります。

| `answer_ok` | `tool_ok` | 結果 |
|---|---|---|
| `True` | `True` | `True` |
| `True` | `False` | `False` |
| `False` | `True` | `False` |
| `False` | `False` | `False` |

つまり、次の両方が正しい実行だけを成功とします。

```text
回答が正しい
AND
ツール選択が正しい
```

---

# 11. Step 9：最適化前の性能を評価する

次の関数は、データセットの問題を1問ずつ実行し、結果を表示・集計します。

長く見えますが、していることは次の4段階です。

```text
1. 1問実行する
2. 回答とツールを採点する
3. OKの個数を数える
4. 最後に割合を表示する
```

## コピーするコード

```python
def evaluate_program(program, dataset, title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    answer_count = 0
    tool_count = 0
    both_count = 0

    for index, example in enumerate(dataset, start=1):
        pred = program(question=example.question)

        answer_ok = answer_is_correct(example, pred)
        tool_ok = tool_is_correct(example, pred)
        both_ok = answer_ok and tool_ok

        answer_count += int(answer_ok)
        tool_count += int(tool_ok)
        both_count += int(both_ok)

        print(f"\n[{index}] {example.question}")
        print("  正解:", example.answer)
        print("  回答:", pred.answer)
        print("  期待ツール:", example.expected_tool)
        print("  使用ツール:", used_tools(pred))
        print("  回答判定:", "OK" if answer_ok else "NG")
        print("  ツール判定:", "OK" if tool_ok else "NG")
        print("  総合判定:", "OK" if both_ok else "NG")

    total = len(dataset)
    answer_accuracy = 100 * answer_count / total
    tool_accuracy = 100 * tool_count / total
    both_accuracy = 100 * both_count / total

    print("\n--- 集計 ---")
    print(f"回答正解率: {answer_count}/{total} = {answer_accuracy:.1f}%")
    print(f"ツール選択正解率: {tool_count}/{total} = {tool_accuracy:.1f}%")
    print(f"両方の成功率: {both_count}/{total} = {both_accuracy:.1f}%")

    return {
        "answer_accuracy": answer_accuracy,
        "tool_accuracy": tool_accuracy,
        "both_accuracy": both_accuracy,
    }
```

## まとまりごとの解説

### 関数の入力

```python
def evaluate_program(program, dataset, title):
```

| 引数 | 入れるもの |
|---|---|
| `program` | `agent`または`optimized_agent` |
| `dataset` | 今回は`testset` |
| `title` | 表示用の文字列。例：`"最適化前"` |

### 見出しを表示する

```python
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
```

`"=" * 60`は、`=`を60回繰り返した文字列です。

結果：

```text
============================================================
最適化前
============================================================
```

処理結果を見やすくするためだけのコードです。

### 正解数を0から始める

```python
    answer_count = 0
    tool_count = 0
    both_count = 0
```

- `answer_count`：回答が正しかった問題数
- `tool_count`：ツール選択が正しかった問題数
- `both_count`：両方が正しかった問題数

まだ1問も実行していないので、すべて0から始めます。

### 問題を1問ずつ取り出す

```python
    for index, example in enumerate(dataset, start=1):
```

`enumerate`を使うと、問題番号と問題データを一緒に取り出せます。

```text
1回目：index = 1、example = 1問目
2回目：index = 2、example = 2問目
```

`start=1`により、番号を0ではなく1から始めます。

### エージェントを実行する

```python
        pred = program(question=example.question)
```

`example.question`から質問文を取り出し、`program`へ渡します。

`program`が`agent`なら最適化前、`optimized_agent`なら最適化後を実行できます。

### 3種類の判定

```python
        answer_ok = answer_is_correct(example, pred)
        tool_ok = tool_is_correct(example, pred)
        both_ok = answer_ok and tool_ok
```

- 回答だけの判定
- ツールだけの判定
- 両方を満たすかの判定

を別々に作ります。

### 正解数を増やす

```python
        answer_count += int(answer_ok)
        tool_count += int(tool_ok)
        both_count += int(both_ok)
```

`+=`は、「今の値に右側を足して保存する」という意味です。

```python
answer_count += 1
```

は、次と同じ意味です。

```python
answer_count = answer_count + 1
```

`int(True)`は`1`、`int(False)`は`0`になります。

そのため、判定が`True`のときだけ正解数が1増えます。

### 1問ごとの結果を表示する

```python
        print(f"\n[{index}] {example.question}")
        print("  正解:", example.answer)
        print("  回答:", pred.answer)
        print("  期待ツール:", example.expected_tool)
        print("  使用ツール:", used_tools(pred))
```

発表で使える失敗例を見つけるため、各問題の内容を表示しています。

### `"OK" if ... else "NG"`

```python
        print("  回答判定:", "OK" if answer_ok else "NG")
```

これは短い条件分岐です。

```text
answer_okがTrue なら "OK"
answer_okがFalseなら "NG"
```

を表示します。

### 問題数と正解率を計算する

```python
    total = len(dataset)
    answer_accuracy = 100 * answer_count / total
    tool_accuracy = 100 * tool_count / total
    both_accuracy = 100 * both_count / total
```

`len(dataset)`で全問題数を求めます。

例えば9問中6問が正しければ、次の計算です。

```text
100 × 6 ÷ 9 = 66.7%
```

### `.1f`の意味

```python
print(f"{answer_accuracy:.1f}%")
```

`.1f`は、小数点以下1桁で表示する指定です。

```text
66.666666... → 66.7
```

### 辞書を返す

```python
    return {
        "answer_accuracy": answer_accuracy,
        "tool_accuracy": tool_accuracy,
        "both_accuracy": both_accuracy,
    }
```

3つの正解率を辞書にまとめて返します。

例：

```python
{
    "answer_accuracy": 100.0,
    "tool_accuracy": 66.7,
    "both_accuracy": 66.7,
}
```

後で最適化前後を比較するときに使います。

---

## 最適化前を実行する

```python
baseline_result = evaluate_program(
    program=agent,
    dataset=testset,
    title="最適化前",
)
```

### 解説

- `baseline`は、改善前の基準となる結果という意味です。
- `program=agent`なので、最適化前のエージェントを使います。
- `dataset=testset`なので、試験問題9問で評価します。
- 戻り値の正解率を`baseline_result`へ保存します。

表示された3つの正解率と、特徴的な失敗例を1問か2問メモしてください。

---

# 12. Step 10：BootstrapFewShotで最適化する

## コピーするコード

```python
optimizer = dspy.BootstrapFewShot(
    metric=metric,
    max_bootstrapped_demos=4,
    max_labeled_demos=0,
    max_rounds=2,
)

optimized_agent = optimizer.compile(
    agent,
    trainset=trainset,
)
```

## 最初に全体像

おおまかな流れは次の通りです。

```text
trainsetをagentで実行する
  ↓
metricで成功・失敗を判定する
  ↓
回答とツール選択の両方が正しかった履歴を集める
  ↓
成功履歴をお手本として持つoptimized_agentを作る
```

GPT-4o自体の重みを再学習するわけではありません。成功した実行例を、プロンプト内のお手本として利用するイメージです。

## 1行ずつ解説

### Optimizerを作る

```python
optimizer = dspy.BootstrapFewShot(
```

- `BootstrapFewShot`は、成功した実行例を集めてお手本として使うDSPyのOptimizerです。
- ここでは最適化の設定を`optimizer`へ保存します。
- この時点では、まだ最適化は実行されていません。

### 採点方法を渡す

```python
    metric=metric,
```

先ほど作った`metric`関数を渡します。

つまり、次を両方満たした実行だけを成功と判断します。

```text
回答が正しい
ツール選択が正しい
```

### 成功履歴の最大数

```python
    max_bootstrapped_demos=4,
```

metricに合格した実行履歴を、最大4個までお手本として利用します。

`demo`は`demonstration`の略で、「お手本となる例」という意味です。

### 生の正解例を何個入れるか

```python
    max_labeled_demos=0,
```

今回は、質問と正解だけを並べた生の例ではなく、ツール選択を含む成功履歴を中心に見たいので0にします。

### 各問題を何回まで試すか

```python
    max_rounds=2,
```

1つの練習問題について、成功履歴を得るため最大2回試します。

回数を増やすと成功例を得やすくなる一方、LLM呼び出し回数と実行時間が増えます。短い実習なので、まず2回にします。

### `.compile()`で最適化を実行する

```python
optimized_agent = optimizer.compile(
    agent,
    trainset=trainset,
)
```

- `compile`を呼んだ時点で、実際の最適化処理が始まります。
- `agent`は最適化したい元のプログラムです。
- `trainset=trainset`は、練習問題として先ほどのtrainsetを渡すという意味です。
- 結果としてできた新しいエージェントを`optimized_agent`へ保存します。

元の`agent`と、新しい`optimized_agent`は分けて残るため、前後比較ができます。

---

# 13. Step 11：最適化後を評価する

## コピーするコード

```python
optimized_result = evaluate_program(
    program=optimized_agent,
    dataset=testset,
    title="最適化後",
)
```

最適化前と同じ`testset`を使うことが重要です。違う問題を使うと、公平に比較できません。

---

# 14. Step 12：最適化前後を並べて表示する

## コピーするコード

```python
def show_comparison(before, after):
    print("\n=== 最適化前後の比較 ===")

    print(
        "回答正解率:",
        f"{before['answer_accuracy']:.1f}%",
        "->",
        f"{after['answer_accuracy']:.1f}%",
    )

    print(
        "ツール選択正解率:",
        f"{before['tool_accuracy']:.1f}%",
        "->",
        f"{after['tool_accuracy']:.1f}%",
    )

    print(
        "両方の成功率:",
        f"{before['both_accuracy']:.1f}%",
        "->",
        f"{after['both_accuracy']:.1f}%",
    )


show_comparison(
    baseline_result,
    optimized_result,
)
```

## 解説

### 関数の入力

```python
def show_comparison(before, after):
```

- `before`には最適化前の辞書を渡します。
- `after`には最適化後の辞書を渡します。

### 辞書から値を取り出す

```python
before["answer_accuracy"]
```

辞書は、名前を指定して値を取り出せます。

```python
before = {
    "answer_accuracy": 100.0,
    "tool_accuracy": 66.7,
    "both_accuracy": 66.7,
}
```

なら、

```python
before["tool_accuracy"]
```

の結果は`66.7`です。

### 最後の呼び出し

```python
show_comparison(
    baseline_result,
    optimized_result,
)
```

保存しておいた最適化前後の結果を関数へ渡します。

表示例：

```text
=== 最適化前後の比較 ===
回答正解率: 100.0% -> 100.0%
ツール選択正解率: 55.6% -> 88.9%
両方の成功率: 55.6% -> 88.9%
```

数値は実際の実行結果に置き換わります。

---

# 15. 結果をどう読むか

## パターンA：回答は正しいが、ツール選択が低い

例：

```text
回答正解率         100.0%
ツール選択正解率    55.6%
両方の成功率        55.6%
```

考察例：

> 簡単な問題ではLLMがツールを使わず、自力で正解する例が見られた。回答正解率だけでは、期待したツール利用ができたか判断できないことが分かった。

## パターンB：最適化後に改善した

考察例：

> 回答とツール選択の両方を確認するmetricを用いて最適化した結果、未知の試験問題でも正しいツールを選ぶ割合が改善した。

## パターンC：最適化前から高かった

考察例：

> 今回の問題は単純で、最適化前から高い性能だったため、大きな差は確認できなかった。今後は、複数のツールで迷いやすい問題や、より複雑な表現を含む問題で検証する必要がある。

## パターンD：最適化後も改善しなかった

考察例：

> DSPyによる評価と最適化の流れは実装できたが、小規模なtrainsetでは十分な改善を確認できなかった。成功した実行履歴の数や問題の多様性を増やす必要がある。

改善しなかった場合も、実験として失敗ではありません。実際の結果と原因候補を正直に説明すれば、発表として成立します。

---

# 16. 発表用に記録するもの

## 結果表

| 評価項目 | 最適化前 | 最適化後 |
|---|---:|---:|
| 回答正解率 | `__ %` | `__ %` |
| ツール選択正解率 | `__ %` | `__ %` |
| 両方の成功率 | `__ %` | `__ %` |

## 代表例

| 質問 | 期待ツール | 最適化前 | 最適化後 |
|---|---|---|---|
| 例：128 × 1.08 | `calculate_expression` | ツールなし | 正しいツールを使用 |
| 例：3.5 km → m | `convert_units` | 正しいツール | 正しいツール |

必ず実際の実行結果に合わせて書いてください。

## 実行環境

次も記録しておくと、再現性の説明がしやすくなります。

```text
利用モデル：gpt-4oの社内Azureデプロイ
DSPyバージョン：実行時の表示値
trainset：9問
testset：9問
最適化手法：BootstrapFewShot
```

APIキーや社内URLは記録・公開しません。

---

# 17. 実習中の進め方

## Day 2：エージェントを動かす

1. メンターのコードを実行する
2. `ToolQA`と`agent`を作る
3. `used_tools`を作る
4. `show_one`で3種類の質問を試す
5. `trajectory`の表示を確認する

この日のゴール：

```text
質問ごとに、実際に使ったツール名を表示できる
```

## Day 3：採点できるようにする

1. `make_example`を作る
2. `trainset`と`testset`を作る
3. `to_number`を作る
4. `answer_is_correct`を作る
5. `tool_is_correct`を作る
6. `metric`を作る
7. 最適化前を評価する

この日のゴール：

```text
回答正解率
ツール選択正解率
両方の成功率
```

を表示できる。

## Day 4：最適化する

1. `BootstrapFewShot`を作る
2. `.compile()`を実行する
3. `optimized_agent`を作る
4. 最適化後を評価する
5. 前後を比較する

## Day 5：発表を作る

1. 結果表を埋める
2. 代表的な成功例・失敗例を選ぶ
3. 考察を書く
4. 6枚程度のスライドへまとめる
5. 10分以内で練習する

---

# 18. 10分発表の構成

スライドは6枚程度で十分です。

## 1枚目：背景と目的（1分30秒）

- AIエージェントでは、いつ・どのツールを使うかが重要
- 簡単な問題では、LLMがツールを使わずに回答することがある
- DSPyでツール選択を評価・最適化する

目的の説明例：

> DSPyを用いて、AIエージェントのツール選択を評価・最適化し、将来のAIエージェント開発・運用への適用可能性を確認した。

## 2枚目：実装した仕組み（1分30秒）

```text
質問
  ↓
DSPy ReAct
  ↓
計算 / 統計 / 単位変換
  ↓
回答
```

説明すること：

- ReActが質問に合うツールを選ぶ
- ツールはMCP経由で実行される
- `trajectory`から、実際に利用したツールを確認した

コード全体をスライドへ載せる必要はありません。

## 3枚目：評価方法（2分）

問題は3種類です。

```text
計算
統計
単位変換
```

評価指標は3つです。

```text
回答正解率
ツール選択正解率
両方の成功率
```

最も大切な説明：

> 回答が正しくても、期待したツールを使っていなければ、ツール選択としては不正解とした。

## 4枚目：最適化方法（1分30秒）

```text
trainset
  ↓
ReActで実行
  ↓
metricで採点
  ↓
成功した実行履歴をお手本にする
  ↓
optimized_agent
```

説明すること：

- GPT-4o自体を再学習したわけではない
- metricに合格した実行履歴をお手本として追加した
- testsetは最適化に使っていない

## 5枚目：結果（2分30秒）

| 評価項目 | 最適化前 | 最適化後 |
|---|---:|---:|
| 回答正解率 | `__ %` | `__ %` |
| ツール選択正解率 | `__ %` | `__ %` |
| 両方の成功率 | `__ %` | `__ %` |

特徴的な1問も示します。

```text
質問：128 * 1.08 を計算してください

最適化前：回答は正しいがツール未使用
最適化後：calculate_expressionを使用して正解
```

実際の結果に合わせて変更してください。

## 6枚目：結論と今後の課題（1分）

結論例：

> DSPyを用いて、最終回答だけでなくツール選択を含めた評価と最適化を実装できた。また、回答正解率だけでは、期待したツール利用ができたか判断できないことを確認した。

今後の課題：

- 問題数を増やす
- ツールに渡した引数も評価する
- 複数ツールを順番に使う問題を試す
- コストや応答時間を測る
- モデル更新後にも性能を維持できるか確認する

---

# 19. 質疑応答の例

## Q. 簡単な計算なら、ツールを使わなくてもよいのでは？

> 実運用ではその設計も考えられます。今回はツール選択性能を切り分けて評価するため、計算問題では計算ツールを使うという実験ルールを設定しました。

## Q. DSPyは何を最適化したのですか？

> GPT-4o自体を再学習したのではなく、metricに合格したツール利用を含む実行履歴を、お手本として追加しました。

## Q. なぜ回答正解率だけでは不十分ですか？

> ツールを使わずに正解した場合と、正しいツールを使って正解した場合を区別できないためです。

## Q. なぜ問題数が少ないのですか？

> 今回は短期間のPoCとして、実装から評価までの一連の流れを確認することを優先しました。本格的な評価には、より多様なデータが必要です。

## Q. ツール名だけを評価すれば十分ですか？

> 今回は初期検証としてツール名と最終回答を評価しました。実運用では、引数、実行エラー、コスト、応答時間も評価する必要があります。

## Q. `to_number`の正規表現は完全ですか？

> 完全ではありません。今回は、整数と一般的な小数を含む簡単な回答に限定した評価用の処理です。複雑な表記を扱う場合は、出力形式をJSONなどに固定する方法も考えられます。

---

# 20. 困ったとき

## `NameError: calculate_expression is not defined`

確認すること：

1. メンターのコードを先に実行したか
2. 関数定義の後で`agent`を作ったか
3. `class ToolQA`が別の関数の中に入っていないか
4. ノートブックなら、上のセルから順番に実行したか

## `IndentationError`

行の先頭の空白が揃っていないエラーです。

特に次を確認します。

```python
def sample():
    print("関数の中")
```

関数や`if`、`for`の中は、同じ幅で右へ下げます。タブと空白を混ぜない方が安全です。

## `used_tools()`が空になる

まず次を表示します。

```python
print(result)
print(getattr(result, "trajectory", None))
print(dspy.__version__)
```

`trajectory`に最初から`finish`だけが入っている場合、LLMがツールを使わずに回答した可能性があります。

これはプログラムの故障とは限りません。今回の評価では、回答が正しくてもツール判定はNGにします。

## 回答は正しいのに総合判定がNG

例：

```text
回答: 138.24
使用ツール: []
```

今回のルールでは次の判定です。

```text
回答判定   OK
ツール判定 NG
総合判定   NG
```

意図した通りの判定です。

## `.compile()`後も結果が変わらない

まず、trainsetの代表問題を確認します。

```python
show_one(agent, trainset[0].question)
show_one(agent, trainset[3].question)
show_one(agent, trainset[6].question)
```

すべて失敗している場合、BootstrapFewShotが成功履歴を集められない可能性があります。

確認すること：

- Signatureに「必ず1回使う」と書いてあるか
- ツール名のスペルが正しいか
- MCPツール自体が正常に動くか
- `max_rounds=2`になっているか

それでも変わらない場合は、その結果を記録します。

考察例：

> 最適化に利用できる成功履歴が十分に得られなかった可能性がある。

## `to_number`が`None`を返す

次を確認します。

```python
print(pred.answer)
```

回答に数値が含まれていなければ`None`になります。

また、今回の正規表現は一般的な整数と小数を対象としており、分数や特殊な指数表記などには対応していません。

## MCPやHTTPのエラー

例：

```text
ConnectError
Timeout
401
403
500
```

接続先、認証、社内ネットワーク、MCPサーバーなどの問題である可能性があります。接続情報をGitHubへ載せず、エラーメッセージだけをメンターへ見せて相談してください。

## 評価の途中でエラーが出て止まる

まず、どの質問で止まったか確認します。問題を1問だけ`show_one`で実行すると原因を見つけやすくなります。

```python
show_one(
    agent,
    "エラーになった質問文",
)
```

---

# 21. 今回やらなくてよいこと

10分発表に向けて、次は無理に追加しなくて構いません。

- MIPROv2
- GEPA
- LLM-as-a-Judge
- ツール引数の完全一致評価
- 複数ツールの連鎖
- 詳細な料金計算
- 大規模データセット
- Webアプリ化

今回の中心は次の一文です。

> **回答の正しさとツール利用の正しさを分けて評価し、DSPyによる最適化前後を比較した。**

ここを自分の言葉で説明できれば、発表として成立します。

---

# 22. 実行順チェックリスト

- [ ] メンターから渡されたコードを実行した
- [ ] `ToolQA`と`agent`を作った
- [ ] DSPyのバージョンを記録した
- [ ] `used_tools()`を作った
- [ ] `show_one()`を作った
- [ ] 3種類の質問で`trajectory`を確認した
- [ ] `make_example()`を作った
- [ ] `trainset`と`testset`を作った
- [ ] `to_number()`を作り、動作確認した
- [ ] `answer_is_correct()`を作った
- [ ] `tool_is_correct()`を作った
- [ ] `metric()`を作った
- [ ] `evaluate_program()`を作った
- [ ] 最適化前の結果を記録した
- [ ] `BootstrapFewShot`を実行した
- [ ] 最適化後の結果を記録した
- [ ] 最適化前後の表を作った
- [ ] 代表的な成功例・失敗例を選んだ
- [ ] 実験の限界を書いた
- [ ] GitHubに秘密情報が入っていないことを確認した

---

# 23. 付録：`class ToolQA`以降の全コード

説明を読み終えた後、全体を確認するためのコードです。メンターから渡されたコードを先に実行してから使用してください。

```python
class ToolQA(dspy.Signature):
    """
    質問に答えてください。

    1つの数式を計算する質問では、
    calculate_expressionを必ず1回使ってください。

    平均値や中央値を求める質問では、
    analyze_numbersを必ず1回使ってください。

    単位変換の質問では、
    convert_unitsを必ず1回使ってください。

    関係のないツールは使わないでください。
    """
    question: str = dspy.InputField()
    answer: str = dspy.OutputField(
        desc="最終的な答えの数値だけを返す"
    )


agent = dspy.ReAct(
    ToolQA,
    tools=[
        calculate_expression,
        analyze_numbers,
        convert_units,
    ],
    max_iters=4,
)


print("DSPy version:", dspy.__version__)


def used_tools(pred):
    """ReActが使った外部ツール名を取り出す。"""
    trajectory = getattr(pred, "trajectory", {})
    tools = []

    if not isinstance(trajectory, dict):
        return tools

    for key, value in trajectory.items():
        if key.startswith("tool_name_"):
            tool_name = str(value)

            if tool_name not in ["finish", "submit"]:
                tools.append(tool_name)

    return tools


def show_one(program, question):
    pred = program(question=question)

    print("質問:", question)
    print("回答:", pred.answer)
    print("使用ツール:", used_tools(pred))
    print("\n行動履歴:")

    trajectory = getattr(pred, "trajectory", {})

    for key, value in trajectory.items():
        print(f"{key}: {value}")

    return pred


def make_example(question, answer, expected_tool):
    return dspy.Example(
        question=question,
        answer=answer,
        expected_tool=expected_tool,
    ).with_inputs("question")


trainset = [
    make_example("25 * 16 を計算してください", 400.0, "calculate_expression"),
    make_example("100 / 4 を計算してください", 25.0, "calculate_expression"),
    make_example("2 ** 10 を計算してください", 1024.0, "calculate_expression"),
    make_example("1, 2, 3, 4, 5 の平均値を求めてください", 3.0, "analyze_numbers"),
    make_example("2, 4, 6, 8, 10 の中央値を求めてください", 6.0, "analyze_numbers"),
    make_example("1, 1, 2, 2, 100 の平均値を求めてください", 21.2, "analyze_numbers"),
    make_example("1 km は何 m ですか", 1000.0, "convert_units"),
    make_example("5000 m は何 km ですか", 5.0, "convert_units"),
    make_example("2 kg は何 g ですか", 2000.0, "convert_units"),
]


testset = [
    make_example("128 * 1.08 を計算してください", 138.24, "calculate_expression"),
    make_example("144 / 12 を計算してください", 12.0, "calculate_expression"),
    make_example("(18 + 7) * 4 を計算してください", 100.0, "calculate_expression"),
    make_example("3, 7, 14 の平均値を求めてください", 8.0, "analyze_numbers"),
    make_example("1, 5, 9, 15 の中央値を求めてください", 7.0, "analyze_numbers"),
    make_example("2, 4, 6, 8 の平均値を求めてください", 5.0, "analyze_numbers"),
    make_example("3.5 km は何 m ですか", 3500.0, "convert_units"),
    make_example("7500 g は何 kg ですか", 7.5, "convert_units"),
    make_example("0.75 kg は何 g ですか", 750.0, "convert_units"),
]


import re


def to_number(value):
    """回答文の最後にある数値をfloatへ変換する。"""
    text = str(value).replace(",", "")

    numbers = re.findall(
        r"[-+]?\d+(?:\.\d+)?",
        text,
    )

    if len(numbers) == 0:
        return None

    last_number = numbers[-1]
    return float(last_number)


def answer_is_correct(example, pred):
    expected = float(example.answer)
    actual = to_number(pred.answer)

    if actual is None:
        return False

    difference = abs(expected - actual)
    return difference < 0.000001


def tool_is_correct(example, pred):
    expected = [str(example.expected_tool)]
    actual = used_tools(pred)

    return actual == expected


def metric(example, pred, trace=None):
    answer_ok = answer_is_correct(example, pred)
    tool_ok = tool_is_correct(example, pred)

    return answer_ok and tool_ok


def evaluate_program(program, dataset, title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    answer_count = 0
    tool_count = 0
    both_count = 0

    for index, example in enumerate(dataset, start=1):
        pred = program(question=example.question)

        answer_ok = answer_is_correct(example, pred)
        tool_ok = tool_is_correct(example, pred)
        both_ok = answer_ok and tool_ok

        answer_count += int(answer_ok)
        tool_count += int(tool_ok)
        both_count += int(both_ok)

        print(f"\n[{index}] {example.question}")
        print("  正解:", example.answer)
        print("  回答:", pred.answer)
        print("  期待ツール:", example.expected_tool)
        print("  使用ツール:", used_tools(pred))
        print("  回答判定:", "OK" if answer_ok else "NG")
        print("  ツール判定:", "OK" if tool_ok else "NG")
        print("  総合判定:", "OK" if both_ok else "NG")

    total = len(dataset)
    answer_accuracy = 100 * answer_count / total
    tool_accuracy = 100 * tool_count / total
    both_accuracy = 100 * both_count / total

    print("\n--- 集計 ---")
    print(f"回答正解率: {answer_count}/{total} = {answer_accuracy:.1f}%")
    print(f"ツール選択正解率: {tool_count}/{total} = {tool_accuracy:.1f}%")
    print(f"両方の成功率: {both_count}/{total} = {both_accuracy:.1f}%")

    return {
        "answer_accuracy": answer_accuracy,
        "tool_accuracy": tool_accuracy,
        "both_accuracy": both_accuracy,
    }


baseline_result = evaluate_program(
    program=agent,
    dataset=testset,
    title="最適化前",
)


optimizer = dspy.BootstrapFewShot(
    metric=metric,
    max_bootstrapped_demos=4,
    max_labeled_demos=0,
    max_rounds=2,
)


optimized_agent = optimizer.compile(
    agent,
    trainset=trainset,
)


optimized_result = evaluate_program(
    program=optimized_agent,
    dataset=testset,
    title="最適化後",
)


def show_comparison(before, after):
    print("\n=== 最適化前後の比較 ===")

    print(
        "回答正解率:",
        f"{before['answer_accuracy']:.1f}%",
        "->",
        f"{after['answer_accuracy']:.1f}%",
    )

    print(
        "ツール選択正解率:",
        f"{before['tool_accuracy']:.1f}%",
        "->",
        f"{after['tool_accuracy']:.1f}%",
    )

    print(
        "両方の成功率:",
        f"{before['both_accuracy']:.1f}%",
        "->",
        f"{after['both_accuracy']:.1f}%",
    )


show_comparison(
    baseline_result,
    optimized_result,
)
```

---

# 24. 参考資料

- DSPy公式：ReActとツール  
  <https://dspy.ai/diving-deeper/tools-react-and-mcp/>
- DSPy公式：Example  
  <https://dspy.ai/api/primitives/Example/>
- DSPy公式：Metrics and evaluation  
  <https://dspy.ai/diving-deeper/metrics-and-evaluation/>
- DSPy公式：BootstrapFewShot  
  <https://dspy.ai/api/optimizers/BootstrapFewShot/>

DSPyは更新される可能性があります。実習環境で挙動が異なる場合は、`dspy.__version__`とエラーメッセージをメンターへ見せて確認してください。
