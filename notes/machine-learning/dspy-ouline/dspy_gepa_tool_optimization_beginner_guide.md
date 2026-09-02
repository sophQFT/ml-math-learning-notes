# DSPy + GEPAでツール利用を評価・最適化する実習ガイド

この資料は、**PythonとDSPyの初心者が、Jupyter Notebookへ上から順にコピーしながら進めるための手順書**です。

今回の方針は、メンターから示された次の構成に合わせています。

```text
AIエージェントが質問に答える
        ↓
実際に使ったツール・引数・結果を記録する
        ↓
別のLLMがツール利用を採点する
        ↓
採点結果と文章フィードバックをGEPAへ渡す
        ↓
GEPAがエージェントの指示文を改善する
        ↓
最適化前と最適化後を比較する
```

この資料で扱うのは、メンターから渡された次の3つの関数を定義した**後の部分**です。

```python
calculate_expression
analyze_numbers
convert_units
```

Azure OpenAIやMCPへ接続する部分は、実習では「3つの道具を使えるようにする準備」と考えて構いません。

> **この資料のコードはPythonの構文チェックを行っています。** ただし、社内のAzure OpenAI、MCPサーバー、DSPyのインストール状況までは外部から確認できないため、実際の接続確認は実習環境で行ってください。

---

## 最重要：個人GitHubへ載せない情報

このMarkdown自体には、APIキーや社内URLを記載していません。

個人GitHubへ上げるときは、次を絶対に含めないでください。

- Azure OpenAIのAPIキー
- MCPのAPIキー
- 社内IPアドレス
- 社内のURLやエンドポイント
- 実際の秘密情報が入った`.ipynb`
- 社外秘の質問データや実行結果
- 社内画面のURLやIPアドレスが写ったスクリーンショット

公開用には、このMarkdownだけを上げるのが安全です。Notebookも上げる場合は、キーやURLを削除したコピーを作ってください。

---

# 0. 実習中の最短手順

時間がないときは、次の順番だけ守ってください。

1. メンターから渡されたコードを実行し、3つのツールを使える状態にする
2. この資料の「セル1」から順番にコピーする
3. まず1問だけ動作確認する
4. LLMによる採点を1問だけ試す
5. 最適化前のテスト結果を保存する
6. GEPAを実行する
7. 最適化後のテスト結果を保存する
8. 最適化前後の表と、代表的な1問を発表に載せる

最初からGEPAを実行しないでください。GEPAはエージェントと採点LLMを何度も呼ぶため、前段のコードに誤りがあると時間を無駄にします。

---

# 1. Jupyter Notebookでの注意

## コードを実行するセルは「Code」にする

写真では、上部のセル種別が`Markdown`になっている場合があります。

Pythonコードを実行するときは、Jupyter上部のプルダウンを次のように変更します。

```text
Markdown → Code
```

- 説明文を書くセル：`Markdown`
- Pythonを実行するセル：`Code`

コードセルは、通常`Shift + Enter`で実行できます。

## カーネルを再起動したら上から実行し直す

Jupyterは、前のセルで作った変数や関数をメモリに保持しています。

カーネルを再起動すると、その記憶が消えます。その場合は、メンターのコードを含めて上から順番に再実行してください。

---

# 2. 今回の登場人物

今回の構成には、役割の違う3つのLLM利用があります。

| 名前 | 役割 |
|---|---|
| エージェント用LM | 質問を読み、ツールを選び、最終回答を作る |
| 採点用LM `judge_lm` | エージェントのツール利用を採点する |
| 振り返り用LM `reflection_lm` | 採点結果を読み、改善された指示文を考える |

同じAzure OpenAIのデプロイメントを使うこともできますが、役割は別です。

今回の実験を一言で表すと、次のようになります。

> **エージェントを別のLLMで採点し、その採点と助言を使ってGEPAが指示文を改善する。**

---

# 3. 最低限覚えるDSPy用語

| 用語 | 今回の意味 |
|---|---|
| `Signature` | LLMへ渡す入力と、LLMから得る出力を決める設計図 |
| `ReAct` | 考える・ツールを使う・結果を見る、を繰り返すエージェント |
| `trajectory` | ReActがどのツールを、どの引数で使ったかを含む行動履歴 |
| `Module` | 複数の処理をまとめた、最適化可能なDSPyプログラム |
| `Prediction` | DSPyプログラムが返す結果の入れ物 |
| `Example` | 練習問題やテスト問題を1件分入れる入れ物 |
| `metric` | 出力を何点とするか決める採点関数 |
| `GEPA` | 点数と文章フィードバックを使い、指示文を改善するOptimizer |
| `compile` | Optimizerを実行し、改善済みプログラムを作る処理 |

---

# 4. 全体の構成

この資料で作るプログラムは、次の構成です。

```text
質問
  ↓
MCPToolAgent
  ↓
dspy.ReAct
  ├─ calculate_expression
  ├─ analyze_numbers
  └─ convert_units
  ↓
最終回答 + ツール呼び出し + ツール結果
  ↓
ToolUseJudge
  ↓
4種類の点数 + 改善のための文章フィードバック
  ↓
GEPA
  ↓
改善されたMCPToolAgent
```

`MCPToolAgent`を作る理由は、ReActの最終回答だけでなく、**ツールの呼び出しと実行結果も採点LLMへ渡すため**です。

---

# 5. 事前確認

メンターから渡されたコードを実行した後、次のセルを実行してください。

## セル0：必要なものが存在するか確認

```python
print("calculate_expression:", calculate_expression)
print("analyze_numbers:", analyze_numbers)
print("convert_units:", convert_units)
print("DSPy version:", dspy.__version__)
print("GEPA is available:", hasattr(dspy, "GEPA"))
```

## 何を確認しているか

### `print(...)`

画面に値を表示します。

### `dspy.__version__`

現在インストールされているDSPyのバージョンを表示します。発表資料にも記録しておくと、実験条件を説明できます。

### `hasattr(dspy, "GEPA")`

`dspy`の中に`GEPA`という機能があるかを確認します。

- `True`：GEPAを使える可能性が高い
- `False`：インストールされているDSPyが古い可能性がある

`False`の場合は、勝手にライブラリを更新せず、メンターへ確認してください。

---

# 6. セル1：importと共通設定

メンターの3ツールを定義した後に、次をコピーしてください。

```python
import json
import dspy

MAX_ITERS = 5

AVAILABLE_TOOLS_TEXT = """
1. calculate_expression
   - 1つの数式を計算するツール
   - 主な引数: expression, precision
   - 例: 128 * 1.08、sqrt(144)

2. analyze_numbers
   - 数値の配列を統計処理するツール
   - 主な引数: values, operations, second_values, outlier_method
   - 例: 平均値、中央値、標準偏差、分散

3. convert_units
   - 単位を変換するツール
   - 主な引数: value, from_unit, to_unit, category, precision
   - 例: kmからm、kgからg
""".strip()
```

## 1行ずつ解説

### `import json`

```python
import json
```

Pythonに標準で入っている`json`という機能を使えるようにします。

今回は、次のようなPythonのデータを、採点LLMが読みやすい文字列へ変換するために使います。

```python
[
    {
        "tool_name": "calculate_expression",
        "arguments": {"expression": "25 * 16"},
    }
]
```

この変換に使う関数が、後で出てくる`json.dumps(...)`です。

### `import dspy`

```python
import dspy
```

DSPyを使えるようにします。すでに前のセルで実行していても、もう一度書いて問題ありません。

### `MAX_ITERS = 5`

```python
MAX_ITERS = 5
```

ReActが、最大何回まで「次に何をするか」を判断できるかを決めます。

`MAX_ITERS`は自分で付けた変数名です。大文字にしているのは、「途中で頻繁に変更しない共通設定」という意味を表す慣習です。

今回の質問は基本的に、次の流れです。

```text
ツールを1回使う
  ↓
finishで終了する
```

そのため、上限5回で十分です。

### `AVAILABLE_TOOLS_TEXT`

```python
AVAILABLE_TOOLS_TEXT = """
...
""".strip()
```

採点LLMへ、「エージェントはどのツールを利用できたのか」を伝える説明文です。

三重引用符`"""`を使うと、複数行の文字列を書けます。

最後の`.strip()`は、文字列の先頭と末尾にある余分な改行や空白を取り除きます。

---

# 7. セル2：エージェントの入出力を定義する

```python
class ToolQA(dspy.Signature):
    """
    ユーザーの質問に答えてください。

    必要な場合だけ、利用可能なツールを使ってください。
    1つの数式の計算には calculate_expression、
    数値配列の統計処理には analyze_numbers、
    単位変換には convert_units を使ってください。

    ツールを使う場合は、質問に合うツールと引数を選び、
    実行結果を最終回答へ正確に反映してください。
    不要なツールや重複したツール呼び出しは避けてください。
    """

    question: str = dspy.InputField(
        desc="ユーザーからの質問"
    )

    answer: str = dspy.OutputField(
        desc="ユーザーの質問に対する最終回答"
    )
```

## 何をしているコードか

このクラスは、エージェントの仕事を次のように定義しています。

```text
入力：question
出力：answer
```

## 1行ずつ解説

### `class ToolQA(dspy.Signature):`

```python
class ToolQA(dspy.Signature):
```

- `class`：設計図を作るためのPythonの書き方
- `ToolQA`：この設計図に付けた名前
- `dspy.Signature`：DSPyでLLMの入出力を定義するための土台
- 最後の`:`：この次からクラスの中身が始まる

### 三重引用符の指示文

```python
    """
    ユーザーの質問に答えてください。
    ...
    """
```

この文章は、単なる人間向けメモではありません。DSPyがLLMへ渡す**初期の指示文**として使います。

GEPAは、評価結果を見ながら、このような指示文を改善します。

つまり今回の「最適化」は、主に次を自動改善する処理です。

```text
どのようなときに、どのツールを使うかという指示
```

GPTなどのモデル本体の重みを学習し直すわけではありません。

### `question: str`

```python
    question: str = dspy.InputField(
        desc="ユーザーからの質問"
    )
```

- `question`：入力欄の名前
- `: str`：文字列を入れる予定という型のヒント
- `dspy.InputField(...)`：これは入力欄であるとDSPyへ伝える
- `desc=...`：入力欄の説明

### `answer: str`

```python
    answer: str = dspy.OutputField(
        desc="ユーザーの質問に対する最終回答"
    )
```

- `answer`：出力欄の名前
- `dspy.OutputField(...)`：これはLLMに作ってほしい出力であると伝える

`answer`を`float`ではなく`str`にしている理由は、数値1つだけでなく、平均値と中央値の両方や、簡単な説明文も扱えるようにするためです。

---

# 8. セル3：ReActの履歴からツール情報を取り出す

ReActを実行すると、`trajectory`という行動履歴が返ります。

概念的には、次のような辞書です。

```python
{
    "thought_0": "計算ツールを使う",
    "tool_name_0": "calculate_expression",
    "tool_args_0": {"expression": "25 * 16"},
    "observation_0": {"result": 400},
    "tool_name_1": "finish",
}
```

採点LLMには、思考文そのものではなく、次を渡します。

- 実際に呼んだツール名
- ツールへ渡した引数
- ツールの実行結果
- 実行順序

## コピーするコード

```python
def extract_tool_information(trajectory):
    """ReActの履歴から、外部ツールの呼び出しと結果を取り出す。"""

    tool_calls = []
    tool_results = []

    if not isinstance(trajectory, dict):
        return tool_calls, tool_results

    for index in range(MAX_ITERS):
        tool_name = trajectory.get(f"tool_name_{index}")

        if tool_name is None:
            continue

        if tool_name == "finish":
            continue

        tool_args = trajectory.get(
            f"tool_args_{index}",
            {},
        )

        observation = trajectory.get(
            f"observation_{index}",
            "",
        )

        tool_calls.append(
            {
                "order": index + 1,
                "tool_name": tool_name,
                "arguments": tool_args,
            }
        )

        tool_results.append(
            {
                "order": index + 1,
                "tool_name": tool_name,
                "result": observation,
            }
        )

    return tool_calls, tool_results
```

## 1行ずつ解説

### 関数を作る

```python
def extract_tool_information(trajectory):
```

- `def`：関数を作る
- `extract_tool_information`：関数名
- `trajectory`：関数が受け取る値

この関数は、ReActの履歴を受け取り、整理されたツール情報を返します。

### 空のリストを2つ作る

```python
    tool_calls = []
    tool_results = []
```

`[]`はリストです。複数の値を順番に保存できます。

- `tool_calls`：ツール名と引数を入れる
- `tool_results`：ツール名と実行結果を入れる

最初は何も入っていないため、空のリストを作ります。

### `trajectory`が辞書か確認する

```python
    if not isinstance(trajectory, dict):
        return tool_calls, tool_results
```

- `isinstance(trajectory, dict)`：`trajectory`が辞書か確認する
- `not`：結果を反対にする
- `if`：条件が成立したときだけ中を実行する

つまり、次の意味です。

```text
trajectoryが辞書ではない場合は、
空のtool_callsとtool_resultsを返して終了する
```

`return`は、関数の結果を呼び出し元へ返し、その関数を終了します。

### 最大5回分を確認する

```python
    for index in range(MAX_ITERS):
```

`for`は繰り返しです。

`MAX_ITERS`が5なら、`index`には順番に次が入ります。

```text
0, 1, 2, 3, 4
```

Pythonでは、番号は0から始まることが多いです。

### ツール名を取り出す

```python
        tool_name = trajectory.get(f"tool_name_{index}")
```

例えば`index`が0なら、f文字列は次の文字列になります。

```python
"tool_name_0"
```

`trajectory.get(...)`は、辞書からその名前の値を取り出します。

普通の`trajectory["tool_name_0"]`と違い、該当するキーがない場合でもエラーにならず、`None`を返します。

### 値がない回を飛ばす

```python
        if tool_name is None:
            continue
```

`None`は「値がない」という意味です。

`continue`は、現在の繰り返しの残りを飛ばして、次の回へ進みます。

### `finish`を除外する

```python
        if tool_name == "finish":
            continue
```

`finish`は、ReActが処理を終了するために使う内部的なツールです。

MCPの計算・統計・単位変換ツールではないため、採点用のツール呼び出しから除外します。

### 引数を取り出す

```python
        tool_args = trajectory.get(
            f"tool_args_{index}",
            {},
        )
```

`get`の2つ目の値`{}`は、該当する引数がなかった場合の初期値です。

`{}`は空の辞書を表します。

### 実行結果を取り出す

```python
        observation = trajectory.get(
            f"observation_{index}",
            "",
        )
```

ReActでは、ツールから返った結果を`observation`として記録します。

該当する結果がない場合は、空文字列`""`を使います。

### ツール呼び出しをリストへ追加する

```python
        tool_calls.append(
            {
                "order": index + 1,
                "tool_name": tool_name,
                "arguments": tool_args,
            }
        )
```

`append(...)`は、リストの末尾に値を1つ追加します。

ここでは辞書を追加しています。

```python
{
    "order": 1,
    "tool_name": "calculate_expression",
    "arguments": {"expression": "25 * 16"},
}
```

`index + 1`としているのは、人間が読みやすいように実行順序を1から始めるためです。

### 結果もリストへ追加する

```python
        tool_results.append(
            {
                "order": index + 1,
                "tool_name": tool_name,
                "result": observation,
            }
        )
```

同じ順序とツール名に、実行結果を対応させます。

### 2つの結果を返す

```python
    return tool_calls, tool_results
```

この関数は2つの値をまとめて返します。

後で次のように受け取ります。

```python
tool_calls, tool_results = extract_tool_information(trajectory)
```

---

# 9. セル4：評価可能なエージェントを作る

メンターの写真にあった`dspy.Module`を使い、ReActを包みます。

## コピーするコード

```python
class MCPToolAgent(dspy.Module):
    """ReActの回答とツール利用履歴をまとめて返すDSPyプログラム。"""

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

        tool_calls, tool_results = extract_tool_information(
            trajectory
        )

        return dspy.Prediction(
            answer=str(
                getattr(result, "answer", "")
            ),
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
            trajectory=trajectory,
        )


program = MCPToolAgent()
```

## なぜ`dspy.Module`が必要なのか

もともとのReActは、主に次を返します。

```text
answer
trajectory
```

しかし採点LLMは、次の名前で情報を受け取る設計です。

```text
tool_calls
tool_results
answer
```

そこで`MCPToolAgent`の中で、ReActの`trajectory`を整理し、採点しやすい形へ変換します。

また、`dspy.Module`の中にReActを置くことで、GEPAが内部のDSPy予測器を見つけて指示文を最適化できます。

## 1行ずつ解説

### `class MCPToolAgent(dspy.Module):`

```python
class MCPToolAgent(dspy.Module):
```

`dspy.Module`を土台にして、自分用のDSPyプログラムを作ります。

### `__init__`

```python
    def __init__(self):
```

`__init__`は、`MCPToolAgent()`を作ったときに最初に1回実行される特別な関数です。

### `self`

`self`は、「今作っているMCPToolAgent自身」を表します。

例えば、次の行はMCPToolAgent自身の中に`agent`を保存しています。

```python
self.agent = ...
```

### `super().__init__()`

```python
        super().__init__()
```

土台である`dspy.Module`側の初期設定を実行します。

初心者の段階では、`dspy.Module`を継承するときに必要な定型文と考えて構いません。

### ReActを作る

```python
        self.agent = dspy.ReAct(
            ToolQA,
            tools=[
                calculate_expression,
                analyze_numbers,
                convert_units,
            ],
            max_iters=MAX_ITERS,
        )
```

- `ToolQA`：エージェントの入出力と初期指示
- `tools=[...]`：利用可能な3つの道具
- `max_iters=MAX_ITERS`：最大反復回数

`self.agent`へ保存するため、後の`forward`から使えます。

### `forward`

```python
    def forward(self, question: str):
```

`dspy.Module`では、プログラムを呼び出したときに行う処理を`forward`へ書きます。

後で次のように書くと、内部では`forward`が実行されます。

```python
program(question="質問")
```

### ReActを実行する

```python
        result = self.agent(
            question=question
        )
```

`forward`が受け取った質問を、内部のReActへ渡します。

結果を`result`という変数に保存します。

### `getattr`

```python
        trajectory = getattr(
            result,
            "trajectory",
            {},
        )
```

`getattr(対象, 名前, 初期値)`は、対象が持つ値を安全に取得します。

この場合は次の意味です。

```text
resultにtrajectoryがあれば取得する
なければ空の辞書 {} を使う
```

通常は`result.trajectory`でも取得できます。ただし、異常終了などで存在しない場合に備えて`getattr`を使っています。

### 2つの値を受け取る

```python
        tool_calls, tool_results = extract_tool_information(
            trajectory
        )
```

前のセルで作った関数へ`trajectory`を渡し、次の2つを受け取ります。

- `tool_calls`
- `tool_results`

### `dspy.Prediction`

```python
        return dspy.Prediction(
            ...
        )
```

DSPyプログラムの結果として、複数の項目をまとめて返します。

今回返すのは次の4項目です。

| 項目 | 内容 |
|---|---|
| `answer` | 最終回答 |
| `tool_calls` | 実際のツール名、引数、順序 |
| `tool_results` | ツールの実行結果 |
| `trajectory` | 元のReAct履歴。デバッグ用 |

### `str(...)`

```python
answer=str(getattr(result, "answer", ""))
```

`str(...)`は、値を文字列へ変換します。

採点LLMへ渡しやすいように、回答を文字列へそろえています。

### `json.dumps(...)`

```python
json.dumps(
    tool_calls,
    ensure_ascii=False,
    default=str,
)
```

リストや辞書をJSON形式の文字列へ変換します。

- `ensure_ascii=False`：日本語を`\uXXXX`のような表記にせず、そのまま残す
- `default=str`：JSONへ直接変換できない値があれば、文字列へ変換する

### `program = MCPToolAgent()`

```python
program = MCPToolAgent()
```

設計図から、実際に利用するプログラムを1つ作ります。

以後、最適化前のエージェントは`program`という名前で扱います。

---

# 10. セル5：まず1問だけ動作確認する

GEPAや採点LLMへ進む前に、エージェント単体が動くか確認します。

```python
prediction = program(
    question=(
        "10, 20, 30, 40, 50 の平均値と中央値を"
        "求めてください"
    )
)

print("最終回答:")
print(prediction.answer)

print("\nツール呼び出し:")
print(prediction.tool_calls)

print("\nツール結果:")
print(prediction.tool_results)

print("\n元のtrajectory:")
print(prediction.trajectory)
```

## 期待する内容

この質問では、通常は次のツールを使うのが適切です。

```text
analyze_numbers
```

`prediction.tool_calls`に、例えば次のような情報があれば、ツールを呼んでいます。

```text
"tool_name": "analyze_numbers"
```

`prediction.tool_calls`が次のようになっている場合、外部ツールを使っていません。

```text
[]
```

`[]`は空のリストです。

## この段階で確認すること

- エラーなく回答が返ったか
- `tool_calls`にツール名が入っているか
- `tool_results`にMCPの結果が入っているか
- `finish`だけが外部ツールとして数えられていないか

このセルが動かないうちは、先へ進まない方が安全です。

---

# 11. セル6：練習問題とテスト問題を作る

今回は、正解ラベルを人間が細かく書く代わりに、LLMが次を見て採点します。

- 質問
- 利用可能だったツール
- 実際のツール呼び出し
- ツール結果
- 最終回答

## Exampleを作る関数

```python
def make_example(question):
    return dspy.Example(
        question=question,
        available_tools=AVAILABLE_TOOLS_TEXT,
    ).with_inputs("question")
```

## 1行ずつ解説

### `make_example`

```python
def make_example(question):
```

質問文から、DSPy用の1件分のデータを作る関数です。

### `dspy.Example`

```python
    return dspy.Example(
        question=question,
        available_tools=AVAILABLE_TOOLS_TEXT,
    )
```

1件のデータに、次の2項目を保存します。

- `question`：エージェントへ入力する質問
- `available_tools`：採点LLMへ見せるツール説明

### `.with_inputs("question")`

```python
.with_inputs("question")
```

`question`だけを、エージェントへ渡す入力として指定します。

`available_tools`はエージェントへの追加ヒントではなく、採点時に使う情報として残ります。

---

## 練習問題 `trainset`

```python
trainset = [
    make_example(
        "25 * 16 を計算してください"
    ),
    make_example(
        "sqrt(144) を計算してください"
    ),
    make_example(
        "1, 2, 6 の平均値を求めてください"
    ),
    make_example(
        "2, 4, 8, 10 の中央値を求めてください"
    ),
    make_example(
        "1 km は何 m ですか"
    ),
    make_example(
        "2.5 kg は何 g ですか"
    ),
    make_example(
        "数字の7をそのまま答えてください"
    ),
    make_example(
        "平均値とは何かを簡潔に説明してください"
    ),
]
```

最後の2問は、外部ツールを使う必要がない問題です。

これを入れることで、次も評価できます。

> **ツールを使うべき問題だけでなく、使わなくてよい問題で無駄なツールを呼ばないか。**

---

## 未知問題 `testset`

```python
testset = [
    make_example(
        "128 * 1.08 を計算してください"
    ),
    make_example(
        "(18 + 7) / 5 を計算してください"
    ),
    make_example(
        "3, 7, 14 の平均値を求めてください"
    ),
    make_example(
        "10, 20, 30, 40, 50 の平均値と中央値を求めてください"
    ),
    make_example(
        "3.5 km は何 m ですか"
    ),
    make_example(
        "7500 g は何 kg ですか"
    ),
    make_example(
        "数字の42をそのまま答えてください"
    ),
    make_example(
        "中央値が外れ値の影響を受けにくい理由を簡潔に説明してください"
    ),
]
```

## `trainset`と`testset`の違い

| データ | 用途 |
|---|---|
| `trainset` | GEPAが指示文を改善するときに使う |
| `testset` | 最適化前後を公平に比較するために使う |

`testset`はGEPAの`compile`へ渡しません。

テスト結果を見た後で、その問題に合わせて手動で指示文を直すと、未知問題ではなくなります。実験としては、同じ`testset`を最適化前後でそのまま使ってください。

---

# 12. セル7：ツール利用を採点するLLMの設計図

ここから、メンターの写真にあった`ToolUseJudge`を作ります。

```python
class ToolUseJudge(dspy.Signature):
    """
    AIエージェントによるツール利用を厳格に評価してください。

    以下の観点を総合的に評価します。

    1. 選択したツールがユーザー要求に適しているか
    2. 外部ツールを使用する必要があったか
    3. ツールに渡した引数が適切か
    4. 最終回答がユーザー要求を満たしているか
    5. 不要または重複したツール呼び出しがないか

    代替ツールでも同じ目的を安全かつ正確に達成できる場合は、
    必ずしも減点しないでください。

    一般的な説明や単純な値の復唱など、外部ツールが不要な質問で
    ツールを使わなかった場合は、ツール非利用を適切と評価してください。
    """

    user_query: str = dspy.InputField(
        desc="ユーザーが入力した質問"
    )

    available_tools: str = dspy.InputField(
        desc="エージェントが利用可能だったツールの名前と説明"
    )

    tool_calls: str = dspy.InputField(
        desc="実際に呼び出したツール、引数、実行順序"
    )

    tool_results: str = dspy.InputField(
        desc="各ツールの実行結果"
    )

    final_answer: str = dspy.InputField(
        desc="エージェントが生成した最終回答"
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
        desc="問題点と、エージェントの指示を改善するための具体的な助言"
    )
```

## 採点LLMへ渡す入力

| 入力 | 内容 |
|---|---|
| `user_query` | 元の質問 |
| `available_tools` | 利用可能だった3つのツール |
| `tool_calls` | 実際に選んだツールと引数 |
| `tool_results` | ツールから返った結果 |
| `final_answer` | エージェントの最終回答 |

## 採点LLMから受け取る出力

| 出力 | 意味 |
|---|---|
| `tool_selection_score` | 質問に合ったツールを選んだか |
| `tool_necessity_score` | ツールを使う・使わない判断が適切か |
| `argument_score` | ツールへ渡した値や単位などが適切か |
| `task_success_score` | 最終的に質問へ正しく答えたか |
| `feedback` | 指示文をどう改善すべきかという文章 |

点数はすべて次の範囲です。

```text
0.0 = まったく良くない
1.0 = とても良い
```

## なぜ4つに分けるのか

最終回答だけを見ると、簡単な計算ではLLMがツールを使わず正解する場合があります。

例えば次の状況です。

```text
回答は正しい
しかしcalculate_expressionを使っていない
```

4項目に分けることで、次を別々に見られます。

```text
答えは正しいか
ツール選択は正しいか
そもそもツールは必要だったか
引数は正しかったか
```

---

# 13. セル8：採点用LMを作る

このコードは、メンターから渡されたAzure OpenAI用の変数を再利用します。

```python
judge_lm = dspy.LM(
    f"azure/{AZURE_OPENAI_DEPLOYMENT}",
    api_key=AZURE_OPENAI_API_KEY,
    api_base=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION,
    temperature=0.0,
)

judge = dspy.ChainOfThought(
    ToolUseJudge
)
```

## 注意

メンターから採点用の別デプロイメント名を指定された場合は、1行目のデプロイメントだけをその指示に合わせてください。

個人GitHub用のMarkdownやNotebookへ、実際のキーと社内エンドポイントを直接書かないでください。

## 1行ずつ解説

### `judge_lm`

```python
judge_lm = dspy.LM(...)
```

採点専用に使うLMを作り、`judge_lm`という名前で保存します。

### `f"azure/{AZURE_OPENAI_DEPLOYMENT}"`

`f`が付いた文字列は、変数の値を文字列の中へ埋め込めます。

例えばデプロイメント名が`my-model`なら、概念的には次の文字列になります。

```text
azure/my-model
```

### `temperature=0.0`

採点結果のばらつきを抑える目的で、出力のランダム性を低くします。

完全に同じ結果を保証するものではありませんが、比較実験では同じ設定を使うことが重要です。

### `dspy.ChainOfThought(ToolUseJudge)`

```python
judge = dspy.ChainOfThought(
    ToolUseJudge
)
```

`ToolUseJudge`の入出力に従い、採点結果を作るDSPyモジュールです。

採点LLMに複数の観点を考慮させた上で、4種類の点数とフィードバックを出させます。

---

# 14. セル9：採点LLMを呼び出す関数

```python
def run_judge(example, prediction):
    with dspy.context(lm=judge_lm):
        return judge(
            user_query=example.question,
            available_tools=example.available_tools,
            tool_calls=prediction.tool_calls,
            tool_results=prediction.tool_results,
            final_answer=prediction.answer,
        )
```

## 何をしているか

1件の問題と、エージェントの実行結果を受け取ります。

その2つから採点に必要な情報を集め、`judge_lm`へ渡します。

## 1行ずつ解説

### 引数

```python
def run_judge(example, prediction):
```

- `example`：元の質問と利用可能ツール
- `prediction`：エージェントの回答、ツール呼び出し、ツール結果

### `with dspy.context(lm=judge_lm):`

```python
    with dspy.context(lm=judge_lm):
```

このブロックの中だけ、使用するLMを`judge_lm`へ一時的に切り替えます。

ブロックが終わると、もとのエージェント用LM設定へ戻ります。

つまり、次の使い分けです。

```text
エージェント実行時 → 通常のlm
採点時             → judge_lm
```

### 各入力を渡す

```python
        return judge(
            user_query=example.question,
            available_tools=example.available_tools,
            tool_calls=prediction.tool_calls,
            tool_results=prediction.tool_results,
            final_answer=prediction.answer,
        )
```

`ToolUseJudge`で定義した入力欄へ、対応する値を渡します。

写真では`example.user_query`となっていましたが、この資料ではデータを簡単にするため、元の質問を`example.question`から渡しています。意味は同じです。

---

# 15. セル10：点数を安全に0.0〜1.0へそろえる

採点LLMには0.0から1.0で返すよう指示していますが、念のため値を確認します。

```python
def to_score(value):
    """値を0.0から1.0のfloatへ変換する。"""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0

    return max(
        0.0,
        min(1.0, number),
    )
```

## 1行ずつ解説

### 関数の目的

```python
def to_score(value):
```

`value`を、0.0から1.0の小数へ変換する関数です。

### `try`

```python
    try:
        number = float(value)
```

`float(...)`は、値を小数へ変換します。

例えば次の変換ができます。

```text
"0.8" → 0.8
1       → 1.0
```

`try`は、「まずこの処理を試す」という意味です。

### `except`

```python
    except (TypeError, ValueError):
        return 0.0
```

例えば、採点結果が誤って`"高い"`のような文字列だった場合、`float("高い")`は失敗します。

そのエラーを受け止め、0.0を返します。

- `TypeError`：値の型が不適切
- `ValueError`：値の内容を数値へ変換できない

### `min(1.0, number)`

```python
min(1.0, number)
```

2つのうち小さい方を選びます。

例えば`number`が1.2なら、1.0になります。

### `max(0.0, ...)`

```python
max(0.0, ...)
```

2つのうち大きい方を選びます。

例えば値が-0.2なら、0.0になります。

全体として、次の範囲に収めています。

```text
0.0以下 → 0.0
0.0〜1.0 → そのまま
1.0以上 → 1.0
```

---

# 16. セル11：4種類の点数から総合点を作る

```python
def get_judgment_scores(judgment):
    tool_selection = to_score(
        judgment.tool_selection_score
    )

    tool_necessity = to_score(
        judgment.tool_necessity_score
    )

    argument = to_score(
        judgment.argument_score
    )

    task_success = to_score(
        judgment.task_success_score
    )

    total = (
        0.40 * tool_selection
        + 0.20 * tool_necessity
        + 0.15 * argument
        + 0.25 * task_success
    )

    return {
        "tool_selection": tool_selection,
        "tool_necessity": tool_necessity,
        "argument": argument,
        "task_success": task_success,
        "total": total,
    }
```

## 何をしているか

採点LLMが返した4つの点数を、次の重みで合計します。

| 項目 | 重み |
|---|---:|
| ツール選択 | 0.40 |
| ツール利用の必要性 | 0.20 |
| 引数 | 0.15 |
| タスク達成度 | 0.25 |
| 合計 | 1.00 |

今回のテーマは「ツール利用最適化」なので、ツール選択を最も重くしています。

この重みは自然法則ではなく、**実験者が決めた評価方針**です。発表では次のように説明できます。

> ツール選択の改善を主目的としたため、ツール選択へ40%の重みを置いた。

メンターから別の重みを指定された場合は、その指示を優先してください。

## `judgment.tool_selection_score`

```python
judgment.tool_selection_score
```

`judgment`という採点結果が持っている`tool_selection_score`へアクセスしています。

`.`は、「その対象が持っている値や機能へアクセスする」記号です。

## 重み付き合計

```python
    total = (
        0.40 * tool_selection
        + 0.20 * tool_necessity
        + 0.15 * argument
        + 0.25 * task_success
    )
```

例えば4項目がすべて1.0なら、次の計算になります。

```text
0.40 + 0.20 + 0.15 + 0.25 = 1.00
```

## 辞書で返す

```python
    return {
        "tool_selection": tool_selection,
        ...
    }
```

後で、総合点だけでなく各項目も表示できるように、辞書として返します。

---

# 17. セル12：GEPA用のmetricを作る

```python
def tool_use_metric(
    example,
    prediction,
    trace=None,
    pred_name=None,
    pred_trace=None,
):
    judgment = run_judge(
        example,
        prediction,
    )

    scores = get_judgment_scores(
        judgment
    )

    return dspy.Prediction(
        score=scores["total"],
        feedback=judgment.feedback,
    )
```

## metricとは

metricは、エージェントの実行結果を何点とするか決める採点関数です。

今回のmetricは、単純な正解・不正解ではありません。

```text
採点LLMを呼ぶ
  ↓
4項目を採点する
  ↓
重み付き総合点を作る
  ↓
点数と文章フィードバックをGEPAへ返す
```

## 5つの引数

```python
example,
prediction,
trace=None,
pred_name=None,
pred_trace=None,
```

最初の2つを主に使います。

| 引数 | 意味 |
|---|---|
| `example` | 元の問題 |
| `prediction` | プログラムの出力 |
| `trace` | プログラム全体のDSPy実行記録 |
| `pred_name` | GEPAが現在注目している内部予測器名 |
| `pred_trace` | その内部予測器の実行記録 |

今回の簡単な実装では、後ろ3つを直接使いません。しかしGEPAがmetricを呼べるように、引数として用意します。

`=None`は、「値が渡されなければNoneを使う」という意味です。

## `dspy.Prediction(score=..., feedback=...)`

```python
    return dspy.Prediction(
        score=scores["total"],
        feedback=judgment.feedback,
    )
```

GEPAへ次の2つを返します。

- `score`：0.0から1.0の総合点
- `feedback`：どこを直すべきかという文章

GEPAでは、点数だけでなく文章フィードバックを使える点が重要です。

例えば採点LLMが次のように返したとします。

```text
単位変換なのにcalculate_expressionを選んでいる。
convert_unitsを選び、from_unitとto_unitを明示する指示へ改善すべき。
```

GEPAは、このような助言を参考に新しい指示文候補を作ります。

---

# 18. セル13：1問だけ採点して確認する

GEPAを実行する前に、採点LLMが正常に動くか確認します。

```python
example = testset[0]

prediction = program(
    question=example.question
)

judgment = run_judge(
    example,
    prediction,
)

scores = get_judgment_scores(
    judgment
)

print("質問:", example.question)
print("回答:", prediction.answer)
print("ツール呼び出し:", prediction.tool_calls)
print("ツール結果:", prediction.tool_results)
print("ツール選択:", scores["tool_selection"])
print("ツール必要性:", scores["tool_necessity"])
print("引数:", scores["argument"])
print("タスク達成度:", scores["task_success"])
print("総合点:", scores["total"])
print("フィードバック:", judgment.feedback)
```

## このセルで確認すること

- 4つの点数が表示される
- 総合点が0.0から1.0に収まっている
- フィードバックが質問と実行結果に対応している
- 明らかに不自然な採点になっていない

この段階では、LLMによる採点を人間が1問だけ確認します。

LLMによる採点は絶対的な正解ではありません。簡単な数問について人間の感覚と大きくずれていないことを確認してから、最適化へ進みます。

---

# 19. セル14：複数問題を評価する関数

次の関数は、テスト問題を1問ずつ実行し、採点結果の平均を計算します。

```python
def evaluate_program(
    program_to_evaluate,
    dataset,
    title,
):
    total_tool_selection = 0.0
    total_tool_necessity = 0.0
    total_argument = 0.0
    total_task_success = 0.0
    total_score = 0.0

    print("=" * 70)
    print(title)
    print("=" * 70)

    for number, example in enumerate(
        dataset,
        start=1,
    ):
        prediction = program_to_evaluate(
            question=example.question
        )

        judgment = run_judge(
            example,
            prediction,
        )

        scores = get_judgment_scores(
            judgment
        )

        total_tool_selection += scores[
            "tool_selection"
        ]
        total_tool_necessity += scores[
            "tool_necessity"
        ]
        total_argument += scores[
            "argument"
        ]
        total_task_success += scores[
            "task_success"
        ]
        total_score += scores["total"]

        print(f"\n{number}. 質問: {example.question}")
        print(f"   回答: {prediction.answer}")
        print(f"   ツール呼び出し: {prediction.tool_calls}")
        print(f"   ツール結果: {prediction.tool_results}")
        print(
            "   スコア: "
            f"選択={scores['tool_selection']:.2f}, "
            f"必要性={scores['tool_necessity']:.2f}, "
            f"引数={scores['argument']:.2f}, "
            f"達成度={scores['task_success']:.2f}, "
            f"総合={scores['total']:.2f}"
        )
        print(f"   フィードバック: {judgment.feedback}")

    count = len(dataset)

    result = {
        "tool_selection": total_tool_selection / count,
        "tool_necessity": total_tool_necessity / count,
        "argument": total_argument / count,
        "task_success": total_task_success / count,
        "total": total_score / count,
    }

    print("\n--- 平均 ---")
    print(
        f"ツール選択: {result['tool_selection']:.3f}"
    )
    print(
        f"ツール必要性: {result['tool_necessity']:.3f}"
    )
    print(
        f"引数: {result['argument']:.3f}"
    )
    print(
        f"タスク達成度: {result['task_success']:.3f}"
    )
    print(
        f"総合: {result['total']:.3f}"
    )

    return result
```

## 主な部分の解説

### 合計を入れる変数

```python
    total_tool_selection = 0.0
    ...
```

最初は0点です。問題を1件評価するたびに点数を足していきます。

### `"=" * 70`

```python
print("=" * 70)
```

`=`という文字を70回並べます。出力を見やすくするための区切り線です。

### `enumerate`

```python
    for number, example in enumerate(
        dataset,
        start=1,
    ):
```

`dataset`の問題を1件ずつ取り出しながら、番号も付けます。

- `number`：1、2、3、…
- `example`：現在処理している問題
- `start=1`：番号を1から始める

### `+=`

```python
total_score += scores["total"]
```

次と同じ意味です。

```python
total_score = total_score + scores["total"]
```

現在までの合計へ、新しい点数を加えます。

### f文字列と小数点以下の桁数

```python
f"総合={scores['total']:.2f}"
```

- `f"..."`：変数を文字列へ埋め込む
- `:.2f`：小数点以下2桁で表示する

例えば`0.8765`は、表示上`0.88`になります。

### 平均

```python
"total": total_score / count
```

全問題の合計点を問題数で割り、平均を計算します。

### `return result`

```python
return result
```

平均点を辞書として返します。後で最適化前後の比較に使います。

---

# 20. セル15：最適化前の結果を測る

```python
baseline_result = evaluate_program(
    program,
    testset,
    "最適化前",
)
```

## 何をしているか

- `program`：最適化前のエージェント
- `testset`：GEPAへ見せない未知問題
- `"最適化前"`：出力画面の見出し

結果は`baseline_result`に保存されます。

このセルの出力は、後で比較できるように保存してください。

Jupyterの出力をテキストファイルへコピーするか、社内ルールの範囲でスクリーンショットを撮ります。公開用の資料へ社内URLや秘密情報が写らないよう注意してください。

---

# 21. セル16：GEPAの振り返り用LMを作る

```python
reflection_lm = dspy.LM(
    f"azure/{AZURE_OPENAI_DEPLOYMENT}",
    api_key=AZURE_OPENAI_API_KEY,
    api_base=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION,
    temperature=0.0,
)
```

## `reflection_lm`の役割

採点LLMが出した次の情報を読みます。

```text
総合点
改善のための文章フィードバック
```

そして、エージェントの指示文をどのように変更すればよいか考えます。

採点用LMと同じ設定でも動かせますが、役割を分かりやすくするため別の変数にしています。

メンターから別の振り返りモデルを指定された場合は、その指示を優先してください。

---

# 22. セル17：GEPAで最適化する

最初は並列数を1にして、安全に動作確認します。

```python
optimizer = dspy.GEPA(
    metric=tool_use_metric,
    auto="light",
    reflection_lm=reflection_lm,
    num_threads=1,
)

optimized_agent = optimizer.compile(
    student=program,
    trainset=trainset,
)
```

## 1行ずつ解説

### `dspy.GEPA`

```python
optimizer = dspy.GEPA(
```

GEPAというOptimizerを作ります。

Optimizerは、metricの点数が高くなるようにDSPyプログラムの指示文を改善する仕組みです。

### `metric=tool_use_metric`

```python
metric=tool_use_metric
```

「この採点関数の点数を高くしてください」とGEPAへ伝えます。

### `auto="light"`

```python
auto="light"
```

最適化に使う計算量を軽めにします。

最初の実習では`light`で十分です。より大きな設定へ変更すると実行時間とAPI利用量が増えます。

### `reflection_lm=reflection_lm`

```python
reflection_lm=reflection_lm
```

改善案を考えるLMとして、前のセルで作った`reflection_lm`を指定します。

### `num_threads=1`

```python
num_threads=1
```

同時に何件処理するかを指定します。

写真では`4`でしたが、初心者が最初に動かすときは`1`の方が安全です。

- MCPサーバーへ同時アクセスしすぎるのを避ける
- Azure OpenAIのレート制限を受けにくくする
- エラーが起きた場所を追いやすくする

正常に動き、メンターから許可があれば`4`へ変更できます。

### `.compile(...)`

```python
optimized_agent = optimizer.compile(
    student=program,
    trainset=trainset,
)
```

- `student=program`：改善したい元のプログラム
- `trainset=trainset`：改善に使う練習問題

結果として、新しい`optimized_agent`が作られます。

元の`program`は残るため、最適化前後を比較できます。

## GEPA内部のイメージ

```text
1. trainsetでprogramを実行する
2. tool_use_metricで採点する
3. 採点LLMのfeedbackを見る
4. reflection_lmが新しい指示文を提案する
5. 新しい指示文で再度実行・採点する
6. 点数の高い指示文を残す
```

GEPAでは、エージェントと採点LLMを複数回呼びます。通常の1問実行より時間がかかるのは正常です。

---

# 23. セル18：最適化後の結果を測る

```python
optimized_result = evaluate_program(
    optimized_agent,
    testset,
    "最適化後",
)
```

最適化前と同じ`testset`、同じ採点LLM、同じ重みを使います。

ここを変えると公平に比較できません。

---

# 24. セル19：最適化前後を比較する

```python
def print_comparison(before, after):
    print("=" * 70)
    print("最適化前後の比較")
    print("=" * 70)

    print(
        "ツール選択: "
        f"{before['tool_selection']:.3f}"
        " -> "
        f"{after['tool_selection']:.3f}"
    )

    print(
        "ツール必要性: "
        f"{before['tool_necessity']:.3f}"
        " -> "
        f"{after['tool_necessity']:.3f}"
    )

    print(
        "引数: "
        f"{before['argument']:.3f}"
        " -> "
        f"{after['argument']:.3f}"
    )

    print(
        "タスク達成度: "
        f"{before['task_success']:.3f}"
        " -> "
        f"{after['task_success']:.3f}"
    )

    print(
        "総合: "
        f"{before['total']:.3f}"
        " -> "
        f"{after['total']:.3f}"
        "  変化="
        f"{after['total'] - before['total']:+.3f}"
    )


print_comparison(
    baseline_result,
    optimized_result,
)
```

## `before['total']`

`before`は辞書です。

```python
before["total"]
```

と書くと、最適化前の総合点を取り出せます。

## `after - before`

```python
after['total'] - before['total']
```

最適化による総合点の変化です。

- 正の値：改善した
- 0付近：ほとんど変わらなかった
- 負の値：悪化した

`:+.3f`は、正の値にも`+`を表示し、小数点以下3桁にする指定です。

---

# 25. セル20：GEPAが変えた指示文を見る

最適化によって何が変わったかを見ると、発表に説得力が出ます。

```python
def show_instructions(program_to_show, title):
    print("=" * 70)
    print(title)
    print("=" * 70)

    for name, predictor in program_to_show.named_predictors():
        print(f"\n予測器名: {name}")
        print(predictor.signature.instructions)


show_instructions(
    program,
    "最適化前の指示文",
)

show_instructions(
    optimized_agent,
    "最適化後の指示文",
)
```

## `named_predictors()`

DSPyプログラムの中にある、LLMを呼ぶ予測器を順番に取り出します。

ReActの内部には、ツール選択を行う部分や、最終回答を作る部分があります。

表示名はDSPyのバージョンにより多少異なる可能性がありますが、概ね次のような予測器が見えます。

```text
agent.react
agent.extract
```

## 発表で使う方法

指示文全体をスライドへ貼る必要はありません。

最適化後に追加・強化された内容を、1〜2個だけ要約します。

例：

```text
最適化前：必要な場合にツールを使う
最適化後：数式・統計・単位変換を明確に分類し、不要な重複呼び出しを避ける
```

実際の出力に基づいて記載し、想像で変更内容を作らないでください。

---

# 26. 発表用に記録するもの

最低限、次を保存してください。

| 記録項目 | 内容 |
|---|---|
| DSPyバージョン | `dspy.__version__`の出力 |
| エージェント用モデル | 使用したAzureデプロイメント。キーは記載しない |
| 採点・振り返りモデル | 使用したデプロイメント。キーは記載しない |
| trainset数 | 今回は8件 |
| testset数 | 今回は8件 |
| 最適化設定 | GEPA、`auto="light"`、`num_threads=1` |
| 最適化前の各平均点 | 5項目 |
| 最適化後の各平均点 | 5項目 |
| 代表的な成功・失敗例 | 1〜2問 |
| 指示文の変化 | 実際の出力から要約 |

## 結果表の例

実際の数値で埋めます。

| 指標 | 最適化前 | 最適化後 | 変化 |
|---|---:|---:|---:|
| ツール選択 | 0.xxx | 0.xxx | +0.xxx |
| ツール必要性 | 0.xxx | 0.xxx | +0.xxx |
| 引数 | 0.xxx | 0.xxx | +0.xxx |
| タスク達成度 | 0.xxx | 0.xxx | +0.xxx |
| 総合 | 0.xxx | 0.xxx | +0.xxx |

---

# 27. 10分発表の構成

## 1枚目：背景と課題（1分）

話す内容の例：

> AIエージェントでは、回答を生成するだけでなく、いつ、どのツールを、どの引数で使うかが品質に影響します。簡単な問題ではLLM自身が答えられるため、最終回答だけでは適切なツール利用だったか評価できません。

## 2枚目：目的（1分）

> DSPyを用いてAIエージェントのツール利用を評価・最適化し、AIエージェント開発・運用への適用可能性を確認しました。

## 3枚目：システム構成（1分30秒）

次の図を載せます。

```text
質問
 ↓
DSPy ReAct
 ↓
計算 / 統計 / 単位変換
 ↓
回答とツール履歴
 ↓
LLM Judge
```

説明するポイント：

- 3つのMCPツールを利用
- `trajectory`から実際のツール名・引数・結果を取得
- 別のLLMが採点

## 4枚目：評価と最適化方法（2分）

評価項目：

```text
ツール選択 40%
必要性     20%
引数       15%
達成度     25%
```

最適化方法：

```text
点数 + 文章フィードバック
          ↓
         GEPA
          ↓
指示文を自動改善
```

## 5枚目：結果（2分30秒）

- 最適化前後の表
- 代表的な1問
- 実際に変化した指示内容

代表例の表：

| 質問 | 最適化前 | 最適化後 |
|---|---|---|
| 3.5 kmは何mか | ツールなし、または誤ツール | `convert_units`を適切な引数で利用 |

実際の結果に合わせて書き換えます。

## 6枚目：考察・結論（2分）

改善した場合：

> GEPAにより、回答だけでなくツール選択・引数・不要な呼び出しを含む総合評価を改善できました。文章フィードバックを使って指示文を自動改善できる点は、モデル更新時の再調整やAIエージェント運用に活用できると考えます。

改善しなかった場合：

> 今回は問題数が少なく、最適化前から性能が高かった、または採点のばらつきがあったため、大きな改善は確認できませんでした。一方で、ツール利用履歴を含む評価からGEPAによる最適化まで、一連の仕組みを構築できました。今後はデータ数と難易度を増やす必要があります。

---

# 28. 結果の読み方

## 総合点が上がった

GEPAが作った指示文により、今回のmetricが重視する行動が改善した可能性があります。

ただし、8問だけの小規模実験なので、「どんな質問でも必ず改善する」とは言えません。

## ツール選択は上がったが、タスク達成度が下がった

ツールを使うことへ意識が寄りすぎ、最終回答の作り方が悪化した可能性があります。

このように項目を分けることで、総合点だけでは見えない変化を説明できます。

## ほとんど変わらなかった

考えられる理由：

- 最適化前から正しくツールを使えていた
- trainsetが少ない
- 問題が簡単すぎる
- 採点フィードバックが具体的でなかった
- `auto="light"`の探索範囲では十分な改善案が見つからなかった

変化がなくても、実験として失敗とは限りません。

## 悪化した

次を確認します。

- testsetが少なく、1問の影響が大きくないか
- 採点LLMの評価が不自然でないか
- 最適化後の指示文が過度に長くなっていないか
- ツールを不要な問題でも呼ぶようになっていないか
- 1回だけの実行による偶然ではないか

---

# 29. この評価方法の限界

発表では、限界も短く述べると誠実です。

## LLMによる採点は完全な正解ではない

採点LLMも誤る可能性があります。

対策として、次を行います。

- 最適化前後で同じ採点LMを使う
- `temperature=0.0`を使う
- 代表的な数問を人間でも確認する
- 明らかに不自然な点数やフィードバックを記録する

## データ数が少ない

8件のtrainsetと8件のtestsetは、実習用の小規模例です。

発表では、次の表現が安全です。

> 小規模な検証では改善傾向を確認した。

または、

> 小規模データでは明確な改善は確認できず、追加検証が必要である。

## 同じ種類の簡単な問題に限定している

今回扱うのは、主に1種類のツールで解ける問題です。

将来の課題として、次があります。

- 複数ツールを順番に使う問題
- ツール失敗後の再試行
- レイテンシとAPIコスト
- モデル更新後の再最適化
- より複雑な業務ツールへの適用

---

# 30. よくあるエラーと対処

## `NameError: calculate_expression is not defined`

原因：メンターから渡されたツール定義セルを実行していません。

対処：`calculate_expression`、`analyze_numbers`、`convert_units`を定義するセルを上から実行します。

---

## `AttributeError: module 'dspy' has no attribute 'GEPA'`

原因：社内環境のDSPyバージョンにGEPAが含まれていない可能性があります。

対処：

```python
print(dspy.__version__)
```

の結果をメンターへ伝え、指定された環境や更新方法を確認します。

---

## `GEPA metric must accept five arguments`

原因：metricの引数が不足しています。

次の5つをそのまま残してください。

```python
def tool_use_metric(
    example,
    prediction,
    trace=None,
    pred_name=None,
    pred_trace=None,
):
```

---

## `Prediction`に`tool_calls`がない

原因：生の`dspy.ReAct`を直接GEPAへ渡している可能性があります。

次を使ってください。

```python
program = MCPToolAgent()
```

そして、

```python
student=program
```

とします。

---

## `Example`に`available_tools`がない

原因：問題を直接`dspy.Example(question=...)`で作った可能性があります。

この資料の関数を使ってください。

```python
make_example("質問")
```

---

## `tool_calls`が常に`[]`

確認すること：

1. `prediction.trajectory`を表示する
2. `tool_name_0`などが存在するか見る
3. 最初から`finish`になっていないか見る
4. `ToolQA`の指示文がツールの用途を説明しているか確認する
5. 質問が本当にツールを必要とする内容か確認する

簡単な計算では、LLMが自力で回答して`finish`する場合があります。それを採点し、GEPAで改善するのが今回の目的です。

---

## MCPのタイムアウトや接続エラー

対処：

- MCPサーバーへ接続できるネットワークか確認する
- メンターから渡された接続情報が正しいか確認する
- `num_threads=1`にする
- まず1問だけ実行する
- 同じセルを何度も連打しない

---

## Azure OpenAIのレート制限

対処：

```python
num_threads=1
```

を使います。

trainsetやtestsetを一時的に4問程度へ減らし、動作確認してから戻す方法もあります。

---

## 採点結果が毎回少し違う

LLMの出力は完全には固定されない場合があります。

- 同じ`judge_lm`を使う
- `temperature=0.0`を使う
- 最適化前後で条件を変えない
- 必要なら代表問題を複数回確認する

---

## Jupyterでコードが実行されない

セル種別が`Markdown`になっていないか確認します。

```text
セル種別をCodeへ変更
```

---

# 31. GitHubへ置く場合の簡単な構成

```text
my-intern-study/
├─ README.md
├─ dspy_gepa_tool_optimization_beginner_guide.md
└─ .gitignore
```

`.gitignore`の例：

```gitignore
.env
.ipynb_checkpoints/
*.key
*.secret
```

ただし、`.gitignore`へ書いたから必ず安全というわけではありません。すでにGitへ追加・コミットした秘密情報は、後から`.gitignore`へ書いても履歴へ残ることがあります。

---

# 32. コード全体の役割を最後に確認

```text
ToolQA
  エージェントの初期指示と入出力

extract_tool_information
  ReActのtrajectoryからツール名・引数・結果を取り出す

MCPToolAgent
  ReActを実行し、回答とツール履歴をまとめて返す

ToolUseJudge
  ツール利用を4項目で採点する設計図

run_judge
  採点用LMで1件を採点する

to_score
  点数を0.0から1.0へそろえる

get_judgment_scores
  4項目を重み付きで合計する

tool_use_metric
  GEPAへ点数と文章フィードバックを返す

evaluate_program
  複数問題を実行し、平均点を出す

GEPA.compile
  trainsetとmetricを使って指示文を改善する
```

今回の研究として最も大切な流れは、次です。

> **ツール利用履歴を観測可能にし、その履歴を含めて評価し、その評価を指示文の最適化へつなげる。**

---

# 33. 一括コピー用コード

以下は、メンターから渡された`convert_units`関数の定義が終わった後へ、上から順にコピーする全コードです。

最初は、この資料のセルごとの説明を読みながら進めることを推奨します。

```python
import json
import dspy

MAX_ITERS = 5

AVAILABLE_TOOLS_TEXT = """
1. calculate_expression
   - 1つの数式を計算するツール
   - 主な引数: expression, precision
   - 例: 128 * 1.08、sqrt(144)

2. analyze_numbers
   - 数値の配列を統計処理するツール
   - 主な引数: values, operations, second_values, outlier_method
   - 例: 平均値、中央値、標準偏差、分散

3. convert_units
   - 単位を変換するツール
   - 主な引数: value, from_unit, to_unit, category, precision
   - 例: kmからm、kgからg
""".strip()


class ToolQA(dspy.Signature):
    """
    ユーザーの質問に答えてください。

    必要な場合だけ、利用可能なツールを使ってください。
    1つの数式の計算には calculate_expression、
    数値配列の統計処理には analyze_numbers、
    単位変換には convert_units を使ってください。

    ツールを使う場合は、質問に合うツールと引数を選び、
    実行結果を最終回答へ正確に反映してください。
    不要なツールや重複したツール呼び出しは避けてください。
    """

    question: str = dspy.InputField(
        desc="ユーザーからの質問"
    )

    answer: str = dspy.OutputField(
        desc="ユーザーの質問に対する最終回答"
    )


def extract_tool_information(trajectory):
    """ReActの履歴から、外部ツールの呼び出しと結果を取り出す。"""

    tool_calls = []
    tool_results = []

    if not isinstance(trajectory, dict):
        return tool_calls, tool_results

    for index in range(MAX_ITERS):
        tool_name = trajectory.get(f"tool_name_{index}")

        if tool_name is None:
            continue

        if tool_name == "finish":
            continue

        tool_args = trajectory.get(
            f"tool_args_{index}",
            {},
        )

        observation = trajectory.get(
            f"observation_{index}",
            "",
        )

        tool_calls.append(
            {
                "order": index + 1,
                "tool_name": tool_name,
                "arguments": tool_args,
            }
        )

        tool_results.append(
            {
                "order": index + 1,
                "tool_name": tool_name,
                "result": observation,
            }
        )

    return tool_calls, tool_results


class MCPToolAgent(dspy.Module):
    """ReActの回答とツール利用履歴をまとめて返すDSPyプログラム。"""

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

        tool_calls, tool_results = extract_tool_information(
            trajectory
        )

        return dspy.Prediction(
            answer=str(
                getattr(result, "answer", "")
            ),
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
            trajectory=trajectory,
        )


program = MCPToolAgent()


def make_example(question):
    return dspy.Example(
        question=question,
        available_tools=AVAILABLE_TOOLS_TEXT,
    ).with_inputs("question")


trainset = [
    make_example(
        "25 * 16 を計算してください"
    ),
    make_example(
        "sqrt(144) を計算してください"
    ),
    make_example(
        "1, 2, 6 の平均値を求めてください"
    ),
    make_example(
        "2, 4, 8, 10 の中央値を求めてください"
    ),
    make_example(
        "1 km は何 m ですか"
    ),
    make_example(
        "2.5 kg は何 g ですか"
    ),
    make_example(
        "数字の7をそのまま答えてください"
    ),
    make_example(
        "平均値とは何かを簡潔に説明してください"
    ),
]


testset = [
    make_example(
        "128 * 1.08 を計算してください"
    ),
    make_example(
        "(18 + 7) / 5 を計算してください"
    ),
    make_example(
        "3, 7, 14 の平均値を求めてください"
    ),
    make_example(
        "10, 20, 30, 40, 50 の平均値と中央値を求めてください"
    ),
    make_example(
        "3.5 km は何 m ですか"
    ),
    make_example(
        "7500 g は何 kg ですか"
    ),
    make_example(
        "数字の42をそのまま答えてください"
    ),
    make_example(
        "中央値が外れ値の影響を受けにくい理由を簡潔に説明してください"
    ),
]


class ToolUseJudge(dspy.Signature):
    """
    AIエージェントによるツール利用を厳格に評価してください。

    以下の観点を総合的に評価します。

    1. 選択したツールがユーザー要求に適しているか
    2. 外部ツールを使用する必要があったか
    3. ツールに渡した引数が適切か
    4. 最終回答がユーザー要求を満たしているか
    5. 不要または重複したツール呼び出しがないか

    代替ツールでも同じ目的を安全かつ正確に達成できる場合は、
    必ずしも減点しないでください。

    一般的な説明や単純な値の復唱など、外部ツールが不要な質問で
    ツールを使わなかった場合は、ツール非利用を適切と評価してください。
    """

    user_query: str = dspy.InputField(
        desc="ユーザーが入力した質問"
    )

    available_tools: str = dspy.InputField(
        desc="エージェントが利用可能だったツールの名前と説明"
    )

    tool_calls: str = dspy.InputField(
        desc="実際に呼び出したツール、引数、実行順序"
    )

    tool_results: str = dspy.InputField(
        desc="各ツールの実行結果"
    )

    final_answer: str = dspy.InputField(
        desc="エージェントが生成した最終回答"
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
        desc="問題点と、エージェントの指示を改善するための具体的な助言"
    )


judge_lm = dspy.LM(
    f"azure/{AZURE_OPENAI_DEPLOYMENT}",
    api_key=AZURE_OPENAI_API_KEY,
    api_base=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION,
    temperature=0.0,
)

judge = dspy.ChainOfThought(
    ToolUseJudge
)


def run_judge(example, prediction):
    with dspy.context(lm=judge_lm):
        return judge(
            user_query=example.question,
            available_tools=example.available_tools,
            tool_calls=prediction.tool_calls,
            tool_results=prediction.tool_results,
            final_answer=prediction.answer,
        )


def to_score(value):
    """値を0.0から1.0のfloatへ変換する。"""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0

    return max(
        0.0,
        min(1.0, number),
    )


def get_judgment_scores(judgment):
    tool_selection = to_score(
        judgment.tool_selection_score
    )

    tool_necessity = to_score(
        judgment.tool_necessity_score
    )

    argument = to_score(
        judgment.argument_score
    )

    task_success = to_score(
        judgment.task_success_score
    )

    total = (
        0.40 * tool_selection
        + 0.20 * tool_necessity
        + 0.15 * argument
        + 0.25 * task_success
    )

    return {
        "tool_selection": tool_selection,
        "tool_necessity": tool_necessity,
        "argument": argument,
        "task_success": task_success,
        "total": total,
    }


def tool_use_metric(
    example,
    prediction,
    trace=None,
    pred_name=None,
    pred_trace=None,
):
    judgment = run_judge(
        example,
        prediction,
    )

    scores = get_judgment_scores(
        judgment
    )

    return dspy.Prediction(
        score=scores["total"],
        feedback=judgment.feedback,
    )


def evaluate_program(
    program_to_evaluate,
    dataset,
    title,
):
    total_tool_selection = 0.0
    total_tool_necessity = 0.0
    total_argument = 0.0
    total_task_success = 0.0
    total_score = 0.0

    print("=" * 70)
    print(title)
    print("=" * 70)

    for number, example in enumerate(
        dataset,
        start=1,
    ):
        prediction = program_to_evaluate(
            question=example.question
        )

        judgment = run_judge(
            example,
            prediction,
        )

        scores = get_judgment_scores(
            judgment
        )

        total_tool_selection += scores[
            "tool_selection"
        ]
        total_tool_necessity += scores[
            "tool_necessity"
        ]
        total_argument += scores[
            "argument"
        ]
        total_task_success += scores[
            "task_success"
        ]
        total_score += scores["total"]

        print(f"\n{number}. 質問: {example.question}")
        print(f"   回答: {prediction.answer}")
        print(f"   ツール呼び出し: {prediction.tool_calls}")
        print(f"   ツール結果: {prediction.tool_results}")
        print(
            "   スコア: "
            f"選択={scores['tool_selection']:.2f}, "
            f"必要性={scores['tool_necessity']:.2f}, "
            f"引数={scores['argument']:.2f}, "
            f"達成度={scores['task_success']:.2f}, "
            f"総合={scores['total']:.2f}"
        )
        print(f"   フィードバック: {judgment.feedback}")

    count = len(dataset)

    result = {
        "tool_selection": total_tool_selection / count,
        "tool_necessity": total_tool_necessity / count,
        "argument": total_argument / count,
        "task_success": total_task_success / count,
        "total": total_score / count,
    }

    print("\n--- 平均 ---")
    print(
        f"ツール選択: {result['tool_selection']:.3f}"
    )
    print(
        f"ツール必要性: {result['tool_necessity']:.3f}"
    )
    print(
        f"引数: {result['argument']:.3f}"
    )
    print(
        f"タスク達成度: {result['task_success']:.3f}"
    )
    print(
        f"総合: {result['total']:.3f}"
    )

    return result


baseline_result = evaluate_program(
    program,
    testset,
    "最適化前",
)


reflection_lm = dspy.LM(
    f"azure/{AZURE_OPENAI_DEPLOYMENT}",
    api_key=AZURE_OPENAI_API_KEY,
    api_base=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION,
    temperature=0.0,
)

optimizer = dspy.GEPA(
    metric=tool_use_metric,
    auto="light",
    reflection_lm=reflection_lm,
    num_threads=1,
)

optimized_agent = optimizer.compile(
    student=program,
    trainset=trainset,
)


optimized_result = evaluate_program(
    optimized_agent,
    testset,
    "最適化後",
)


def print_comparison(before, after):
    print("=" * 70)
    print("最適化前後の比較")
    print("=" * 70)

    print(
        "ツール選択: "
        f"{before['tool_selection']:.3f}"
        " -> "
        f"{after['tool_selection']:.3f}"
    )

    print(
        "ツール必要性: "
        f"{before['tool_necessity']:.3f}"
        " -> "
        f"{after['tool_necessity']:.3f}"
    )

    print(
        "引数: "
        f"{before['argument']:.3f}"
        " -> "
        f"{after['argument']:.3f}"
    )

    print(
        "タスク達成度: "
        f"{before['task_success']:.3f}"
        " -> "
        f"{after['task_success']:.3f}"
    )

    print(
        "総合: "
        f"{before['total']:.3f}"
        " -> "
        f"{after['total']:.3f}"
        "  変化="
        f"{after['total'] - before['total']:+.3f}"
    )


print_comparison(
    baseline_result,
    optimized_result,
)


def show_instructions(program_to_show, title):
    print("=" * 70)
    print(title)
    print("=" * 70)

    for name, predictor in program_to_show.named_predictors():
        print(f"\n予測器名: {name}")
        print(predictor.signature.instructions)


show_instructions(
    program,
    "最適化前の指示文",
)

show_instructions(
    optimized_agent,
    "最適化後の指示文",
)
```

---

# 34. 参考にしたDSPy公式資料

社内環境のDSPyバージョンによって、細かなAPIや表示形式が異なる可能性があります。

- DSPy Module: https://dspy.ai/api/modules/Module/
- DSPy ReAct: https://dspy.ai/api/modules/ReAct/
- DSPy GEPA: https://dspy.ai/api/optimizers/GEPA/overview/
- DSPy metrics and evaluation: https://dspy.ai/diving-deeper/metrics-and-evaluation/
- DSPy context: https://dspy.ai/api/utils/context/

