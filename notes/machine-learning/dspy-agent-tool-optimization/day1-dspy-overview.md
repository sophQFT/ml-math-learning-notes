# Day 1 — DSPyの調査

> 目的：DSPyのコードを見たときに、「今どの部品を定義しているのか」を追える状態にする。

## 0. 最初に全体像

DSPyは、LLMアプリケーションを「巨大なプロンプト文字列」として扱うのではなく、**入出力仕様・推論方法・ツール・評価指標・最適化処理をPythonの部品として構成する**ためのフレームワークである。

今回まず覚える流れは次の通り。

```text
LM
 ↓
Signature      : 何を入力して何を出力するか
 ↓
Module         : その仕事をどう実行するか
 ↓
Tool / ReAct   : 外部機能を使うAgent
 ↓
Prediction     : 実行結果
 ↓
Metric         : 良し悪しを数値化
 ↓
Optimizer      : Metricが良くなるように改善
 ↓
compiled program
```

公式トップページも、DSPyを「structured signaturesでタスクを表現し、modulesで実行方法を選び、metricsに対してoptimizersでcompileする」枠組みとして説明している。

---

## 1. まず環境を確認する

```python
import dspy

print(dspy.__version__)
```

JupyterLabでは、知らないオブジェクトが出てきたら次も便利。

```python
dspy.ReAct?
```

さらに実装まで見たい場合。

```python
dspy.ReAct??
```

Python標準の方法なら、

```python
help(dspy.ReAct)
```

---

## 2. Signatureとは何か

### 2.1 一言で

**Signature = LLMにやらせたい仕事の入出力仕様。**

数学の関数

$$
f:X\rightarrow Y
$$

に近い感覚で、

```text
question → answer
```

のような写像を定義する。

最短では文字列で書ける。

```python
qa = dspy.Predict("question -> answer")
```

より明示的に書くなら、class-based Signatureを使う。

```python
class QA(dspy.Signature):
    """質問に簡潔かつ正確に答える。"""

    question: str = dspy.InputField()
    answer: str = dspy.OutputField()
```

### 2.2 コードを分解

```python
class QA(dspy.Signature):
```

- `class QA`: `QA`というクラスを定義する。
- `(dspy.Signature)`: DSPyのSignatureを継承する。
- 今は「QAという入出力仕様書を作る」と理解すればよい。

```python
question: str = dspy.InputField()
```

- フィールド名：`question`
- 型：`str`
- 入力：`InputField`

```python
answer: str = dspy.OutputField()
```

- フィールド名：`answer`
- 型：`str`
- 出力：`OutputField`

### 2.3 docstringは何に使われるか

```python
"""質問に簡潔かつ正確に答える。"""
```

Class-based Signatureのdocstringは、DSPyがLMへ渡す**task instructions**として利用する。

つまりSignatureは、

```text
入力フィールド
出力フィールド
型
タスクの説明
```

を構造化して持つ。

公式ドキュメントでは、field descriptionも追加できる。

```python
class QA(dspy.Signature):
    """質問に答える。"""

    question: str = dspy.InputField(
        desc="ユーザーから与えられる質問"
    )
    answer: str = dspy.OutputField(
        desc="質問に対する最終回答"
    )
```

ただし、説明を長大な手動プロンプトに戻してしまうのではなく、簡潔な入出力定義を基本にする。

---

## 3. Moduleとは何か

### 3.1 SignatureとModuleの違い

Signatureは、

> **何をするか**

Moduleは、

> **どうやってするか**

を表す。

同じSignatureでもModuleを変えられる。

```python
predictor = dspy.Predict(QA)
```

通常の予測。

```python
predictor = dspy.ChainOfThought(QA)
```

推論過程を加える。

```python
agent = dspy.ReAct(QA, tools=[...])
```

ツールを使うAgentにする。

イメージ：

```text
                 ┌─ Predict
QA Signature ────┼─ ChainOfThought
                 └─ ReAct
```

Signatureを保ったまま、実行戦略を交換できることがDSPyのモジュール性の一つ。

---

## 4. Toolとは何か

DSPyのToolは基本的に**普通のPython関数**として書ける。

```python
def calculator(x: int, y: int) -> int:
    """2つの整数を加算する。"""
    return x + y
```

### 4.1 なぜ型ヒントを書くのか

Pythonとしては、

```python
def calculator(x, y):
    return x + y
```

でも動く。

しかし、

```python
def calculator(x: int, y: int) -> int:
```

とすると、

```text
x      : int
y      : int
return : int
```

という型情報が明示される。

### 4.2 なぜdocstringを書くのか

```python
"""2つの整数を加算する。"""
```

は関数のdocstring。

確認：

```python
help(calculator)
```

または、

```python
calculator.__doc__
calculator.__annotations__
```

### 4.3 DSPyではToolのメタデータが重要

公式のTools with ReActでは、DSPyがPython関数の

- 関数名
- 引数名
- 型
- docstring

を読み、LMにToolの説明として提示すると説明されている。

したがって、

```python
def f(a, b):
    ...
```

より、

```python
def add_integers(x: int, y: int) -> int:
    """Add two integers and return the sum."""
    ...
```

の方が、Toolの用途と引数が明確。

**重要：これはDSPy全体の定義ではない。**

DSPyはさらにSignature、Module、評価、Optimizerなどを組み合わせるフレームワークであり、「型ヒントを書くこと」そのものがDSPyなのではない。

---

## 5. ReActとは何か

ReAct = **Reasoning + Acting**。

Agentが、

```text
考える
 ↓
Toolを選ぶ
 ↓
引数を決める
 ↓
Toolを実行
 ↓
Observationを見る
 ↓
次に何をするか決める
 ↓
...
 ↓
finish
```

というループを行う。

### 5.1 最小例

```python
def calculator(x: int, y: int) -> int:
    """2つの整数を加算する。"""
    return x + y


class QA(dspy.Signature):
    """必要ならツールを利用して質問に答える。"""

    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


agent = dspy.ReAct(
    QA,
    tools=[calculator],
    max_iters=5
)
```

### 5.2 `tools=[calculator]` の `[]`

`[]`はPythonのlist。

```python
tools = [calculator]
```

は、

> Agentに利用可能なToolとして`calculator`を渡す。

という意味。

複数なら、

```python
tools=[calculator, search, lookup]
```

となる。

### 5.3 `max_iters`

```python
max_iters=5
```

はReActループの最大反復数。

無限にToolを呼び続けないための上限でもある。

---

## 6. ReAct内部で何が起きているか

現行の`dspy.ReAct`は概念的に各ステップで、

```text
next_thought
next_tool_name
next_tool_args
```

をLMに生成させ、Toolを呼び、その結果を`observation`としてtrajectoryへ追加する。

公式APIの実装では、おおむね次の情報がtrajectoryに記録される。

```text
thought_0
tool_name_0
tool_args_0
observation_0

thought_1
tool_name_1
tool_args_1
observation_1
...
```

終了時には`finish`という特別なToolが使われる。`finish`は通常、ユーザーが定義しなくてもReAct側で追加される。

---

## 7. Predictionとtrajectory

Agentを実行すると、

```python
result = agent(question="100と250を足してください")
```

`result`は単なる文字列ではなく、DSPyのPrediction。

回答：

```python
print(result.answer)
```

Tool利用履歴：

```python
print(result.trajectory)
```

### なぜtrajectoryを見るのか

最終回答だけ合っていても、

- 本当に正しいToolを使ったか
- Toolが必要なのに使わなかったか
- 不要なToolを使っていないか
- 引数が正しかったか
- 同じToolを何回も無駄に呼んでいないか

は分からない。

今回のような**Tool利用性能評価ではtrajectoryが重要な観測対象**になる。

---

## 8. Example / trainset / devset / testset

DSPyでは学習・評価用の例を`dspy.Example`として持てる。

```python
example = dspy.Example(
    question="100と200を足してください",
    answer="300",
    expected_tool="calculator"
).with_inputs("question")
```

多数の例をlistにする。

```python
examples = [
    ...,
    ...,
]
```

一般には、

```text
trainset
  ↓
Optimizerが改善に利用

devset / valset
  ↓
候補比較・開発中の評価

testset
  ↓
最後の未見評価
```

と分ける。

**最適化に使ったデータだけで性能向上を主張しない**ことが重要。

---

## 9. Metricとは何か

Metricは、

> **「良い出力」とは何かをPython関数で定義するもの。**

最小例：

```python
def exact_match(example, pred, trace=None):
    return example.answer == pred.answer
```

DSPy OptimizerはMetricを使って、候補となるプログラムの良し悪しを比較する。

Tool利用では回答正解率だけでなく、

- Tool選択
- Tool利用の必要性判断
- 引数
- 呼び出し順
- 呼び出し回数
- エラー率

も評価候補になる。

---

## 10. Optimizerとは何か

Optimizerは、Metricが高くなるようにDSPyプログラムを改善する。

現行公式ガイドでは、Optimizerが調整する主な対象は、

1. instructions
2. demos（few-shot examples）
3. weights

のいずれか、または組み合わせ。

多くのOptimizerはモデルの重みを変えず、**instructionsやdemosを最適化するprompt-only optimization**である。

概念図：

```text
program
  +
trainset
  +
metric
  ↓
Optimizer.compile()
  ↓
optimized program
```

例：

```python
optimizer = dspy.BootstrapFewShot(
    metric=my_metric
)

optimized_agent = optimizer.compile(
    agent,
    trainset=trainset
)
```

重要：

```python
optimized_agent = ...
```

は、元の`agent`を書き換えるというより、compileされた新しいプログラムを受け取ると考える。

---

## 11. 主なOptimizerのざっくり位置づけ

| Optimizer | 主に調整 | 初学者向け理解 |
|---|---|---|
| `BootstrapFewShot` | demos | まず試す基準。成功例をfew-shot化 |
| `BootstrapFewShotWithRandomSearch` | demos | 複数のdemo候補を探索 |
| `COPRO` | instructions | 指示文を改善 |
| `GEPA` | instructions | feedbackを利用した反省・進化的探索 |
| `MIPROv2` | instructions + demos | 両方を探索 |
| `SIMBA` | instructions + demos | 失敗例を見ながら反復改善 |
| `AvatarOptimizer` | agent instructions | Agent/tool-use向けの選択肢 |

公式の2026年Optimizer選択ガイドでは、Agent / tool-use taskに`AvatarOptimizer`または`GEPA`が候補として挙げられている。ただし、**実習で指定されたOptimizerがあるならその指定を優先する。**

---

## 12. Day 1の最小チートシート

```python
import dspy

# 1. Signature
class Task(dspy.Signature):
    """必要ならToolを利用して質問に答える。"""
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

# 2. Tool
def my_tool(query: str) -> str:
    """Toolが何をするかを簡潔に書く。"""
    return "result"

# 3. Agent
agent = dspy.ReAct(
    Task,
    tools=[my_tool],
    max_iters=5
)

# 4. Run
pred = agent(question="...")

# 5. Output
print(pred.answer)

# 6. Tool-use history
print(pred.trajectory)
```

---

## 13. コードを見たらこの順で読む

実習中に長いコードが来ても、上から全部理解しようとしない。

1. `dspy.Signature` / `InputField` / `OutputField` を探して入出力を確認。
2. `def` と `tools=[...]` を探してTool一覧を確認。
3. `dspy.ReAct` / `dspy.Module` / `forward` を探してAgent本体を確認。
4. `dspy.Example` / `trainset` / `devset` / `testset` を確認。
5. `def metric(...)` を探して「良い」の定義を確認。
6. `optimizer.compile(...)` を探して最適化箇所を確認。

---

## 14. Day 1で最低限説明できればよい言葉

- **Signature**：タスクの入出力仕様
- **Module**：Signatureをどう実行するか
- **Tool**：Agentが呼び出せるPython関数など
- **ReAct**：推論とTool実行を反復するAgent方式
- **Prediction**：DSPyの実行結果
- **trajectory**：Tool選択・引数・結果などの履歴
- **Example**：学習・評価用データ
- **Metric**：良し悪しの採点関数
- **Optimizer**：Metricが改善するようプログラムをcompileする仕組み

---

## 15. 公式資料

2026-09-01確認。

- Overview  
  https://dspy.ai/
- Class-based signatures  
  https://dspy.ai/getting-started/class-based-signatures/
- Tools with ReAct  
  https://dspy.ai/getting-started/react-and-tools/
- ReAct API  
  https://dspy.ai/api/modules/ReAct/
- Metrics  
  https://dspy.ai/getting-started/metrics/
- Optimizers  
  https://dspy.ai/diving-deeper/choosing-an-optimizer/
