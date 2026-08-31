# Day 2 — AIエージェントのツール利用最適化：調査と実装

> 目的：Agentを「動かす」だけでなく、Tool利用を観測・評価し、DSPy Optimizerで改善できる実験系を作る。

## 0. 「ツール利用最適化」とは何を最適化するのか

AIエージェントのTool利用は、単に「Toolを呼べたか」ではない。

最低でも次の観点に分ける。

```text
ユーザー入力
   ↓
[1] Toolが必要か判断
   ↓
[2] 正しいToolを選択
   ↓
[3] 正しい引数を生成
   ↓
[4] 必要なら正しい順序で複数Toolを実行
   ↓
[5] Tool結果を使って回答
   ↓
[6] 不要な呼び出しをせず終了
```

したがってTool利用性能は、例えば以下で評価できる。

- Task success / Answer accuracy
- Tool selection accuracy
- Tool necessity accuracy
- Argument accuracy
- Invalid tool-call rate
- Tool execution error rate
- Average tool calls
- Redundant call rate
- Latency
- Token / API cost

---

## 1. Day 2の実装方針

最初からOptimizerを掛けない。

```text
A. Baseline Agentを作る
        ↓
B. Baselineを評価する
        ↓
C. Failure caseを見る
        ↓
D. Metricを決める
        ↓
E. DSPy Optimizerでcompile
        ↓
F. 同じ評価条件で再評価
        ↓
G. Before / After比較
```

**Baselineなしの最適化は、改善したか判断できない。**

---

# Part A — 最小Agentを作る

## 2. LM設定

会社・研究室環境では、LM設定が既に用意されている場合がある。その場合は提供された設定を使う。

```python
import dspy

print(dspy.__version__)
```

自分で設定する場合の概念形：

```python
# 例。実際のprovider/modelは利用環境に合わせる。
# lm = dspy.LM("<provider>/<model>")
# dspy.configure(lm=lm)
```

**APIキーをNotebookやGitHubに直書きしない。**

---

## 3. Toy Toolsを定義する

```python
def add_numbers(x: int, y: int) -> int:
    """Add two integers and return the sum."""
    return x + y


def multiply_numbers(x: int, y: int) -> int:
    """Multiply two integers and return the product."""
    return x * y


def lookup_product(product_name: str) -> str:
    """Look up basic information for a product name."""
    database = {
        "alpha": "Alpha: category A",
        "beta": "Beta: category B",
    }
    return database.get(product_name.lower(), "not found")
```

確認：

```python
help(add_numbers)
print(add_numbers.__annotations__)
print(add_numbers.__doc__)
```

---

## 4. Signatureを定義

```python
class ToolQA(dspy.Signature):
    """Use tools when necessary and answer the user's question accurately."""

    question: str = dspy.InputField()
    answer: str = dspy.OutputField()
```

---

## 5. ReAct Agentを作る

```python
agent = dspy.ReAct(
    ToolQA,
    tools=[
        add_numbers,
        multiply_numbers,
        lookup_product,
    ],
    max_iters=5,
)
```

見るポイント：

- `ToolQA`：何を入出力するか
- `tools=[...]`：何を使えるか
- `max_iters=5`：最大何ステップ動けるか

---

## 6. 1問だけ動かしてtrajectoryを見る

```python
pred = agent(
    question="12と30を足してください。"
)

print(pred.answer)
print(pred.trajectory)
```

**最初は正解だけでなくtrajectoryを必ず見る。**

---

# Part B — 評価データを作る

## 7. Exampleを作る

```python
examples = [
    dspy.Example(
        question="12と30を足してください。",
        answer="42",
        expected_tool="add_numbers",
        max_tool_calls=1,
    ).with_inputs("question"),

    dspy.Example(
        question="7と8を掛けてください。",
        answer="56",
        expected_tool="multiply_numbers",
        max_tool_calls=1,
    ).with_inputs("question"),

    dspy.Example(
        question="alphaの商品情報を調べてください。",
        answer="Alpha: category A",
        expected_tool="lookup_product",
        max_tool_calls=1,
    ).with_inputs("question"),

    dspy.Example(
        question="こんにちは。",
        answer="こんにちは。",
        expected_tool="none",
        max_tool_calls=0,
    ).with_inputs("question"),
]
```

`expected_tool`はDSPyの必須フィールドではない。

**Tool利用を評価するために追加した教師ラベル**である。

---

## 8. train / dev / testを分ける

```python
trainset = examples[:2]
devset = examples[2:3]
testset = examples[3:]
```

実際にはもっと多くの例を使う。

```text
trainset
  Optimizerが見る

devset
  開発中の比較に使う

testset
  最後まで触らない
```

---

# Part C — trajectoryを解析する

## 9. 使用Tool名を取り出す

現行`dspy.ReAct`のtrajectoryは、Tool名を`tool_name_0`, `tool_name_1`, ...として持つ。

```python
def get_used_tools(pred):
    """Predictionのtrajectoryからfinish以外のTool名を順番に取り出す。"""
    trajectory = pred.trajectory

    used_tools = []

    for key, value in trajectory.items():
        if key.startswith("tool_name_") and value != "finish":
            used_tools.append(value)

    return used_tools
```

使用例：

```python
used_tools = get_used_tools(pred)
print(used_tools)
```

### Python初心者向け分解

```python
for key, value in trajectory.items():
```

辞書を1組ずつ取り出す。

```python
key.startswith("tool_name_")
```

はkeyが`tool_name_`から始まるか判定。

```python
used_tools.append(value)
```

listの末尾にTool名を追加。

---

# Part D — Metricを設計する

## 10. Tool選択Accuracy

```python
def tool_choice_metric(example, pred, trace=None):
    used_tools = get_used_tools(pred)
    expected = example.expected_tool

    if expected == "none":
        return len(used_tools) == 0

    if len(used_tools) == 0:
        return False

    return used_tools[0] == expected
```

これは、

> 最初に選択したToolが正しいか

を評価する。

---

## 11. Answer Accuracy

```python
def normalize(text):
    return str(text).strip().lower()


def answer_metric(example, pred, trace=None):
    return normalize(example.answer) == normalize(pred.answer)
```

実タスクでは完全一致が厳しすぎることがある。

その場合は、

- ルールベース
- 数値誤差許容
- semantic similarity
- LLM judge
- task-specific validator

などを考える。

---

## 12. 不要なTool呼び出し数

```python
def extra_tool_calls(example, pred):
    used_tools = get_used_tools(pred)

    extra = len(used_tools) - example.max_tool_calls

    return max(0, extra)
```

正解Toolを使っても、同じToolを何度も呼ぶなら効率は悪い。

---

## 13. Composite Metric

$$
S
=
0.6 A
+
0.4 T
-
0.05 C
$$

- $A$: answer correctness
- $T$: tool-choice correctness
- $C$: extra tool calls

```python
def combined_metric(example, pred, trace=None):
    answer_ok = answer_metric(example, pred)
    tool_ok = tool_choice_metric(example, pred)
    extra = extra_tool_calls(example, pred)

    score = (
        0.6 * float(answer_ok)
        + 0.4 * float(tool_ok)
        - 0.05 * extra
    )

    return max(0.0, score)
```

### 注意

`0.6`, `0.4`, `0.05`に絶対的な正解はない。

**評価設計そのものが研究・検証対象。**

---

# Part E — Baselineを測る

## 14. まず手動ループ

```python
scores = []

for example in devset:
    pred = agent(question=example.question)

    score = combined_metric(
        example,
        pred
    )

    scores.append(score)

    print("Question:", example.question)
    print("Answer:", pred.answer)
    print("Tools:", get_used_tools(pred))
    print("Score:", score)
    print("---")
```

平均：

```python
baseline_score = sum(scores) / len(scores)

print("Baseline:", baseline_score)
```

---

## 15. DSPyのEvaluate

```python
evaluate = dspy.Evaluate(
    devset=devset,
    metric=combined_metric,
    display_progress=True,
    display_table=True,
)

baseline_score = evaluate(agent)
```

---

# Part F — Failure Analysis

## 16. Optimizerの前に失敗を分類する

| Failure | 例 | 改善候補 |
|---|---|---|
| Tool不要なのに呼ぶ | 挨拶で検索 | Agent instruction |
| Toolが必要なのに呼ばない | 計算を暗算 | instruction / examples |
| Tool選択ミス | 足算でmultiply | Tool description / optimizer |
| 引数ミス | 不正な値 | type / parameter description |
| redundant call | 同じToolを3回 | instruction / metric penalty |
| early finish | 情報不足でfinish | Agent instruction |
| 長いloop | Toolを反復 | max_iters / metric |
| 正しいToolだが回答ミス | observationを無視 | final-answer instruction |

---

# Part G — Tool descriptionを実験する

## 17. Tool metadataによる差

条件A：

```python
def search(q: str) -> str:
    """Search."""
    ...
```

条件B：

```python
def search_product_catalog(product_name: str) -> str:
    """Search the product catalog for exact product information by product name."""
    ...
```

DSPy公式ではToolの名前・引数・docstringがLMへ提示される。

したがって、

> **Tool metadataがTool選択性能に与える影響**

は自然な実験軸になる。

| 条件 | Tool名 | docstring | Tool Accuracy |
|---|---|---|---:|
| A | 曖昧 | 短い | 測定 |
| B | 明確 | 明確 | 測定 |
| C | 明確 | 過剰に長い | 測定 |

結果は実測する。事前に「Bが絶対よい」と決めつけない。

---

# Part H — Optimizer

## 18. BootstrapFewShot

公式Optimizerガイドでは、「何から始めればよいか分からない」場合の基準として`BootstrapFewShot`が挙げられている。

```python
optimizer = dspy.BootstrapFewShot(
    metric=combined_metric,
)

optimized_agent = optimizer.compile(
    agent,
    trainset=trainset,
)
```

評価：

```python
optimized_score = evaluate(
    optimized_agent
)

print("Baseline:", baseline_score)
print("Optimized:", optimized_score)
```

---

## 19. GEPAを使う場合

2026年公式ガイドではAgent / tool-use taskの候補として`GEPA`も挙げられている。

```python
# 実際のDSPyバージョンと実習環境の指定に合わせること
optimizer = dspy.GEPA(
    metric=gepa_metric,
    auto="light",
)

optimized_agent = optimizer.compile(
    agent,
    trainset=trainset,
)
```

現行公式ガイドではGEPAは自然言語`feedback`を利用できる。

概念例：

```python
def gepa_metric(example, pred, trace=None):
    used_tools = get_used_tools(pred)

    tool_ok = tool_choice_metric(example, pred)
    answer_ok = answer_metric(example, pred)

    score = (
        0.5 * float(tool_ok)
        + 0.5 * float(answer_ok)
    )

    feedback_parts = []

    if not tool_ok:
        feedback_parts.append(
            f"Expected tool was {example.expected_tool}, "
            f"but used tools were {used_tools}."
        )

    if not answer_ok:
        feedback_parts.append(
            "The final answer did not match the expected answer."
        )

    if not feedback_parts:
        feedback_parts.append(
            "Tool selection and final answer were both correct."
        )

    return dspy.Prediction(
        score=score,
        feedback=" ".join(feedback_parts),
    )
```

**実際のAPIは`help(dspy.GEPA)`で確認する。**

---

## 20. MIPROv2 / SIMBA

Tool利用では、

- instructions
- demos

の両方が影響する可能性がある。

その場合の候補：

```python
dspy.MIPROv2(...)
dspy.SIMBA(...)
```

2026年公式のAdvanced Tool Use tutorialでは、カスタムTool-use Agentに対してSIMBAを使い、development accuracyを35%から60.7%へ改善した例がある。

これは**公式Tutorial固有のデータ・モデル・評価条件**であり、自分のタスクで同じ改善率になるという意味ではない。

---

# Part I — Before / After

## 21. 最低限の表

| Metric | Baseline | Optimized | Difference |
|---|---:|---:|---:|
| Task success |  |  |  |
| Tool selection accuracy |  |  |  |
| No-tool accuracy |  |  |  |
| Invalid call rate |  |  |  |
| Avg. tool calls |  |  |  |
| Avg. latency |  |  |  |

一つの総合Scoreだけでなく、分解した指標も残す。

---

# Part J — 実験案

## 22. 小さく試せる実験

1. Tool description：曖昧 vs 明確。
2. 類似Tool：似た検索Toolを複数置く。
3. Tool不要問題：Toolを呼ばないことが正解の例。
4. 引数：整数、文字列、Literal、複数引数。
5. `max_iters`：性能・tool calls・latency。
6. Optimizer：Baseline / BootstrapFewShot / GEPA / MIPROv2 / SIMBA。

---

# Part K — デバッグ

## 23. 動かないときの順番

1. Pythonエラーか確認。
2. Tool単体で動くか。
3. Signatureの入出力名は一致しているか。
4. Agentを1問だけ実行。
5. `pred.trajectory`を見る。
6. expectedとactual toolを並べる。
7. Metricを単体実行。
8. Baselineを測る。
9. 最後にOptimizer。

---

# Part L — Pythonで見慣れない構文

## 24. `lambda`

```python
lambda x: x + 1
```

は短い無名関数。

## 25. list comprehension

```python
[x * 2 for x in values]
```

はfor文でlistを作る短縮形。

## 26. dict comprehension

```python
{name: fn for name, fn in functions.items()}
```

はfor文で辞書を作る短縮形。

## 27. `*args`, `**kwargs`

- `*args`：位置引数をまとめて受け取る。
- `**kwargs`：キーワード引数をまとめて受け取る。

Frameworkコードで頻出する。

---

# Part M — チェックリスト

## 28. 実装前

- [ ] DSPyバージョン確認
- [ ] LM設定確認
- [ ] Signature確認
- [ ] Tool一覧確認
- [ ] 各Toolを単体実行
- [ ] Tool名・型・docstring確認

## 29. Baseline

- [ ] 1問実行
- [ ] trajectory確認
- [ ] 複数問実行
- [ ] Baseline Score保存
- [ ] Failure case分類

## 30. Optimization

- [ ] Metricが目的と一致しているか
- [ ] train/dev/testを混ぜていないか
- [ ] Optimizerの対象はinstructionsかdemosか
- [ ] compile前後を同じ条件で比較
- [ ] 乱数seedやモデル設定を記録

---

# Part N — LLMOpsへの接続

運用では、

```text
Agent version
Model version
Tool version
Prompt / compiled program
Dataset version
Metric
Evaluation result
Latency / cost
```

を追跡する必要がある。

DSPyのOptimizerは、

```text
評価
 ↓
改善
 ↓
再評価
```

をプログラムとして回せるため、LLMOpsの評価・改善サイクルに組み込みやすい可能性がある。

ただし実運用適用では、

- optimization cost
- reproducibility
- version control
- model/tool change時の再評価
- security
- observability
- rollback

まで含めて評価する必要がある。

---

# Part O — Day 2最小コード骨格

```python
import dspy

def add_numbers(x: int, y: int) -> int:
    """Add two integers and return the sum."""
    return x + y


def multiply_numbers(x: int, y: int) -> int:
    """Multiply two integers and return the product."""
    return x * y


class ToolQA(dspy.Signature):
    """Use tools when necessary and answer accurately."""

    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


agent = dspy.ReAct(
    ToolQA,
    tools=[add_numbers, multiply_numbers],
    max_iters=5,
)


example = dspy.Example(
    question="12と30を足してください。",
    answer="42",
    expected_tool="add_numbers",
    max_tool_calls=1,
).with_inputs("question")


pred = agent(
    question=example.question
)

print(pred.answer)
print(pred.trajectory)


def get_used_tools(pred):
    used_tools = []

    for key, value in pred.trajectory.items():
        if key.startswith("tool_name_") and value != "finish":
            used_tools.append(value)

    return used_tools


def metric(example, pred, trace=None):
    used_tools = get_used_tools(pred)

    tool_ok = (
        len(used_tools) > 0
        and used_tools[0] == example.expected_tool
    )

    answer_ok = (
        str(pred.answer).strip()
        == str(example.answer).strip()
    )

    return (
        0.4 * float(tool_ok)
        + 0.6 * float(answer_ok)
    )
```

---

# Part P — 参考資料

2026-09-01確認。

1. Tools with ReAct  
   https://dspy.ai/getting-started/react-and-tools/

2. ReAct API  
   https://dspy.ai/api/modules/ReAct/

3. Metrics  
   https://dspy.ai/getting-started/metrics/

4. Optimizer selection  
   https://dspy.ai/diving-deeper/choosing-an-optimizer/

5. Advanced Tool Use tutorial  
   https://dspy.ai/tutorials/tool_use/

6. Class-based signatures  
   https://dspy.ai/getting-started/class-based-signatures/

---

## 最後に覚える1文

**Tool利用最適化とは、「最終回答を良くする」だけではなく、Agentが必要なときに適切なToolを、適切な引数・順序・回数で利用できるようにし、その改善をMetricで検証することである。**
