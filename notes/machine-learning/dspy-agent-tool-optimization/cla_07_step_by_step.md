# metric の続きから発表まで

Python初心者むけ。
**上から順にコピペするだけ**で
発表まで到達できるようにしてある。

---

## 使い方

Jupyter Notebook なら
1ステップ＝1セルで貼る。

`.py` ファイルでも、
上から順に並べれば同じように動く。

**必ず順番どおりに実行すること。**
前のステップで作った変数を
後ろで使っている。

---

## 全体の流れ

```
STEP 0  下ごしらえ
STEP 1  データを増やす
STEP 2  metric を直す      ★発表の売り
STEP 3  評価の関数を作る
STEP 4  最適化「前」を測る  ★これが命
STEP 5  失敗を目で見る
STEP 6  最適化する
STEP 7  最適化「後」を測る
STEP 8  指示文の変化を見る  ★発表の山場
STEP 9  発表用の表を出す
```

所要時間の目安は
STEP 0〜5 で半日、
STEP 6〜9 で半日。

---

## STEP 0 — 下ごしらえ

### 0-1. インデントを直す

`def calculate_expression` と
`class ToolQA` が4字下がっている。
**行頭に寄せる。**

下がっていると
`call_mcp_tool` や `convert_units` の
「中」に入ってしまい、
`NameError` になる。

### 0-2. LM に temperature を足す

```python
lm = dspy.LM(
    f"azure/{AZURE_OPENAI_DEPLOYMENT}",
    api_key=AZURE_OPENAI_API_KEY,
    api_base=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION,
    temperature=0.0,      # ← 追加
)
dspy.configure(lm=lm)
```

`temperature=0.0` にしないと
実行のたびに答えが変わる。

そうなると
「最適化で良くなった」のか
「たまたま」なのか
区別がつかなくなる。

### 0-3. agent を作り直す

```python
agent = dspy.ReAct(
    ToolQA,
    tools=[
        calculate_expression,
        analyze_numbers,
        convert_units,
    ],
    max_iters=5,          # ← 追加
)
```

`max_iters` を書かないと既定は20。
MCPサーバに20回HTTPが飛ぶので、
5に絞る。

### 0-4. 動作確認

```python
r = agent(question="25 * 16 は？")
print(r.answer)
print(r.trajectory)
```

`trajectory` に
`tool_name_0` が出ていればOK。

**ここが通らないうちは先に進まない。**

---

## STEP 1 — データを増やす

今のデータ9件は
「数式なら計算ツール」
「配列なら統計ツール」
「単位なら変換ツール」と
**分かれすぎていて簡単すぎる**。

このままだと
最初から満点に近くなり、
**最適化しても改善しないので
発表で見せるものが無くなる。**

そこで2種類を足す。

- **複合問題**（2つのツールが要る）
- **ツール不要問題**（呼ばない判断）

さらに
**学習用と評価用に分ける**。
学習に使ったデータで評価すると
数字が嘘になるため。

```python
CALC = "calculate_expression"
STAT = "analyze_numbers"
CONV = "convert_units"


def ex(question, answer, tools):
    """例を1つ作る便利関数"""
    return dspy.Example(
        question=question,
        answer=answer,
        expected_tools=tools,
    ).with_inputs("question")


# 学習用（最適化器に見せる）
trainset = [
    ex("25 * 16 を計算してください",
       "400", [CALC]),
    ex("100 / 4 を計算してください",
       "25", [CALC]),
    ex("1, 2, 3, 4, 5 の平均を"
       "求めてください",
       "3", [STAT]),
    ex("2, 4, 6, 8, 10 の中央値を"
       "求めてください",
       "6", [STAT]),
    ex("1 km は何 m ですか？",
       "1000", [CONV]),
    ex("2 kg は何 g ですか？",
       "2000", [CONV]),
    ex("5 km を3回走りました。"
       "合計は何 m ですか？",
       "15000", [CONV, CALC]),
    ex("日本の首都はどこですか？",
       "東京", []),
]

# 評価用（最適化器には見せない）
testset = [
    ex("128 * 1.08 を計算してください",
       "138.24", [CALC]),
    ex("(45 + 55) * 3 を"
       "計算してください",
       "300", [CALC]),
    ex("1, 1, 2, 2, 100 の平均を"
       "求めてください",
       "21.2", [STAT]),
    ex("3, 7, 1, 9, 5 の最大値を"
       "求めてください",
       "9", [STAT]),
    ex("5000 m は何 km ですか？",
       "5", [CONV]),
    ex("500 g は何 kg ですか？",
       "0.5", [CONV]),
    ex("2 kg と 500 g の合計は"
       "何 g ですか？",
       "2500", [CONV, CALC]),
    ex("虹は一般に何色と"
       "言われますか？",
       "7", []),
]

print("学習用", len(trainset), "件")
print("評価用", len(testset), "件")
```

### ここでやった大事なこと

`expected_tools` という
**正解ラベルを足した**。

「この質問には
このツールを使うべき」
という情報。

これが無いと
**ツールの使い方を評価できない。**

テーマが
「ツール利用最適化」である以上、
これは必須。

**発表でここを説明すること。**

---

## STEP 2 — metric を直す

### 今の metric の問題

```python
def metric(example, pred, trace=None):
    expected = str(example.answer).lower().strip()
    actual = str(pred.answer).lower().strip()
    return expected in actual
```

これは**部分文字列一致**。
実際に試すとこうなる。

| 期待 | 出力 | 判定 |
|---|---|---|
| 3 | 平均は 3 です | ○ 正しい |
| 3 | 数値は 1,2,3,4,5 です | ○ **誤り** |
| 1000 | 1 km は 1,000 m | × **誤り** |

**2行目が致命的。**

「1, 2, 3, 4, 5 の平均」の答え `3` は
**質問文の中に入っている**。

だから
ツールを一度も呼ばずに
質問をオウム返ししただけで
正解になってしまう。

ツールを使わせたいのに、
ツールを使わなくても満点。
これでは最適化が意味を失う。

3行目は
`1,000` の桁区切りで落ちている。

### 直し方の方針

1. カンマと全角を消す
2. **出力の一番最後の数値**を
   答えとみなす
3. ツールの使い方も一緒に測る

### コピペするコード

```python
import re


def clean(text):
    """カンマと全角数字を整える"""
    text = str(text).strip().replace(",", "")
    return text.translate(str.maketrans(
        "０１２３４５６７８９．",
        "0123456789."))


def is_correct(example, pred):
    """答えが合っているか"""
    expected = clean(example.answer)
    actual = clean(getattr(pred, "answer", ""))

    if actual == "":
        return False

    try:
        expected_num = float(expected)
    except ValueError:
        # 数値でない答え（例：東京）
        return expected in actual

    # 出力の最後の数値を答えとみなす
    numbers = re.findall(r"-?\d+\.?\d*", actual)
    if len(numbers) == 0:
        return False
    return abs(float(numbers[-1])
               - expected_num) < 0.001


def used_tools(pred):
    """実際に呼ばれたツール名を取り出す"""
    trajectory = getattr(pred, "trajectory", {})
    tools = []
    i = 0
    while f"tool_name_{i}" in trajectory:
        name = trajectory[f"tool_name_{i}"]
        if name != "finish":   # 組み込みは除く
            tools.append(name)
        i += 1
    return tools


def tool_f1(example, pred):
    """ツール選択の正しさ（0〜1）"""
    expected = set(example.expected_tools)
    actual = set(used_tools(pred))

    # 呼ばなくてよい問題で呼ばなかった
    if len(expected) == 0 and len(actual) == 0:
        return 1.0
    # 呼ばなくてよいのに呼んだ／逆
    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    hit = len(expected & actual)
    if hit == 0:
        return 0.0
    precision = hit / len(actual)
    recall = hit / len(expected)
    return (2 * precision * recall
            / (precision + recall))


def metric(example, pred, trace=None):
    """最適化器に渡す指標"""
    correct = is_correct(example, pred)
    f1 = tool_f1(example, pred)

    if trace is not None:
        # 最適化器がお手本を選んでいる最中は
        # 「完璧なものだけ」採用する
        return correct and f1 == 1.0

    return 0.7 * float(correct) + 0.3 * f1
```

### metric が壊れていないか確認する

**これは必ずやること。**
metric が壊れていると
最適化が効かないのに
原因が分からなくなる。

LMを呼ばないので無料。

```python
# にせの予測を作るための道具
def fake(answer, tools):
    p = dspy.Prediction()
    p.answer = answer
    traj = {}
    for i, t in enumerate(tools):
        traj[f"tool_name_{i}"] = t
    traj[f"tool_name_{len(tools)}"] = "finish"
    p.trajectory = traj
    return p


print(is_correct(trainset[2],
      fake("平均は 3 です", [STAT])))
# True であってほしい

print(is_correct(trainset[2],
      fake("数値は 1, 2, 3, 4, 5 です", [STAT])))
# False であってほしい（オウム返しを弾く）

print(is_correct(trainset[4],
      fake("1 km は 1,000 m です", [CONV])))
# True であってほしい（桁区切りに対応）

print(tool_f1(trainset[7],
      fake("東京です", [])))
# 1.0（呼ばない判断ができた）

print(tool_f1(trainset[7],
      fake("東京です", [CALC])))
# 0.0（呼ぶ必要がないのに呼んだ）
```

**この5行の出力を
発表のスライドに貼るとよい。**
「指標をテストした」という
一言があるだけで説得力が変わる。

---

## STEP 3 — 評価の関数を作る

```python
def evaluate(agent, dataset, name):
    """データセット全部を解かせて集計する"""
    rows = []

    for i, example in enumerate(dataset, 1):
        try:
            pred = agent(question=example.question)
        except Exception as e:
            print(f"  [{i}] エラー: {e}")
            pred = dspy.Prediction()
            pred.answer = ""
            pred.trajectory = {}

        correct = is_correct(example, pred)
        f1 = tool_f1(example, pred)
        tools = used_tools(pred)

        rows.append({
            "question": example.question,
            "expected": example.answer,
            "predicted": getattr(pred, "answer", ""),
            "correct": correct,
            "expected_tools": example.expected_tools,
            "used_tools": tools,
            "tool_f1": f1,
        })

        mark = "○" if correct else "×"
        print(f"  [{i:2d}] {mark} {tools}")

    n = len(rows)
    summary = {
        "name": name,
        "accuracy": sum(r["correct"] for r in rows) / n,
        "tool_f1": sum(r["tool_f1"] for r in rows) / n,
        "avg_calls": sum(len(r["used_tools"])
                         for r in rows) / n,
        "rows": rows,
    }

    print(f"\n=== {name} ===")
    print(f"  正解率        : "
          f"{summary['accuracy']:.3f}")
    print(f"  ツール選択F1  : "
          f"{summary['tool_f1']:.3f}")
    print(f"  平均呼び出し数: "
          f"{summary['avg_calls']:.2f}")
    return summary
```

---

## STEP 4 — 最適化「前」を測る

**これが一番大事。**

最適化前の数字が無いと
「良くなった」が言えない。
発表が成立しなくなる。

```python
before = evaluate(agent, testset, "最適化前")
```

数字をメモしておくこと。
（変数 `before` に入っているが、
念のため紙にも書く）

---

## STEP 5 — 失敗を目で見る

```python
for r in before["rows"]:
    if not r["correct"] or r["tool_f1"] < 1.0:
        print("Q :", r["question"])
        print("期待:", r["expected"],
              r["expected_tools"])
        print("出力:", r["predicted"],
              r["used_tools"])
        print("-" * 30)
```

失敗を4つに分類する。

- **ツールを間違えた**
  → 指示文か説明文の問題
- **引数を間違えた**
  → 許容値が分からないのが原因
- **ツールを呼ばなかった**
  → 指示文の問題
- **ツールは合うが答えが変**
  → 最終回答の合成の問題

**この分類を発表に入れること。**
「何が壊れていたか」を語れる人だけが
「最適化が何を直したか」を語れる。

引数エラーが多い場合は、
実際のプロンプトを見る。

```python
dspy.inspect_history(n=1)
```

---

## STEP 6 — 最適化する

### その前に確認

最適化は
**MCPサーバに何十〜何百回も
HTTPを投げる。**

**メンターに一言確認してから
実行すること。**

### 実行

```python
optimizer = dspy.BootstrapFewShot(
    metric=metric,
    max_bootstrapped_demos=2,
    max_labeled_demos=2,
    max_rounds=1,
)

optimized = optimizer.compile(
    agent,
    trainset=trainset,
)

print("最適化 完了")
```

`BootstrapFewShot` は
最適化器の中で**一番安くて速い**。

やっていることは単純で、

1. 学習データを実際に解かせる
2. metric が満点だったものを
   「お手本」として選ぶ
3. そのお手本をプロンプトに埋め込む

つまり
**うまくいった実例を
プロンプトに入れる**だけ。

初心者でも説明しやすいので、
発表にはこれで十分。

### 保存

```python
optimized.save("optimized.json")
```

---

## STEP 7 — 最適化「後」を測る

**同じ testset で測ること。**
違うデータで測ると比較にならない。

```python
after = evaluate(optimized, testset,
                 "最適化後")
```

---

## STEP 8 — 指示文の変化を見る

**ここが発表の山場。**

DSPy が実際に
プロンプトの何を変えたのかを見る。

```python
print("### 最適化前 ###")
for name, p in agent.named_predictors():
    print(f"\n--- {name} ---")
    print("お手本の数:", len(p.demos))

print("\n### 最適化後 ###")
for name, p in optimized.named_predictors():
    print(f"\n--- {name} ---")
    print("お手本の数:", len(p.demos))
```

`react` と `extract.predict` という
2つの部品にお手本が付いたはず。

- `react` … 次にどのツールを呼ぶか決める
- `extract.predict` … 最終回答を作る

実際に入ったお手本の中身も見る。

```python
for name, p in optimized.named_predictors():
    if len(p.demos) > 0:
        print(f"--- {name} のお手本1つ目 ---")
        print(p.demos[0])
        break
```

### 念のため確認すること

ツール一覧が壊れていないか。

```python
print(optimized.react.signature.instructions)
```

DSPy はツールの説明文を
この指示文の中に埋め込んでいる。

3つのツールが
ちゃんと残っているか目で見る。

**この確認をしたと発表で言うと、
仕組みを理解している証拠になる。**

---

## STEP 9 — 発表用の表を出す

```python
print("| 指標 | 最適化前 | 最適化後 |")
print("|---|---|---|")
print(f"| 正解率 | {before['accuracy']:.3f} "
      f"| {after['accuracy']:.3f} |")
print(f"| ツール選択F1 | {before['tool_f1']:.3f} "
      f"| {after['tool_f1']:.3f} |")
print(f"| 平均呼び出し数 | {before['avg_calls']:.2f} "
      f"| {after['avg_calls']:.2f} |")
```

出た表をそのまま
スライドに貼る。

### 1問ずつの比較も出す

```python
for b, a in zip(before["rows"], after["rows"]):
    if b["used_tools"] != a["used_tools"]:
        print("Q :", b["question"])
        print("前:", b["used_tools"])
        print("後:", a["used_tools"])
        print("-" * 30)
```

**ツールの選び方が変わった問題**を
1〜2件スライドに載せると、
「何が起きたか」が一目で伝わる。

---

## 数字が改善しなかったら

**それでも発表は成立する。**
むしろ正直に報告するほうが良い。

言い方の例:

> 今回のタスクは3つのツールの
> 役割が明確に分かれており、
> 最適化前の時点でツール選択が
> ほぼ正しかった。
> そのため改善の余地が小さかった。
>
> これは逆に、
> ツール設計が適切であれば
> 自動最適化に頼らずとも
> 十分な性能が出ることを示している。
>
> 最適化が効くのは、
> ツールの役割が曖昧な場合や
> ツール数が多い場合だと考えられる。

**これは立派な考察。**
実運用への示唆になっている。

数字が下がった場合も同じで、
「学習データ8件では
お手本の質が不十分だった」
という仮説を述べればよい。

---

## 10分発表の構成

スライド8枚。時間配分つき。

### 1. タイトル（0:30）

- テーマ名
- DSPy を題材にした調査・実装・評価

### 2. 背景と課題（1:30）

- AIエージェントの品質は
  ツールを**いつ・どう使うか**で決まる
- その制御はプロンプトに書かれている
- 現状、その改善は**手作業**
- 属人的・再現性がない・
  モデル更新で壊れる

### 3. DSPy とは（1:30）

一言で:

> プロンプトを手で書く代わりに、
> 入出力の契約をコードで宣言し、
> プロンプトは最適化アルゴリズムに
> 書かせるフレームワーク

3つの部品:

- **Signature** … 入出力の契約
- **Module** … 呼び出し戦略（ReAct）
- **Optimizer** … 自動最適化

### 4. 実装したもの（1:30）

- Azure OpenAI + MCP の3ツール
- `dspy.ReAct` でエージェント化
- ツールは
  **docstring付きのPython関数**を
  渡すだけ

図を1枚:

```
質問 → ReAct → ツール選択
     → MCPサーバ → 結果
     → 最終回答
```

### 5. 評価の設計（2:00）★ここが売り

**このスライドに一番時間を使う。**

- 回答の正誤だけでは不十分
- 「答えは合っているが
  余計なツールを呼んだ」を
  検出できない
- そこで正解ラベルに
  `expected_tools` を追加
- 指標を2階層にした
  - タスク階層：正解率
  - 呼び出し階層：ツール選択F1
- 「ツール不要」の問題も入れた
  （呼ばない判断も性能のうち）
- 指標そのものをテストした

STEP 2 の確認出力を貼る。

### 6. 結果（2:00）

- STEP 9 の表
- ツール選択が変わった問題を1〜2件
- 最適化でプロンプトに
  お手本が入ったこと

### 7. 考察と適用可能性（1:00）

**効きそうな場面**
- ツール数が多い
- ツールの役割が曖昧
- 評価データが作れる業務

**課題**
- 評価データ作成のコスト
- 最適化中のAPI課金・
  ツールサーバへの負荷
- 何を測るかの設計が
  結果を決めてしまう

### 8. まとめ（0:30）

> DSPy を使うと、人間の仕事は
> 「プロンプトを書くこと」から
> 「何を良しとするかを定義すること」
> に移る。
>
> 本実習で最も時間を使ったのは
> 評価指標の設計だった。

---

## 発表で必ず言うと良い一言

質疑で効く。

**① 学習と評価を分けた**
> 最適化器に見せたデータで
> 評価すると数字が嘘になるので、
> 8件と8件に分けました。

**② 指標をテストした**
> 指標が壊れていると
> 最適化が効かない原因が
> 分からなくなるので、
> LMを呼ばずに指標だけを
> 先にテストしました。

**③ ツールを呼ばない判断も測った**
> 何でもツールを呼ぶと
> 実運用ではコストと
> 待ち時間が増えるので、
> 呼ばない判断も
> 評価対象に入れました。

---

## 困ったときの対処

**`NameError`**
→ インデントが下がっている（STEP 0）

**`AttributeError: answer`**
→ エージェントが途中で失敗している。
　`getattr` を使っているか確認

**引数のエラーが出る**
→ `analyze_numbers` の
　`operations` に何を書けばいいか
　メンターに聞く

**metric が全部 0 か全部 1**
→ STEP 2 の確認コードを実行して
　どこが壊れているか特定する

**時間がかかりすぎる**
→ `max_iters=5` になっているか、
　`trainset` が8件のままか確認

**答えが英語になる**
→ `ToolQA` の docstring に
　"Answer in Japanese." を足す

**15分詰まったらメンターに聞く。**
実習は5日しかない。
