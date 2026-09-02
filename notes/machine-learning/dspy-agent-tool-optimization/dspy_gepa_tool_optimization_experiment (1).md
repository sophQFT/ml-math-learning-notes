# DSPy GEPAツール利用最適化：複数ツール問題の追加

> 既存のNotebookの続きに、上から順番にコピー＆ペーストして実行するための手順です。元の6問の`trainset`は書き換えません。

## 1. 今回行うこと

これまでの6問は、1回のツール利用で回答できる問題でした。

| 問題の種類 | 使用するツール |
|---|---|
| 数値計算 | `calculate_expression` |
| 統計 | `analyze_numbers` |
| 単位変換 | `convert_units` |

今回は次の複数ツール問題を追加します。

| 追加する種類 | 理想的なツールの順番 |
|---|---|
| 変換→計算 | `convert_units` → `calculate_expression` |
| 変換→統計 | `convert_units` → `analyze_numbers` |

複数ツール問題では、1つ目のツールの結果を2つ目のツールへ渡す必要があります。そのため、簡単な1ツール問題よりもGEPAによる改善を確認しやすくなります。

---

## 2. 実行前の確認

この手順は、元のNotebookで次の変数や関数を定義した後に実行します。

```python
agent
make_example
trainset
tool_use_metric
score_only_metric
reflection_lm
```

元の`trainset`には6問が入っています。

```python
print(len(trainset))
```

次のように表示されれば問題ありません。

```text
6
```

---

## 3. 複数ツール問題を追加する

### セル1：追加問題を作る

次のコードを、Notebookの新しいセルへそのままコピーして実行します。

```python
multi_tool_examples = [
    # -------------------------
    # 変換→計算
    # -------------------------
    make_example(
        "2 kmをmに直し、350 mを足してください"
    ),
    make_example(
        "1.5 kgをgに直し、そこから250 gを引いてください"
    ),
    make_example(
        "3.4 kmをmに直し、600 mを足してください"
    ),
    make_example(
        "2.25 kgをgに直し、750 gを足してください"
    ),

    # -------------------------
    # 変換→統計
    # -------------------------
    make_example(
        "1.2 km、800 m、1500 mの平均をmで求めてください"
    ),
    make_example(
        "500 cm、7 m、900 cmの中央値をcmで求めてください"
    ),
    make_example(
        "0.75 kg、1200 g、950 gの平均をgで求めてください"
    ),
    make_example(
        "1.8 km、2400 m、3 kmの中央値をmで求めてください"
    ),
]

# 元のtrainsetは変更せず、新しいリストを作る。
full_trainset = trainset + multi_tool_examples

print("元の問題数:", len(trainset))
print("追加問題数:", len(multi_tool_examples))
print("追加後の問題数:", len(full_trainset))
```

次のように表示されれば成功です。

```text
元の問題数: 6
追加問題数: 8
追加後の問題数: 14
```

次の行では、元の6問と追加した8問を連結しています。

```python
full_trainset = trainset + multi_tool_examples
```

`trainset`自体は変更されません。したがって、セル1を再実行しても問題が重複して増えることはありません。

`trainset.extend(multi_tool_examples)`でも追加できますが、同じセルを再実行するたびに8問ずつ重複します。Notebookでは、今回の`full_trainset = trainset + multi_tool_examples`の書き方が安全です。

---

## 4. 変換→計算を1問だけ確認する

### セル2：7問目を実行する

```python
example = full_trainset[6]

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

Pythonのリストは0番から数えます。そのため、`full_trainset[6]`は7問目、つまり最初に追加した問題です。

```text
2 kmをmに直し、350 mを足してください
```

理想的には、次の順番でツールが呼ばれます。

```text
convert_units
calculate_expression
```

処理内容は次のとおりです。

```text
2 km → 2000 m
2000 + 350 → 2350
最終回答 → 2350 m
```

この時点でツールの順番が違っても、すぐにコードを変更する必要はありません。その失敗例とJudgeのfeedbackをGEPAが最適化に使用します。

---

## 5. 変換→統計を1問だけ確認する

### セル3：11問目を実行する

```python
example = full_trainset[10]

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

`full_trainset[10]`は11問目で、最初の変換→統計問題です。

```text
1.2 km、800 m、1500 mの平均をmで求めてください
```

理想的には、次の順番でツールが呼ばれます。

```text
convert_units
analyze_numbers
```

処理内容は次のとおりです。

```text
1.2 km → 1200 m
[1200, 800, 1500]の平均 → 約1166.67
最終回答 → 約1166.67 m
```

---

## 6. Baselineを測る

Baselineとは、GEPAで最適化する前の`agent`の点数です。GEPA後の点数と比較するため、先に記録します。

### セル4：最適化前を評価する

```python
evaluator = dspy.Evaluate(
    devset=full_trainset,
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

重要なのは、`devset=trainset`ではなく、次のように`full_trainset`を指定することです。

```python
devset=full_trainset
```

前回よりBaselineが下がっても異常ではありません。複数ツール問題が追加され、課題が難しくなったためです。

---

## 7. GEPAを設定する

前回のようにmetric callが約766回にならないよう、最大回数を指定します。

### セル5：optimizerを作る

```python
optimizer = dspy.GEPA(
    metric=tool_use_metric,
    max_metric_calls=80,
    reflection_lm=reflection_lm,
    num_threads=4,
)
```

`max_metric_calls=80`は、GEPAがmetricを呼び出せる最大回数です。

`auto="light"`は書かないでください。`auto`と`max_metric_calls`は同時に指定できません。

```python
# この組み合わせにはしない。
# auto="light",
# max_metric_calls=80,
```

レート制限エラーが出た場合だけ、次のように変更します。

```python
num_threads=1
```

---

## 8. GEPAで最適化する

### セル6：`compile()`を実行する

```python
optimized_agent = optimizer.compile(
    student=agent,
    trainset=full_trainset,
)
```

ここでも`trainset=trainset`ではなく、追加問題を含む`full_trainset`を指定します。

GEPAは、おおまかに次の処理を繰り返します。

1. Agentを問題で実行する。
2. Judgeがツール利用と回答を採点する。
3. Judgeが日本語のfeedbackを返す。
4. Reflection LMがfeedbackを読む。
5. Agentの指示文の改善案を作る。
6. より高得点の候補を`optimized_agent`として返す。

モデル自体の重みを学習する処理ではありません。主にReAct Agentへ与える指示文を改善しています。

---

## 9. GEPA後を評価する

### セル7：同じ14問で比較する

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

表示例です。

```text
Baseline: 82.5
GEPA後: 90.0
改善幅: 7.5
```

BaselineとGEPA後は、どちらも同じ`evaluator`、同じ14問、同じmetricで評価します。条件をそろえることで、最適化前後を比較できます。

LLMとJudgeの出力には揺らぎがあるため、必ず点数が上がるとは限りません。点数だけでなく、次の手順でツールの順番も確認します。

---

## 10. 追加した8問のツール利用を確認する

### セル8：最適化後の`tool_calls`を表示する

```python
for example in full_trainset[6:]:
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

確認する点は次の2つだけです。

- 変換→計算問題で、`convert_units`の後に`calculate_expression`を使用したか。
- 変換→統計問題で、`convert_units`の後に`analyze_numbers`を使用したか。

---

## 11. 途中から実行してエラーになった場合

### `NameError: name 'trainset' is not defined`

元のNotebookにある`trainset`を定義するセルを先に実行してください。

### `NameError: name 'make_example' is not defined`

元のNotebookにある`make_example()`を定義するセルを先に実行してください。

### `NameError: name 'score_only_metric' is not defined`

元のNotebookにある`score_only_metric()`を定義するセルを先に実行してください。

### `NameError: name 'reflection_lm' is not defined`

元のNotebookにある`reflection_lm`を作るセルを先に実行してください。

### 問題数が14問にならない

次を実行してください。

```python
print(len(trainset))
print(len(multi_tool_examples))
print(len(full_trainset))
```

正しい表示は次のとおりです。

```text
6
8
14
```

元の`trainset`をすでに書き換えている場合は、元の6問を定義するセルをもう一度実行してから、セル1を実行してください。

---

## 12. 結果が改善しなかった場合

Baselineがすでに高く、複数ツールも正しい順番で使えている場合は、改善幅が小さくても不自然ではありません。

GEPA後も失敗している場合は、まず次を確認します。

1. Judgeの`feedback`が日本語で具体的に出ているか。
2. `convert_units`の結果が`tool_results`へ入っているか。
3. 2番目のツールへ渡した数値が正しいか。
4. GEPAがエラーなく完了したか。

処理は正常でも探索回数が足りないと判断した場合は、次の実験で`max_metric_calls=120`を試します。一度目の結果も消さずに記録してください。

---

## 13. 実行順チェックリスト

- [ ] 元のNotebookで`trainset`まで定義する。
- [ ] `score_only_metric`と`reflection_lm`まで定義する。
- [ ] セル1で`full_trainset`を作る。
- [ ] 6問、8問、14問と表示されることを確認する。
- [ ] セル2で変換→計算を1問確認する。
- [ ] セル3で変換→統計を1問確認する。
- [ ] セル4でBaselineを測る。
- [ ] セル5で`max_metric_calls=80`のGEPAを作る。
- [ ] セル6で最適化する。
- [ ] セル7でBaselineとGEPA後を比較する。
- [ ] セル8で追加した8問のツール利用を確認する。

---

## 14. 発表での説明例

> 単一ツール問題だけではBaselineが約99点となり、最適化による改善を確認しにくかった。そこで、単位変換の結果を数値計算または統計分析へ渡す複数ツール問題を追加した。これにより、ツールの選択だけでなく、実行順序とツール間の結果の受け渡しも評価対象とした。

結果は次の形式で記録します。

```text
Baseline: 実測値
GEPA後: 実測値
改善幅: 実測値
```

代表的な実行履歴として、次の2種類を1件ずつ示すと分かりやすくなります。

```text
convert_units → calculate_expression
convert_units → analyze_numbers
```

---

## 15. GitHubへ載せないもの

- Azure OpenAI APIキー
- Azure endpoint
- MCP APIキー
- MCPサーバーのURLやIPアドレス
- 公開許可を得ていない実行ログ

質問データと評価結果をGitHubへ公開してよいかも、メンターへ確認してください。
