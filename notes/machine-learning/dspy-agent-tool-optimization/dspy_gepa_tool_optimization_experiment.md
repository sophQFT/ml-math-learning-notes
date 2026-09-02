# DSPy GEPAツール利用最適化：複数ツール問題追加版

> 現在のNotebookを大きく変更せず、`変換→計算`と`変換→統計`の問題を追加して、GEPAの改善を確認するための手順です。

## 1. 今回変更するもの

前回は、次のような1ツール問題だけで評価しました。

```text
25 * 16
→ calculate_expression

1, 2, 3, 4, 5の平均
→ analyze_numbers

10 kmは何mか
→ convert_units
```

これらは簡単なので、Baselineが最初から約99点になりました。そこで、次の2種類だけを追加します。

```text
単位変換
→ 数値計算

単位変換
→ 統計分析
```

`ToolQA`、`MCPToolAgent`、`ToolUseJudge`、`run_judge()`、`tool_use_metric()`は変更しません。

---

## 2. なぜ複数ツール問題を追加するのか

例えば、次の質問を考えます。

```text
2 kmをmに直し、350 mを足してください。
```

理想的な処理は次のとおりです。

```text
convert_units
2 km → 2000 m
        ↓
calculate_expression
2000 + 350 → 2350
        ↓
最終回答
2350 m
```

次の質問では、変換後に統計ツールを使います。

```text
1.2 km、800 m、1500 mの平均をmで求めてください。
```

理想的な処理です。

```text
convert_units
1.2 km → 1200 m
        ↓
analyze_numbers
[1200, 800, 1500]の平均
        ↓
最終回答
約1166.67 m
```

1つ目のツール結果を2つ目のツールへ正しく渡す必要があるため、1ツール問題よりもReAct Agentの能力差が出やすくなります。

---

## 3. 変更前に確認するもの

現在のNotebookで、次がすでに定義されていれば、そのまま使えます。

```python
agent
make_example
ToolUseJudge
judge_lm
judge
run_judge
tool_use_metric
score_only_metric
reflection_lm
```

これらのコードは書き換えません。

---

## 4. `trainset`のセルだけ置き換える

現在の`trainset`を、次の内容へ置き換えて実行します。

### セル1：新しい`trainset`

```python
trainset = [
    # -------------------------
    # 1ツール：数値計算
    # -------------------------
    make_example(
        "25 * 16を計算してください"
    ),
    make_example(
        "2の10乗を求めてください"
    ),

    # -------------------------
    # 1ツール：統計
    # -------------------------
    make_example(
        "1, 2, 3, 4, 5の平均を求めてください"
    ),
    make_example(
        "2, 4, 6, 8, 10の中央値を求めてください"
    ),

    # -------------------------
    # 1ツール：単位変換
    # -------------------------
    make_example(
        "10 kmは何mですか？"
    ),
    make_example(
        "2 kgは何gですか？"
    ),

    # -------------------------
    # 2ツール：変換→計算
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
    # 2ツール：変換→統計
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
```

全部で14問です。

| 種類 | 問題数 |
|---|---:|
| 1ツール問題 | 6問 |
| 変換→計算 | 4問 |
| 変換→統計 | 4問 |

---

## 5. まず複数ツール問題を1問だけ確認する

GEPAを実行する前に、`trainset[6]`を試します。

### セル2：変換→計算の確認

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

理想的な`tool_calls`は、おおむね次の順番です。

```text
convert_units
calculate_expression
```

次に、変換→統計も確認します。

### セル3：変換→統計の確認

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

理想的な`tool_calls`は、おおむね次の順番です。

```text
convert_units
analyze_numbers
```

Agentが別の方法を選ぶ場合もあります。この時点では修正せず、その実行履歴をJudgeとGEPAへ渡します。

---

## 6. 新しいtrainsetでBaselineを測る

既存の`score_only_metric`をそのまま使用します。

### セル4：Baseline評価

```python
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

前回のBaselineは約98.67点でした。今回は複数ツール問題が入るため、それより低くなっても異常ではありません。むしろ、GEPAが改善できる余地ができたことを意味します。

---

## 7. GEPAを設定する

766回にならないよう、回数を明示します。

### セル5：optimizer

```python
optimizer = dspy.GEPA(
    metric=tool_use_metric,
    max_metric_calls=80,
    reflection_lm=reflection_lm,
    num_threads=4,
)
```

`auto="light"`は削除してください。`auto`と`max_metric_calls`は同時に指定できません。

```python
# このように両方は書かない。
# auto="light",
# max_metric_calls=80,
```

`80`はmetricを呼び出せる最大回数です。LMの総呼び出し回数そのものではありませんが、前回の約766回より大幅に少なくなります。

レート制限エラーが出る場合だけ、`num_threads=1`へ変更します。

---

## 8. 新しいtrainsetで最適化する

### セル6：compile

```python
optimized_agent = optimizer.compile(
    student=agent,
    trainset=trainset,
)
```

GEPAは複数ツール問題の実行履歴とJudgeのfeedbackを読み、ReActのinstructionを改善します。

```text
変換結果を得る
        ↓
次の計算・統計ツールへ結果を渡す
        ↓
最終回答を作る
```

という手順を安定して実行できる指示が作られることを期待します。

---

## 9. 最適化後を同じ条件で評価する

### セル7：比較

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

例えば次のようになれば、改善が確認できます。

```text
Baseline：82.5
GEPA後  ：90.0
改善幅  ：7.5
```

実際の数値はLMの出力によって変わるため、必ず改善するとは限りません。

---

## 10. 複数ツール問題だけを確認する

全体スコアだけでなく、追加した8問のツール履歴を確認します。

### セル8：最適化後のツール利用

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

    print("---")
```

見るべき点は次の2つです。

- 変換→計算問題で、`convert_units`の後に`calculate_expression`を使ったか。
- 変換→統計問題で、`convert_units`の後に`analyze_numbers`を使ったか。

---

## 11. 結果が改善しなかった場合

まず、BaselineとGEPA後の`tool_calls`を見比べます。

Baselineも最初から複数ツールを正しく使えている場合は、改善余地が少ないため、スコア差が小さくても不自然ではありません。

GEPA後も複数ツール利用に失敗している場合は、すぐにコードを増やさず、次だけ確認します。

1. Judgeの`feedback`が日本語で具体的に出ているか。
2. `convert_units`の結果が`tool_results`へ入っているか。
3. 2番目のツールへ渡した値が正しいか。
4. 80回すべて正常に完了したか。

動作が正常で探索量だけが不足していると判断できた場合に限り、次回は`max_metric_calls=120`を試します。

高得点が出るまで繰り返すのではなく、最初の結果と変更条件を記録してください。

---

## 12. 発表での説明例

> 単一ツール問題ではBaselineが約99点となり、最適化効果を確認できなかった。そこで、単位変換の結果を数値計算または統計分析へ渡す複数ツール問題を追加した。これにより、ツールの選択だけでなく、実行順序と結果の受け渡しも評価対象とした。

結果は次の形式で示します。

```text
Baseline：実測値
GEPA後  ：実測値
改善幅  ：実測値
```

代表例として、次の2種類のtrajectoryを1件ずつ示すと分かりやすくなります。

```text
convert_units → calculate_expression
convert_units → analyze_numbers
```

---

## 13. 実行順

- [ ] 新しい`trainset`のセルを実行する。
- [ ] 変換→計算を1問だけ確認する。
- [ ] 変換→統計を1問だけ確認する。
- [ ] 新しいtrainsetでBaselineを測る。
- [ ] `max_metric_calls=80`でoptimizerを作る。
- [ ] GEPAの`compile()`を実行する。
- [ ] 同じevaluatorでGEPA後を評価する。
- [ ] 追加した8問の`tool_calls`を確認する。

---

## 14. GitHubへ載せないもの

- Azure OpenAI APIキー
- Azure endpoint
- MCP APIキー
- MCPサーバーのURL、IPアドレス
- 公開許可を得ていない実行ログ

今回変更した質問データと評価結果を公開してよいかも、メンターへ確認してください。
