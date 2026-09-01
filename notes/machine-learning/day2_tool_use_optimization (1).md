# Day2 — ツール利用最適化の調査と実装

> **今日のゴールはただ 1 つ。「最適化していない ReAct エージェントが動き、その性能が数字で記録されている」状態を作ること。**
> 最適化そのものは Day3。Day2 に最適化まで手を出すと、ほぼ確実に破綻する。

---

## 0. タイムテーブル（想定）

| 時刻 | やること | 完了条件 |
|---|---|---|
| 8:30–9:00 | メンターに確認（§1） | 使う LM が決まっている |
| 9:00–10:00 | 環境構築（§2） | `python agent.py` で回答が返る |
| 10:00–12:00 | ツール設計（§4） | 自分のタスクのツールが 3〜5 個定義できた |
| 13:00–15:00 | 評価データ作成（§5） | `dspy.Example` が 20 件ある |
| 15:00–16:30 | メトリクス設計（§6） | `metric_tool_aware` が自分のタスク用に直せた |
| 16:30–17:15 | ベースライン測定（§7） | `results/baseline.json` が出来ている |

---

## 1. 朝イチでメンターに確認すること

初心者にとって最大のリスクは技術力ではなく「詰まったまま半日溶かすこと」。着席したらこれを聞く。

- [ ] **使える LM は何か**（OpenAI / Azure OpenAI / 社内エンドポイント / ローカル LLM）。DSPy は LiteLLM 経由なので `"<provider>/<model>"` 形式で指定する
- [ ] **API キーの環境変数名**と、**課金上限・レート制限**。最適化は LM を大量に呼ぶのでここは必須
- [ ] **外部ネットワークに出られるか**（公式チュートリアルの Wikipedia 検索がそのまま動くか）
- [ ] **題材のタスクとツールは自分で決めてよいか**、それとも社内の既存アプリのログ・ツール定義を使うのか
- [ ] **成果物の形式**（スライド枚数、コードの提出有無、社外秘の扱い）

> このリポジトリのコードは **外部ネットワーク不要**（すべてダミーの辞書 DB）にしてある。社内の題材が決まらなくても、これで丸 1 日ぶんの実験は回せる。

---

## 2. 環境構築（30 分で終わらせる）

```bash
# 1. 仮想環境
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. インストール
#    注意: PyPI の "dspy-ai" は互換用エイリアス。必ず "dspy" を入れる
pip install -U dspy
pip install mlflow                 # 強く推奨（後述）

# 3. バージョン確認（3.3 以上ならこのリポジトリのコードがそのまま動く）
python -c "import dspy; print(dspy.__version__)"
```

Python は 3.9 以上が必要。

### LM の設定

`code/config.py` の `MODEL` だけ書き換えれば全スクリプトに反映される。環境変数でも上書きできる。

```bash
export OPENAI_API_KEY="sk-..."
export DSPY_MODEL="openai/gpt-4o-mini"
# 社内の OpenAI 互換エンドポイントなら
# export DSPY_API_BASE="https://internal.example.com/v1"
```

```python
import dspy
lm = dspy.LM("openai/gpt-4o-mini", temperature=0.0, max_tokens=4000)
dspy.configure(lm=lm)
```

**`temperature=0.0` にすること。** 評価の再現性が確保できないと Day4 の比較が意味を失う。

### 疎通確認

```bash
cd code
python -c "
import dspy
from config import setup_lm
setup_lm()
print(dspy.settings.lm('こんにちは、と返してください'))
"
```

ここが通らないうちは先へ進まない。**通らなければ即メンターに聞く。**

### MLflow トレースを最初に入れる

これは本当に効く。MLflow は DSPy とネイティブに統合された LLMOps ツールで、プロンプトと最適化の進行をトレースとして可視化できる。

```bash
# 別ターミナルで
mlflow ui --port 5000
```

```python
from config import enable_mlflow
enable_mlflow()          # 以降のすべての LM 呼び出しが記録される
```

手軽な代替として、直前に実際に送られたプロンプトを見るこれも常用する。

```python
dspy.inspect_history(n=1)
```

**「思った通りのプロンプトになっていない」がバグの 8 割。** 必ず目で見る癖をつける。

---

## 3. 最小構成を写経する（20 分）

まずこれが動くことを確認してから自分のタスクに移る。

```python
import dspy

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

def add(a: float, b: float) -> float:
    """2 つの数を足す。"""
    return a + b

def multiply(a: float, b: float) -> float:
    """2 つの数を掛ける。"""
    return a * b

agent = dspy.ReAct("question -> answer: float", tools=[add, multiply])
pred = agent(question="3と4を足して、その結果に5を掛けるといくつ？")

print(pred.answer)
print(pred.trajectory)      # ← ここを必ず見る
```

`trajectory` に `thought_0` / `tool_name_0` / `tool_args_0` / `observation_0` … が並んでいれば理解できている。

---

## 4. ツールの設計（本日の山場 その1）

### 4.1 ツールは docstring 付きの Python 関数

DSPy でのツール定義は、docstring と型ヒントを付けた Python 関数を用意し、タスクを定義する signature と一緒に `dspy.ReAct` に渡すだけ。

```python
def get_leave_balance(employee_id: str) -> dict:
    """社員IDから、その社員の有給休暇の残日数を取得する。

    引数は氏名ではなく社員ID (例: "E001")。氏名しか分からない場合は
    先に search_employee で社員IDを引くこと。
    有給の残日数はこのツールで取得する（get_attendance では取れない）。
    """
    ...
```

### 4.2 docstring の書き方 — ここが最適化の起点

**docstring がそのまま LM への説明文になる。** つまり docstring はプロンプトの一部であり、本テーマの調査対象そのもの。書くべき要素は 4 つ。

1. **何をするツールか**（1 行）
2. **引数の形式**（「氏名ではなく社員ID」のような具体例つき）
3. **前提条件**（「先に X を呼ぶこと」）
4. **使うべきでない場面**（「制度の質問には使わない」）← **これを書く人が少ないが効果が大きい**

### 4.3 意図的に「紛らわしいツール」を混ぜる

最適化の効果を測るには、**最適化前に失敗が起きる余地**が必要。全問正解のタスクでは改善幅がゼロになり、発表するものが無くなる。

`code/tools.py` では次のように設計してある。

| ツール | 紛らわしさの仕込み |
|---|---|
| `search_employee` | 氏名 → 社員ID。これを飛ばすと後続が全滅する |
| `get_leave_balance` | 有給残。`get_attendance` と混同されやすい |
| `get_attendance` | 勤怠実績。有給残は取れない |
| `search_faq` | 制度の質問専用。個人データは無い |
| `calculate` | 四則演算のみ |

**Day4 の実験に効く仕込み:** `ALL_TOOLS`（5 個）と `MINIMAL_TOOLS`（3 個）を用意してある。ツール数を変えて性能を比べる実験ができる。ある研究では、ReAct エージェントのツール数を 3 個から 10 個へ増やすと最適化後の F1 が U 字型のトレンドを示し、ツールを増やすことが必ずしも性能向上につながらなかったと報告されている。これを自分のデータで再現できれば、東芝への提言として非常に強い。

### 4.4 max_iters を必ず絞る

```python
dspy.ReAct(HelpdeskQA, tools=ALL_TOOLS, max_iters=6)
```

既定は 20。暴走したときのコストが跳ねるので、タスクに必要なホップ数 +αに絞る。本タスクは最大 3 ホップなので 6 で十分。

---

## 5. 評価データセットの作成（本日の山場 その2）

### 5.1 `dspy.Example` と `.with_inputs()`

```python
ex = dspy.Example(
    question="山田さんの有給休暇の残日数は何日ですか？",
    answer="12",
    expected_tools=["search_employee", "get_leave_balance"],
    expected_steps=2,
).with_inputs("question")     # ← どれが「入力」かの宣言。忘れると最適化器が動かない
```

**`.with_inputs()` を忘れるのが初心者の第一の関門。** 忘れると「正解も入力として LM に渡ってしまう」か、最適化器がエラーを吐く。

### 5.2 本テーマ特有のポイント: 正解に「呼ぶべきツール」を含める

普通の QA なら `answer` だけで足りる。しかし本テーマは**ツール利用の最適化**なので、それだけでは

> 「答えは合っているが、無駄なツールを 5 回呼んだ」

を検出できない。だから `expected_tools`（呼ぶべきツールの集合）と `expected_steps`（期待ステップ数）を正解ラベルとして持たせる。**この設計判断は発表で必ず説明すること。テーマの理解度がここに出る。**

### 5.3 件数と分割

| 用途 | 件数の目安 | 本リポジトリ |
|---|---|---|
| trainset（最適化器が学習に使う） | 8〜30 | 8 件 |
| valset（最適化器が候補選択に使う） | 5〜20 | 6 件 |
| devset（最終評価。最適化器には見せない） | 5〜20 | 6 件 |

DSPy の最適化器は少数（10〜20 件程度）の例でも動くよう設計されている。**まず 20 件でよい。**

**最適化器に見せたデータでそのまま最終評価すると数字が嘘になる。** 必ず分ける。これも発表で触れるべき点。

### 5.4 問題の作り方

難易度と種類をばらけさせる。`code/dataset.py` の構成:

- **1 ホップ**（社員IDが与えられている / 制度FAQ）— 簡単。ここで落ちるなら根本的な問題
- **2 ホップ**（氏名 → 社員ID → データ）— 標準
- **3 ホップ**（氏名 → 社員ID → 勤怠 → 計算）— 難しい
- **紛らわしい問い**（有給を勤怠ツールで取ろうとしがち）— ツール選択の誤りを誘発
- **エラーケース**（存在しない社員）— 頑健性を見る

---

## 6. メトリクスの設計（本日の山場 その3・最重要）

**DSPy で人間がやる仕事の本体はここ。** プロンプトを書く代わりに「何を良いとするか」を定義する。

### 6.1 メトリクス関数の形

```python
def my_metric(example, pred, trace=None) -> float:
    ...
```

- `example` は正解つきの `dspy.Example`、`pred` はプログラムの出力
- 返り値は `float` か `bool`
- `trace` は DSPy が最適化中に渡してくる。**`trace is not None` なら「最適化器が few-shot デモ候補を選別している最中」** なので、そのときは厳しめの真偽値を返すのが DSPy の慣例

### 6.2 本テーマ用の複合メトリクス

`code/metrics.py` の実装:

```
score = 0.6 × 回答正解  +  0.3 × ツール選択F1  +  0.1 × ステップ効率
```

「答えさえ合えばよい」を避け、**ツールの使い方そのものを最適化対象に引き上げる**のが狙い。重みを変えて感度を見ると良い考察になる（例: ツール項の重みを 0 にしたら最適化後のツール選択はどうなるか）。

### 6.3 trajectory から指標を計算する

押さえるべき挙動:

- ツール名は `trajectory["tool_name_0"]`, `["tool_name_1"]`, … と並ぶ
- **終了用の組み込みツール `finish` は自作ツールの呼び出しに数えない**
- ツールが例外を投げても ReAct は落ちない。`observation_i` が `"Execution error in <ツール名>: ..."` という文字列になる。この文字列を見てエラー率を数える

```python
def extract_tool_calls(trajectory):
    calls, idx = [], 0
    while f"tool_name_{idx}" in trajectory:
        if trajectory[f"tool_name_{idx}"] != "finish":
            calls.append(trajectory[f"tool_name_{idx}"])
        idx += 1
    return calls
```

### 6.4 GEPA 用のフィードバック付きメトリクス

GEPA は**ドメイン固有のテキストフィードバック**を活用して急速に改善できる。`dspy.Prediction(score=..., feedback=...)` を返せばよい。

```python
return dspy.Prediction(
    score=0.4,
    feedback="不要なツールを呼んでいる: ['get_attendance']。有給残日数は get_leave_balance で取得すること。"
)
```

**これは本テーマと相性が抜群。** 「ツール選択のどこが間違っていたか」を日本語で返してやると、GEPA がその指摘を指示文に反映する。Day3〜4 の目玉にできる。

---

## 7. ベースラインの測定（本日の締め）

```bash
cd code
python run_baseline.py
```

出力される指標:

```
--- baseline (n=12) ---
  accuracy          : 0.583
  tool_precision    : 0.694
  tool_recall       : 0.833
  tool_f1           : 0.742
  step_efficiency   : 0.806
  avg_calls         : 2.417
  tool_error_rate   : 0.250
  crash_rate        : 0.000
  avg_latency_sec   : 3.412
```
（数値は例。実際の値は使う LM による）

結果は `results/baseline.json` に保存される。**これが無いと Day3 以降の「改善した」が言えない。**

### 今日の終わりにやること

1. `results/baseline.json` が出来ているか確認
2. **失敗した問題を 3 件、trajectory ごと目で読む。** どこで間違えているかをメモする
   - ツール選択を間違えた? → 指示文 / docstring の問題
   - 引数を間違えた? → 型ヒント / docstring の例示不足
   - 呼ぶべきツールを飛ばした? → 前提条件の記述不足
   - ツールは合っているが最終回答が変? → `extract` 側の問題
3. 明日 Day3 で回す最適化器の順序を決める（推奨: `bootstrap` → `mipro` → 時間が余れば `gepa`）

この「失敗の目視 3 件」が Day5 の発表の質を決める。**最適化器が何を直したかを語れるのは、最適化前に何が壊れていたかを知っている人だけ。**

---

## 8. Day3 の予告（今日は読むだけ）

```bash
python run_optimize.py bootstrap    # まずこれ。安い・速い
python run_optimize.py mipro        # 指示文 + few-shot を同時最適化
python run_optimize.py gepa         # トレースを振り返って改善
python run_compare.py bootstrap mipro gepa
```

**コストの見積もりを必ず先に。** ざっくり

```
LM 呼び出し回数 ≒ 候補数 × 試行数 × データ件数 × ReAct のステップ数
```

MIPROv2 は `auto="light"` から始める（`light` → `medium` → `heavy` の順に高価）。`minibatch=True` なら各試行では `minibatch_size`（既定 35）件のミニバッチだけで評価し、`minibatch_full_eval_steps`（既定 5）ごとに検証セット全体で評価する。データが 8 件しかないなら、ミニバッチサイズは自動的にそれ以下になる。

`run_optimize.py` は最適化の前後で ReAct 内部の 2 つの predictor（`react` と `extract.predict`）の指示文を表示する。**この差分が発表のクライマックス。**

---

## 9. ハマりどころ早見表

| 症状 | 原因 | 対処 |
|---|---|---|
| `ValueError: Example has no input keys` | `.with_inputs()` 忘れ | データセットを見直す |
| 最適化器が全く改善しない | メトリクスが常に 0 か常に 1 | メトリクスを単体でテストする |
| ツールが呼ばれない | docstring が無い / 型ヒントが無い | 両方付ける |
| 引数の型エラーが多発 | docstring に具体例が無い | `(例: "E001")` を書く |
| 無限にツールを呼ぶ | `max_iters` が大きすぎる | 6 前後に絞る |
| 回答が英語になる | 指示文に言語指定が無い | Signature の `desc` に「日本語で」と書く |
| 最適化が終わらない | `auto` が heavy / データが多い | `auto="light"`、trainset を減らす |
| トークン課金が怖い | 見積もりをしていない | 小さい devset で 1 回試して係数を出す |
| `dspy-ai` を入れて動かない | エイリアスパッケージ | `pip install -U dspy` |
| 結果が毎回違う | `temperature` が 0 でない | `temperature=0.0` |

---

## 10. 参考リンク（Day2 で開くもの）

| リンク | いつ見るか |
|---|---|
| <https://dspy.ai/learn/programming/tools/> | ツールを定義するとき |
| <https://dspy.ai/tutorials/customer_service_agent/> | ReAct エージェントの組み立て方 |
| <https://dspy.ai/api/modules/ReAct/> | `max_iters` などの引数確認 |
| <https://dspy.ai/learn/evaluation/metrics/> | メトリクスの書き方と `trace` の意味 |
| <https://dspy.ai/learn/evaluation/data/> | `dspy.Example` とデータ分割 |
| <https://zenn.dev/akasan/articles/a040961f463bee> | 日本語。trajectory とエラー時の挙動 |
