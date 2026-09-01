# DSPy によるツール利用最適化 — 実習ノート

> テーマ: **AIエージェントにおけるツール利用最適化技術の調査と評価**
> DSPy を題材として、AIエージェントにおけるツール利用最適化への適用可能性を調査し、
> 将来的な AIエージェント開発・運用への適用可能性を評価する。

このリポジトリは実習中に手元で開くための自習ノートと、そのまま動かせる実験コード一式。
**外部ネットワークに出られない環境でも動く**ように、題材はすべてダミーの辞書 DB にしてある。

---

## 目次

| ドキュメント | 内容 |
|---|---|
| [docs/day1_dspy_survey.md](docs/day1_dspy_survey.md) | **Day1**: DSPy の調査。Signature / Module / Optimizer、ReAct の内部構造、評価設計の考え方 |
| [docs/day2_tool_use_optimization.md](docs/day2_tool_use_optimization.md) | **Day2**: ツール利用最適化の調査と実装。当日のタイムテーブル付き |
| [docs/cheatsheet.md](docs/cheatsheet.md) | API 逐引き。困ったらここ |

---

## 5 日間の計画

| Day | やること | その日の終わりに手元にあるべきもの |
|---|---|---|
| 1 | DSPy 調査 | 用語と全体像の整理（→ `docs/day1_*`） |
| 2 | 実装 | **最適化していない ReAct が動き、`results/baseline.json` がある** |
| 3 | 最適化の適用 | `optimizer.compile()` が通り、指示文の前後差分が見えている |
| 4 | 評価・検証 | 前後比較の表（`results/comparison.md`） |
| 5 | まとめ・発表 | スライド |

**Day3 までに「動くベースライン」があるかで発表の質がほぼ決まる。**
Day2 は最適化に手を出さず、動かすことだけに全力を注ぐこと。

---

## クイックスタート

```bash
git clone <このリポジトリ>
cd dspy-internship

python -m venv .venv && source .venv/bin/activate
pip install -r code/requirements.txt

export OPENAI_API_KEY="sk-..."          # 使う LM に合わせて
export DSPY_MODEL="openai/gpt-4o-mini"

cd code
python agent.py            # 1 問だけ試す（疎通確認）
python run_baseline.py     # ベースライン評価  → results/baseline.json
python run_optimize.py bootstrap   # 最適化      → artifacts/optimized_bootstrap.json
python run_compare.py bootstrap    # 前後比較    → results/comparison.md
```

LM の設定は `code/config.py` の `MODEL` を書き換えるだけ。

---

## コードの構成

```
code/
├── config.py        LM の設定を 1 か所に集約。ここだけ書き換えれば全部に反映
├── tools.py         ツール定義（社内ヘルプデスク想定・ダミーDB・外部通信なし）
├── agent.py         Signature と ReAct エージェント
├── dataset.py       dspy.Example 20 件。train/val/dev に分割
├── metrics.py       ★ 評価指標。本テーマの中核
├── harness.py       評価ハーネス。全指標をまとめて集計
├── run_baseline.py  Day2: ベースライン測定
├── run_optimize.py  Day3: bootstrap / mipro / gepa
└── run_compare.py   Day4: 前後比較 → Markdown 表 + CSV
```

### なぜこの構成か

- **`metrics.py` が主役。** DSPy では人間の仕事が「プロンプトを書く」から「何を良しとするかを定義する」に移る。だから最も時間をかけるべきはここ
- **ツールは意図的に紛らわしくしてある。** `get_leave_balance`（有給残）と `get_attendance`（勤怠実績）、`search_faq`（制度）と個人データ系。最適化前に失敗する余地がないと、改善幅がゼロになって発表するものが無くなる
- **正解ラベルに `expected_tools` を持たせてある。** 回答の正誤だけでは「答えは合っているが無駄なツールを 5 回呼んだ」を検出できないため

---

## 測定する指標

| 階層 | 指標 | 意味 |
|---|---|---|
| タスク | 回答正解率 | 最終的に正しく答えられたか |
| 呼び出し | ツール選択 precision | **余計なツールを呼んでいないか** |
| 呼び出し | ツール選択 recall | 必要なツールを呼べたか |
| 呼び出し | ツール選択 F1 | 上の調和平均 |
| 効率 | ステップ効率 / 平均呼び出し回数 | 冗長さ |
| 効率 | 平均レイテンシ | 運用コスト |
| 頑健性 | ツール実行エラー率 | 引数生成の質 |

この 2 階層構成は既存ベンチマークの設計に倣っている。BFCL が呼び出しレベルの正確さを、τ-bench が複数のツール呼び出しにまたがるタスクレベルの完遂率を測る、という分担。

---

## 発表で使える論点（Day5 の仕込み）

1. **最適化前後で指示文がどう変わったか。** `run_optimize.py` が ReAct 内部の 2 つの predictor（`react` と `extract.predict`）の指示文を前後で表示する。この差分がクライマックス
2. **ツール数 vs 性能。** `tools.py` に `MINIMAL_TOOLS`（3 個）と `ALL_TOOLS`（5 個）がある。ある研究では ReAct のツール数を 3 → 10 に増やすと最適化後の F1 が U 字型のトレンドを示し、増やすことが必ずしも性能向上につながらなかったと報告されている。自分のデータで再現できれば「ツールは絞れ、docstring を磨け」という実運用への提言になる
3. **メトリクスの重みの感度。** ツール項の重みを 0 にしたら、最適化後のツール選択はどうなるか
4. **最適化器の比較。** bootstrap / mipro / gepa をコストと改善幅の両軸でプロット
5. **GEPA への日本語フィードバック。** `metrics.py` の `metric_with_feedback` は「不要なツールを呼んでいる」等を日本語で返す。GEPA はドメイン固有のテキストフィードバックを活用できるので、ツール利用に特化した指摘が指示文にどう反映されるかを見せられる

---

## 参考リンク

| リンク | 内容 |
|---|---|
| <https://dspy.ai/> | 公式 |
| <https://dspy.ai/learn/programming/tools/> | ツール利用 |
| <https://dspy.ai/api/modules/ReAct/> | ReAct API |
| <https://dspy.ai/learn/optimization/optimizers/> | 最適化器の一覧と選び方 |
| <https://dspy.ai/api/optimizers/MIPROv2/> | MIPROv2 の詳細 |
| <https://dspy.ai/getting-started/gepa-optimization/> | GEPA で ReAct を最適化する例 |
| <https://dspy.ai/tutorials/customer_service_agent/> | ReAct エージェント構築チュートリアル |
| <https://zenn.dev/akasan/articles/a040961f463bee> | 日本語の実践記事 |

---

## 注意

- このリポジトリのデータはすべて架空のダミー。実データは一切含まない
- 実習で扱う社内情報・社内コードはこのリポジトリに置かないこと
- 検証環境: dspy 3.3.1 / Python 3.12
