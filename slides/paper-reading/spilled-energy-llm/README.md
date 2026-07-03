# Paper Reading: Spilled Energy in Large Language Models

Minut, Dewidar & Masi "Spilled Energy in Large Language Models" (ICLR 2026) の論文紹介スライドです。
追加学習なしで LLM の誤答（ハルシネーション）を検出する手法を扱っており、LLM の1ステップ予測を logit energy と marginal energy で書き直して Energy-Based Model として解釈し、本来一致すべき2量のズレ（spilled energy）を検出スコアとするまでの導出を5ステップで整理しています。既存手法との比較、exact answer token や pooling を含む実用パイプライン、合成算術タスク・実世界9ベンチマークでの実験、手法の限界と実装チェックリストまで扱います。

Slides on a training-free hallucination-detection method: the LLM's one-step prediction is rewritten with logit and marginal energies and read as an energy-based model, and the mismatch between two quantities that should agree — the spilled energy — becomes the detection score.

**[📄 スライドPDFを見る / View slides (PDF)](https://sophqft.github.io/ml-math-learning-notes/slides/paper-reading/spilled-energy-llm/spilled-energy-llm.pdf)**

- 論文 / Paper: [arXiv:2602.18671](https://arxiv.org/abs/2602.18671)
- 形式: Beamer / 27 pages
