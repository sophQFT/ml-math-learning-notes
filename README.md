# ML / Math Learning Notes

機械学習・深層学習を、数理統計・確率論・統計力学の観点から理解するために作成している学習ノート・ゼミ発表資料・論文紹介資料をまとめたリポジトリです。

📄 **Portfolio Site:** https://sophqft.github.io/ml-math-learning-notes/

単にモデルを利用するだけでなく、数式・実装・発表資料を通じて、その背後にある考え方を理解することを目的としています。特に、softmax とボルツマン分布の対応、Energy-Based Model、LLM のハルシネーション検出など、深層学習と統計力学・数理統計との接続に関心があります。

主に以下の内容を整理しています。

* 数理統計学・統計力学の行間補足ノート
* Dive into Deep Learning 輪読ゼミ用の発表資料
* 論文紹介資料
* Python / PyTorch による数値実験ノートブック（準備中）

## Selected Works

各資料の PDF は、上記 Portfolio Site（GitHub Pages）からも閲覧できます。

### Paper Reading: Spilled Energy in Large Language Models

追加学習なしで LLM の誤答（ハルシネーション）を検出する論文（Minut, Dewidar & Masi, ICLR 2026）の紹介資料です。
LLM の 1 ステップ予測を logit energy と marginal energy で書き直して Energy-Based Model として解釈し、本来一致すべき 2 量のズレ（spilled energy）を検出スコアとするまでの導出を 5 ステップで整理しています。

logit confidence や probe classifier など既存手法との違い、exact answer token の重要性や pooling を含む実用パイプライン、合成算術タスク・実世界 9 ベンチマークでの実験、手法の限界と実装チェックリストまで扱います。

* [Slide folder](slides/paper-reading/spilled-energy-llm/)
* [View PDF](https://sophqft.github.io/ml-math-learning-notes/slides/paper-reading/spilled-energy-llm/spilled-energy-llm.pdf)

**Keywords:** LLM, Hallucination Detection, Energy-Based Model, Softmax, Logit, Uncertainty Estimation

---

### Thermodynamics and Statistical Mechanics Notes

熱力学・統計力学を初めて学ぶ読者でも、磁性体の相転移を理解できるように作成した行間補足ノート（40ページ）です。教科書が省略しがちな導出を、できるだけ省かずに埋めることを主眼としています。

* **第1章 熱力学関数** — 断熱的到達可能性によるエントロピーの公理的導入（加法性・示量性・凹性）、基本関係式、ルジャンドル変換の詳説、ヘルムホルツ／ギッブスの自由エネルギー、磁化と磁場を共役変数とする磁性体の熱力学
* **第2章 カノニカル分布** — 分配関数の計算（鞍点法に対応する近似を含む）
* **第3章 相互作用がない電子系の磁性** — ディラック方程式の非相対論的近似からのパウリ項（ゼーマン相互作用）の導出、スピンと磁気モーメントの符号の注意、常磁性の統計力学的扱い
* **第4章 強磁性 Ising モデル** — 平均場近似、自己無撞着方程式、相転移と自発的対称性の破れ

* [Notes folder](notes/statistical-mechanics/thermo-stat-mech/)
* [View PDF](https://sophqft.github.io/ml-math-learning-notes/notes/statistical-mechanics/thermo-stat-mech/thermo-stat-mech-notes.pdf)

**Keywords:** Thermodynamics, Statistical Mechanics, Entropy, Legendre Transform, Canonical Distribution, Ising Model, Mean-Field Approximation, Phase Transition, Spontaneous Symmetry Breaking

---

### Softmax Regression and Boltzmann Distribution

Dive into Deep Learning 4.1〜4.3 節の輪読ゼミ発表資料（48ページ）です。
線形モデルからソフトマックス関数を導き、ボルツマン分布・分配関数との対応、最尤推定からの交差エントロピー損失の導出、情報量・情報エントロピーによる解釈までを整理しています。

後半では、PyTorch による Fashion-MNIST の読み込み・ミニバッチ・可視化と、Classifier 基底クラス・精度（accuracy）計算の実装を扱っています。

* [Slide folder](slides/d2l-softmax-regression/)
* [View PDF](https://sophqft.github.io/ml-math-learning-notes/slides/d2l-softmax-regression/softmax-regression-d2l.pdf)

**Keywords:** Softmax Regression, Cross Entropy, Boltzmann Distribution, Partition Function, PyTorch, Fashion-MNIST

---

### CNN: From Fully Connected Layers to Convolution

Dive into Deep Learning 7.1〜7.3 節の輪読ゼミ発表資料（26ページ）です。
全結合層に平行移動不変性と局所性という帰納バイアスの制約を課すと畳み込み層が導かれることを式変形で示し、チャネルの扱いまで整理しています。

後半では、相互相関演算と `corr2d`・畳み込み層の PyTorch 実装、エッジ検出の例、カーネルの学習、パディングとストライドを扱っています。

* [Slide folder](slides/d2l-cnn/)
* [View PDF](https://sophqft.github.io/ml-math-learning-notes/slides/d2l-cnn/d2l-cnn.pdf)

**Keywords:** CNN, Convolution, Inductive Bias, Translation Invariance, Locality, Weight Sharing, PyTorch

---

### Mathematical Statistics Notes

久保川『現代数理統計学の基礎』を学習する中で作成した行間補足ノート（14ページ＋章別PDF）です。

* **§1 統計量・標本分布と推定量** — 統計量と標本分布、推定量と不偏性、標本平均・標本分散・不偏分散
* **§2 標本平均・標本分散の独立性と t 分布** — ヘルマート行列を用いた独立性の証明と t 分布の導出
* **§3 確率変数列の収束と漸近理論** — 確率収束と大数の弱法則、分布収束と中心極限定理、スラツキーの定理などの収束の諸結果、順序統計量

* [Notes folder](notes/statistics/kubokawa-math-statistics/)
* [View PDF](https://sophqft.github.io/ml-math-learning-notes/notes/statistics/kubokawa-math-statistics/pdf/kubokawa-math-statistics-notes.pdf)

**Keywords:** Mathematical Statistics, Sampling Distribution, Unbiased Estimator, Helmert Matrix, Central Limit Theorem, Slutsky's Theorem, Order Statistics

---

### Article: ヘルマート行列で示す標本平均と不偏分散の独立性と t 分布

ヘルマート行列を用いて、正規母集団における標本平均と不偏分散の独立性、および t 分布の導入を整理した Zenn 記事です。上記の数理統計ノート §2 と対応しています。

* [Zenn Article](https://zenn.dev/soph/articles/53c0c74bf90e06)
* [PDF Notes](https://sophqft.github.io/ml-math-learning-notes/notes/statistics/kubokawa-math-statistics/pdf/02-normal-sample-distribution-and-independence.pdf)

**Keywords:** Helmert Matrix, t Distribution, Normal Distribution, Independence

## Repository Structure

```text
.
├── notes/
│   ├── statistics/
│   │   └── kubokawa-math-statistics/   # 数理統計ノート（LaTeX + 章別PDF）
│   ├── statistical-mechanics/
│   │   └── thermo-stat-mech/           # 熱力学・統計力学ノート
│   └── machine-learning/               # 機械学習の数理ノート（準備中）
├── slides/
│   ├── d2l-softmax-regression/         # D2L輪読: ソフトマックス回帰
│   ├── d2l-cnn/                        # D2L輪読: CNN
│   └── paper-reading/
│       └── spilled-energy-llm/         # 論文紹介: Spilled Energy in LLMs
├── notebooks/                          # 数値実験ノートブック（準備中）
│   ├── softmax-temperature/            #   softmax温度パラメータの可視化（予定）
│   └── clt-simulation/                 #   中心極限定理のシミュレーション（予定）
└── articles/                           # 記事原稿（準備中）
```

## Technical Skills Demonstrated

* 数式を用いた機械学習・深層学習の理解（数理統計・確率論・統計力学との接続）
* 論文の数式展開を追い、発表資料として整理する力
* LaTeX / Beamer による数式資料の作成
* PyTorch を用いた基礎的な実装（スライド内のコード。数値実験ノートブックは順次追加予定）
* GitHub / Zenn を用いた学習内容の継続的な発信

## Disclaimer

このリポジトリは個人の学習・発表資料を整理したものです。内容には誤りを含む可能性があります。教科書・講義資料・論文等を参考にした場合は、各資料内で出典を明記しています。
