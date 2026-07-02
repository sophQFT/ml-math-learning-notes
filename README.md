# ML / Math Learning Notes

機械学習・深層学習を、数理統計・確率論・統計力学の観点から理解するために作成している学習ノート・ゼミ発表資料・論文紹介資料・実装メモをまとめたリポジトリです。

📄 **Portfolio Site:** https://sophqft.github.io/ml-math-learning-notes/

単にモデルを利用するだけでなく、数式・実装・発表資料を通じて、その背後にある考え方を理解することを目的としています。特に、Transformer / LLM、Energy-Based Model、Softmax、Boltzmann 分布、ハルシネーション検出など、深層学習と統計力学・数理統計との接続に関心があります。

主に以下の内容を整理しています。

* 深層学習・機械学習の学習ノート
* Dive into Deep Learning 輪読ゼミ用の発表資料
* 数理統計学・統計力学の行間補足ノート
* 論文紹介資料
* Python / PyTorch による基礎的な実装・数値実験
# ML / Math Learning Notes

機械学習・深層学習を、数理統計・確率論・統計力学の観点から理解するために作成している学習ノート・ゼミ発表資料・論文紹介資料・実装メモをまとめたリポジトリです。

📄 **Portfolio Site:** https://sophqft.github.io/ml-math-learning-notes/

単にモデルを利用するだけでなく、数式・実装・発表資料を通じて、その背後にある考え方を理解することを目的としています。特に、Transformer / LLM、Energy-Based Model、Softmax、Boltzmann 分布、ハルシネーション検出など、深層学習と統計力学・数理統計との接続に関心があります。

主に以下の内容を整理しています。

* 深層学習・機械学習の学習ノート
* Dive into Deep Learning 輪読ゼミ用の発表資料
* 数理統計学・統計力学の行間補足ノート
* 論文紹介資料
* Python / PyTorch による基礎的な実装・数値実験


## Selected Works

各資料の PDF は、上記 Portfolio Site（GitHub Pages）からも閲覧できます。

### Paper Reading: Spilled Energy in Large Language Models

LLM のハルシネーション検出に関する論文紹介資料です。
最終 softmax を Energy-Based Model として解釈し、連続する生成ステップ間のエネルギーのズレを用いて誤答を検出する考え方を整理しています。

特に、logit confidence や probe classifier との違い、exact answer token の重要性、spilled energy の定義と実用パイプラインをまとめています。

* [Slide folder](slides/paper-reading/spilled-energy-llm/)
* [View PDF](https://sophqft.github.io/ml-math-learning-notes/slides/paper-reading/spilled-energy-llm/spilled-energy-llm.pdf)

**Keywords:** LLM, Hallucination Detection, Energy-Based Model, Softmax, Logit, Uncertainty Estimation

---

### Thermodynamics and Statistical Mechanics Notes

熱力学・統計力学を初めて学ぶ読者でも、磁性体の相転移を理解できるように作成した行間補足ノートです。
エントロピーの公理的導入、ルジャンドル変換、ヘルムホルツ自由エネルギー、カノニカル分布、磁場中のスピン、常磁性、強磁性 Ising モデル、平均場近似、自発的対称性の破れまでを整理しています。

分配関数・自由エネルギー・平均場近似の考え方を、深層学習や Energy-Based Model を理解するための基礎として位置づけています。

* [Notes folder](notes/statistical-mechanics/thermo-stat-mech/)
* [View PDF](https://sophqft.github.io/ml-math-learning-notes/notes/statistical-mechanics/thermo-stat-mech/thermo-stat-mech-notes.pdf)

**Keywords:** Thermodynamics, Statistical Mechanics, Entropy, Legendre Transform, Canonical Distribution, Ising Model, Mean-Field Approximation, Phase Transition

---

### Softmax Regression and Boltzmann Distribution

Dive into Deep Learning 輪読ゼミで作成したソフトマックス回帰の発表資料です。
線形モデル、ソフトマックス関数、Boltzmann 分布との対応、交差エントロピー損失、PyTorch による Fashion-MNIST の読み込みなどを整理しています。

深層学習の出力層としての softmax を、統計力学の Boltzmann 分布との対応から理解することを意識しています。

* [Slide folder](slides/d2l-softmax-regression/)
* [View PDF](https://sophqft.github.io/ml-math-learning-notes/slides/d2l-softmax-regression/softmax-regression-d2l.pdf)

**Keywords:** Softmax Regression, Cross Entropy, Boltzmann Distribution, PyTorch, Fashion-MNIST

---

### CNN: From Fully Connected Layers to Convolution

Dive into Deep Learning 輪読ゼミで作成した CNN の導入資料です。
画像認識において全結合層ではなく畳み込み層を使う理由を、平行移動等価性・局所性・重み共有という観点から整理しています。

相互相関演算、エッジ検出、パディング、ストライドなど、CNN の基本的な計算も数式と図で説明しています。

* [Slide folder](slides/d2l-cnn/)
* [View PDF](https://sophqft.github.io/ml-math-learning-notes/slides/d2l-cnn/d2l-cnn.pdf)

**Keywords:** CNN, Convolution, Inductive Bias, Translation Equivariance, Locality, Weight Sharing

---

### Mathematical Statistics Notes

久保川『現代数理統計学の基礎』を学習する中で作成した行間補足ノートです。
標本分布、不偏性、ヘルマート行列を用いた独立性の証明、確率収束、大数の弱法則、中心極限定理、スラツキーの定理などを整理しています。

* [Notes folder](notes/statistics/kubokawa-math-statistics/)
* [View PDF](https://sophqft.github.io/ml-math-learning-notes/notes/statistics/kubokawa-math-statistics/pdf/kubokawa-math-statistics-notes.pdf)

**Keywords:** Mathematical Statistics, Sampling Distribution, Unbiased Estimator, Central Limit Theorem, Slutsky's Theorem

---

### Article: ヘルマート行列で示す標本平均と不偏分散の独立性と t 分布

ヘルマート行列を用いて、正規母集団における標本平均と不偏分散の独立性、および t 分布の導入を整理した Zenn 記事です。

* [Zenn Article](https://zenn.dev/soph/articles/53c0c74bf90e06)
* [PDF Notes](https://sophqft.github.io/ml-math-learning-notes/notes/statistics/kubokawa-math-statistics/pdf/02-normal-sample-distribution-and-independence.pdf)

**Keywords:** Helmert Matrix, t Distribution, Normal Distribution, Independence


## Repository Structure

```text
.
├── notes/
│   ├── statistics/              # 数理統計・確率論の学習ノート
│   ├── statistical-mechanics/   # 統計力学・熱力学の学習ノート
│   └── machine-learning/        # 機械学習・深層学習の数理ノート
├── slides/
│   ├── d2l-softmax-regression/  # D2L輪読ゼミ用のSoftmax回帰資料
│   ├── d2l-cnn/                 # D2L輪読ゼミ用のCNN資料
│   └── paper-reading/           # 論文紹介・輪読資料
├── notebooks/
│   ├── softmax-temperature/     # Softmax温度パラメータの数値実験
│   └── clt-simulation/          # 中心極限定理などのシミュレーション
└── articles/
    ├── zenn/                    # Zenn記事の原稿・関連資料
    └── qiita/                   # Qiita記事の原稿・関連資料
```


## Technical Skills Demonstrated

このリポジトリでは、以下のようなスキル・関心を示すことを意識しています。

* 数式を用いた機械学習・深層学習の理解
* 確率論・統計力学・数理統計と深層学習の接続
* 論文の数式展開を追い、発表資料として整理する力
* LaTeX / Beamer による数式資料の作成
* Python / PyTorch による基礎的な実装
* GitHub / Zenn を用いた学習内容の継続的な発信

## Disclaimer

このリポジトリは個人の学習・発表資料を整理したものです。内容には誤りを含む可能性があります。教科書・講義資料・論文等を参考にした場合は、各資料内で出典を明記しています。
