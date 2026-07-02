# ML / Math Learning Notes

機械学習・深層学習を、数理統計・確率論・統計力学の観点から理解するために作成している学習ノート、ゼミ発表資料、論文紹介資料、実装メモを整理しています。

特に、Transformer、Energy-Based Model、Softmax、Boltzmann分布、ハルシネーション検出など、深層学習と数理的構造のつながりに関心があります。

* [GitHub Repository](https://github.com/sophQFT/ml-math-learning-notes)
* [Zenn](https://zenn.dev/soph)

## Selected Works

### Paper Reading: Spilled Energy in Large Language Models

LLMのハルシネーション検出に関する論文紹介資料です。
最終softmaxをEnergy-Based Modelとして解釈し、連続する生成ステップ間のエネルギーのズレを用いて誤答を検出する考え方を整理しています。

* [View PDF](https://sophqft.github.io/ml-math-learning-notes/slides/paper-reading/spilled-energy-llm/spilled-energy-llm.pdf)
* [Source folder](https://github.com/sophQFT/ml-math-learning-notes/tree/main/slides/paper-reading/spilled-energy-llm)

**Keywords:** LLM, Hallucination Detection, Energy-Based Model, Softmax, Logit, Uncertainty Estimation

---

### Thermodynamics and Statistical Mechanics Notes

熱力学・統計力学を初めて学ぶ読者でも、磁性体の相転移を理解できるように作成した行間補足ノートです。
エントロピーの公理的導入、ルジャンドル変換、ヘルムホルツ自由エネルギー、カノニカル分布、磁場中のスピン、常磁性、強磁性Isingモデル、平均場近似、自発的対称性の破れまでを整理しています。

統計力学における分配関数・自由エネルギー・平均場近似の考え方を、深層学習やEnergy-Based Modelを理解するための基礎として位置づけています。

* [View PDF](https://sophqft.github.io/ml-math-learning-notes/notes/statistical-mechanics/thermo-stat-mech/thermo-stat-mech-notes.pdf)
* [Source folder](https://github.com/sophQFT/ml-math-learning-notes/tree/main/notes/statistical-mechanics/thermo-stat-mech)

**Keywords:** Thermodynamics, Statistical Mechanics, Entropy, Legendre Transform, Canonical Distribution, Ising Model, Mean-Field Approximation, Phase Transition

---

### Softmax Regression and Boltzmann Distribution

Dive into Deep Learning 輪読ゼミで作成したソフトマックス回帰の発表資料です。
線形モデル、ソフトマックス関数、ボルツマン分布との対応、交差エントロピー損失、PyTorchによるFashion-MNISTの読み込みなどを整理しています。

* [View PDF](https://sophqft.github.io/ml-math-learning-notes/slides/d2l-softmax-regression/softmax-regression-d2l.pdf)
* [Source folder](https://github.com/sophQFT/ml-math-learning-notes/tree/main/slides/d2l-softmax-regression)

**Keywords:** Softmax Regression, Cross Entropy, Boltzmann Distribution, PyTorch, Fashion-MNIST

---

### CNN: From Fully Connected Layers to Convolution

Dive into Deep Learning 輪読ゼミで作成したCNNの導入資料です。
画像認識において全結合層ではなく畳み込み層を使う理由を、平行移動等価性、局所性、重み共有という観点から整理しています。

* [View PDF](https://sophqft.github.io/ml-math-learning-notes/slides/d2l-cnn/d2l-cnn.pdf)
* [Source folder](https://github.com/sophQFT/ml-math-learning-notes/tree/main/slides/d2l-cnn)

**Keywords:** CNN, Convolution, Inductive Bias, Translation Equivariance, Locality, Weight Sharing

---

### Mathematical Statistics Notes

久保川『現代数理統計学の基礎』を学習する中で作成した行間補足ノートです。
標本分布、不偏性、ヘルマート行列を用いた独立性の証明、確率収束、大数の弱法則、中心極限定理、スラツキーの定理などを整理しています。

* [View PDF](https://sophqft.github.io/ml-math-learning-notes/notes/statistics/kubokawa-math-statistics/pdf/kubokawa-math-statistics-notes.pdf)
* [Source folder](https://github.com/sophQFT/ml-math-learning-notes/tree/main/notes/statistics/kubokawa-math-statistics)

**Keywords:** Mathematical Statistics, Sampling Distribution, Unbiased Estimator, Central Limit Theorem, Slutsky's Theorem

---

### Article: ヘルマート行列で示す標本平均と不偏分散の独立性と t 分布

ヘルマート行列を用いて、正規母集団における標本平均と不偏分散の独立性、および t 分布の導入を整理したZenn記事です。

* [Zenn Article](https://zenn.dev/soph/articles/53c0c74bf90e06)
* [PDF Notes](https://sophqft.github.io/ml-math-learning-notes/notes/statistics/kubokawa-math-statistics/pdf/02-normal-sample-distribution-and-independence.pdf)

**Keywords:** Helmert Matrix, t Distribution, Normal Distribution, Independence

## Repository Structure

```text
.
├── notes/
│   ├── statistics/
│   ├── statistical-mechanics/
│   └── machine-learning/
├── slides/
│   ├── d2l-softmax-regression/
│   ├── d2l-cnn/
│   └── paper-reading/
├── notebooks/
│   ├── softmax-temperature/
│   └── clt-simulation/
└── articles/
    ├── zenn/
    └── qiita/
```

## Technical Skills Demonstrated

* 数式を用いた機械学習・深層学習の理解
* 確率論・統計力学・数理統計と深層学習の接続
* 論文の数式展開を追い、発表資料として整理する力
* LaTeX / Beamer による数式資料作成
* Python / PyTorch による基礎的な実装
* GitHub / Zenn を用いた学習内容の継続的な発信


## Notes

このサイトは個人の学習・発表資料を整理したものです。内容には誤りを含む可能性があります。

教科書・講義資料・論文等を参考にした場合は、各資料内で出典を明記しています。
