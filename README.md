# ML / Math Learning Notes

機械学習・深層学習を、数理統計・確率論・統計力学の観点から理解するために作成している学習ノート、ゼミ発表資料、論文紹介資料、実装メモを整理するリポジトリです。

特に、Transformer、Energy-Based Model、Softmax、Boltzmann分布、ハルシネーション検出など、深層学習と数理的構造のつながりに関心があります。

## Portfolio Site

- [GitHub Pages](https://sophqft.github.io/ml-math-learning-notes/)

## About This Repository

このリポジトリでは、単に機械学習モデルを利用するだけでなく、数式・実装・発表資料を通じて、モデルの背後にある考え方を理解することを目的としています。

主に以下の内容を整理しています。

* 深層学習・機械学習の学習ノート
* Dive into Deep Learning 輪読ゼミ用の発表資料
* 数理統計学・統計力学の行間補足ノート
* 論文紹介資料
* Python / PyTorch による基礎的な実装・数値実験

## Interests

* Machine Learning / Deep Learning
* Transformer / Attention
* Energy-Based Model
* Hallucination Detection
* Mathematical Statistics
* Statistical Mechanics
* Probabilistic Modeling

## Selected Works

### Paper Reading: Spilled Energy in Large Language Models

LLMのハルシネーション検出に関する論文紹介資料です。
最終softmaxをEnergy-Based Modelとして解釈し、連続する生成ステップ間のエネルギーのズレを用いて誤答を検出する考え方を整理しています。

特に、logit confidence や probe classifier との違い、exact answer token の重要性、spilled energy の定義と実用パイプラインをまとめています。

* [Slide folder](slides/paper-reading/spilled-energy-llm/)
* [View PDF](https://sophqft.github.io/ml-math-learning-notes/slides/paper-reading/spilled-energy-llm/spilled-energy-llm.pdf)

**Keywords:** LLM, Hallucination Detection, Energy-Based Model, Softmax, Logit, Uncertainty Estimation

---

### Thermodynamics and Statistical Mechanics Notes

熱力学・統計力学を初めて学ぶ読者でも、磁性体の相転移を理解できるように作成した行間補足ノートです。
エントロピーの公理的導入、ルジャンドル変換、ヘルムホルツ自由エネルギー、カノニカル分布、磁場中のスピン、常磁性、強磁性Isingモデル、平均場近似、自発的対称性の破れまでを整理しています。

統計力学における分配関数・自由エネルギー・平均場近似の考え方を、深層学習やEnergy-Based Modelを理解するための基礎として位置づけています。

* [Notes folder](notes/statistical-mechanics/thermo-stat-mech/)
* [View PDF](https://sophqft.github.io/ml-math-learning-notes/notes/statistical-mechanics/thermo-stat-mech/thermo-stat-mech-notes.pdf)

**Keywords:** Thermodynamics, Statistical Mechanics, Entropy, Legendre Transform, Canonical Distribution, Ising Model, Mean-Field Approximation, Phase Transition

---

### Softmax Regression and Boltzmann Distribution

Dive into Deep Learning 輪読ゼミで作成したソフトマックス回帰の発表資料です。
線形モデル、ソフトマックス関数、ボルツマン分布との対応、交差エントロピー損失、PyTorchによるFashion-MNISTの読み込みなどを整理しています。

深層学習の出力層としてのsoftmaxを、統計力学のBoltzmann分布との対応から理解することを意識しています。

* [Slide folder](slides/d2l-softmax-regression/)
* [View PDF](https://sophqft.github.io/ml-math-learning-notes/slides/d2l-softmax-regression/softmax-regression-d2l.pdf)

**Keywords:** Softmax Regression, Cross Entropy, Boltzmann Distribution, PyTorch, Fashion-MNIST

---

### CNN: From Fully Connected Layers to Convolution

Dive into Deep Learning 輪読ゼミで作成したCNNの導入資料です。
画像認識において全結合層ではなく畳み込み層を使う理由を、平行移動等価性、局所性、重み共有という観点から整理しています。

相互相関演算、エッジ検出、パディング、ストライドなど、CNNの基本的な計算も数式と図で説明しています。

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

ヘルマート行列を用いて、正規母集団における標本平均と不偏分散の独立性、および t 分布の導入を整理したZenn記事です。

* [Zenn Article](https://zenn.dev/soph/articles/53c0c74bf90e06)
* [Notes folder](notes/statistics/kubokawa-math-statistics/)
* [PDF Notes](notes/statistics/kubokawa-math-statistics/pdf/02-normal-sample-distribution-and-independence.pdf)

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

## Contents

### Notes

* `notes/statistics/`

  * 数理統計学の学習ノート
  * 標本分布、不偏性、確率収束、大数の弱法則、中心極限定理、ヘルマート行列を用いた独立性の証明など

* `notes/statistical-mechanics/`

  * 統計力学の学習ノート
  * エントロピー、ルジャンドル変換、カノニカル分布、分配関数、自由エネルギー、Isingモデル、平均場近似など


* `notes/machine-learning/`

  * 機械学習・深層学習の数理的理解に関するノート

### Slides

* `slides/d2l-softmax-regression/`

  * Dive into Deep Learning 輪読ゼミ用に作成したソフトマックス回帰の発表資料
  * Softmax、交差エントロピー、Boltzmann分布との対応、PyTorchによるFashion-MNISTの読み込みなど

* `slides/d2l-cnn/`

  * Dive into Deep Learning 輪読ゼミ用に作成したCNNの導入資料
  * 畳み込み層、平行移動等価性、局所性、重み共有、相互相関演算、パディング、ストライドなど

* `slides/paper-reading/`

  * 深層学習・言語モデル・統計的機械学習に関する論文紹介資料

### Notebooks

* `notebooks/softmax-temperature/`

  * ソフトマックス関数と温度パラメータの関係を確認するための数値実験

* `notebooks/clt-simulation/`

  * 中心極限定理などをPythonで数値的に確認するための実験

### Articles

* `articles/zenn/`

  * Zenn公開用の記事原稿

* `articles/qiita/`

  * Qiita公開用の記事原稿

## Technical Skills Demonstrated

このリポジトリでは、以下のようなスキル・関心を示すことを意識しています。

* 数式を用いた機械学習・深層学習の理解
* 確率論・統計力学・数理統計と深層学習の接続
* 論文の数式展開を追い、発表資料として整理する力
* LaTeX / Beamer による数式資料作成
* Python / PyTorch による基礎的な実装
* GitHub / Zenn を用いた学習内容の継続的な発信

## Purpose

将来的には、機械学習エンジニア、データサイエンティスト、または深層学習の研究開発職を目指しています。

そのために、モデルをブラックボックスとして使うだけでなく、数理的背景を理解し、実装・実験・発表資料としてアウトプットすることを重視しています。

## Notes

このリポジトリは個人の学習・発表資料を整理したものです。内容には誤りを含む可能性があります。教科書・講義資料・論文等を参考にした場合は、各資料内で出典を明記します。
