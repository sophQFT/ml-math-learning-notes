# ML / Math Learning Notes

機械学習・深層学習・数理統計・統計力学を数理的に理解するために作成している学習ノート、ゼミ発表資料、実装メモを整理するリポジトリです。

## Interests

- Machine Learning / Deep Learning
- Transformer / Attention
- Mathematical Statistics
- Statistical Mechanics
- Probabilistic Modeling

## Highlights

### Softmax Regression and Boltzmann Distribution

Dive into Deep Learning 輪読ゼミで作成したソフトマックス回帰の発表資料です。  
線形モデル、ソフトマックス関数、ボルツマン分布との対応、交差エントロピー損失、PyTorchによるFashion-MNISTの読み込みなどを整理しています。

- [Slide folder](slides/d2l-softmax-regression/)
- [Slide PDF](slides/d2l-softmax-regression/softmax-regression-d2l.pdf)

### CNN: From Fully Connected Layers to Convolution

Dive into Deep Learning 輪読ゼミで作成したCNNの導入資料です。  
画像認識において全結合層ではなく畳み込み層を使う理由を、平行移動等価性、局所性、重み共有という観点から整理しています。相互相関演算、エッジ検出、パディング、ストライドなど、CNNの基本的な計算も数式と図で説明しています。

- [Slide folder](slides/d2l-cnn/)
- [Slide PDF](slides/d2l-cnn/d2l-cnn.pdf)

### Paper Reading: Spilled Energy in Large Language Models

LLMのハルシネーション検出に関する論文紹介資料です。  
最終softmaxをEnergy-Based Modelとして解釈し、連続する生成ステップ間のエネルギーのズレを用いて誤答を検出する考え方を整理しています。logit confidence や probe classifier との違い、exact answer token の重要性、spilled energy の定義と実用パイプラインをまとめています。

- [Slide folder](slides/paper-reading/spilled-energy-llm/)
- [Slide PDF](slides/paper-reading/spilled-energy-llm/spilled-energy-llm.pdf)


### Mathematical Statistics Notes

久保川『現代数理統計学の基礎』を学習する中で作成した行間補足ノートです。  
標本分布、不偏性、ヘルマート行列を用いた独立性の証明、確率収束、大数の弱法則、中心極限定理、スラツキーの定理などを整理しています。

- [Notes folder](notes/statistics/kubokawa-math-statistics/)
- [Full PDF](notes/statistics/kubokawa-math-statistics/pdf/kubokawa-math-statistics-notes.pdf)

### Article: ヘルマート行列で示す標本平均と不偏分散の独立性と t 分布

ヘルマート行列を用いて、正規母集団における標本平均と不偏分散の独立性、および t 分布の導入を整理したZenn記事です。  

- [Zenn Article](https://zenn.dev/soph/articles/53c0c74bf90e06)
- [Notes Folder](notes/statistics/kubokawa-math-statistics/)
- [PDF Notes](notes/statistics/kubokawa-math-statistics/pdf/02-normal-sample-distribution-and-independence.pdf)
## Contents

### Notes

- `notes/statistics/`
  - 数理統計学の学習ノート
  - 標本分布、不偏性、確率収束、大数の弱法則、中心極限定理、ヘルマート行列を用いた独立性の証明など

- `notes/thermodynamics/`
  - 熱力学の学習ノート

- `notes/statistical-mechanics/`
  - 統計力学の学習ノート
  - 分配関数、ボルツマン分布、自由エネルギーなど

- `notes/machine-learning/`
  - 機械学習・深層学習の数理的理解に関するノート

### Slides

- `slides/d2l-softmax-regression/`
  - Dive into Deep Learning 輪読ゼミ用に作成したソフトマックス回帰の発表資料
  - 線形モデル、ソフトマックス関数、ボルツマン分布との対応、交差エントロピー損失、PyTorchによるFashion-MNISTの読み込みなどを整理

- `slides/paper-reading/`
  - 論文紹介ゼミで作成した資料を整理予定

### Notebooks

- `notebooks/softmax-temperature/`
  - ソフトマックス関数と温度パラメータの関係を確認する予定

- `notebooks/clt-simulation/`
  - 中心極限定理などをPythonで数値的に確認する予定

### Articles

- `articles/zenn/`
  - Zenn公開用の記事原稿

- `articles/qiita/`
  - Qiita公開用の記事原稿

## Purpose

単に機械学習モデルを使うだけでなく、確率論・統計力学・数理統計の観点から、深層学習を理解することを目的としています。

## Notes

このリポジトリは個人の学習・発表資料を整理したものです。内容には誤りを含む可能性があります。教科書・講義資料・論文等を参考にした場合は、各資料内で出典を明記します。
