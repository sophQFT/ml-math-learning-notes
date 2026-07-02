# Kubokawa Mathematical Statistics Notes

久保川『現代数理統計学の基礎』を学習する中で、定義・定理・証明の行間を自分の理解のために補った学習ノートです。

標本分布、推定量、不偏性、正規母集団における標本平均と不偏分散の独立性、t分布、確率変数列の収束、中心極限定理、順序統計量などを整理しています。

## PDF

### Full Version

* [View PDF](https://sophqft.github.io/ml-math-learning-notes/notes/statistics/kubokawa-math-statistics/pdf/kubokawa-math-statistics-notes.pdf)

### Chapter-wise PDFs

* [第1章：統計量・標本分布と推定量](https://sophqft.github.io/ml-math-learning-notes/notes/statistics/kubokawa-math-statistics/pdf/01-sampling-distribution-and-estimation.pdf)
* [第2章：標本平均・標本分散の独立性と t 分布](https://sophqft.github.io/ml-math-learning-notes/notes/statistics/kubokawa-math-statistics/pdf/02-normal-sample-distribution-and-independence.pdf)
* [第3章：確率変数列の収束と漸近理論](https://sophqft.github.io/ml-math-learning-notes/notes/statistics/kubokawa-math-statistics/pdf/03-convergence-and-asymptotic-theory.pdf)

## Overview

本ノートでは、数理統計学の基礎概念を、久保川『現代数理統計学の基礎』の行間を埋める形で整理しています。

前半では、無作為標本、統計量、標本分布、推定量、不偏性を導入し、標本平均、標本分散、不偏分散の性質を確認します。特に、標本分散が母分散の不偏推定量ではなく、不偏分散を用いる理由を、期待値の計算を通じて整理しています。

中盤では、正規母集団からの無作為標本に対して、標本平均と不偏分散が独立に分布することを扱います。ヘルマート行列を用いた直交変換により、標本平均、不偏分散、カイ二乗分布の関係を整理し、そこから t 分布の導出へつなげています。

後半では、確率収束、分布収束、大数の弱法則、中心極限定理、連続写像定理、スラツキーの定理を扱います。さらに、順序統計量を導入し、標本から得られる統計量の分布をどのように考えるかを整理しています。

## Contents

1. 統計量・標本分布と推定量

   * 無作為標本
   * 統計量と標本分布
   * 推定量と不偏性
   * 標本平均
   * 標本分散
   * 不偏分散
   * 標本分散が不偏でない理由

2. 標本平均・標本分散の独立性と t 分布

   * 正規母集団からの無作為標本
   * 標本平均の分布
   * 不偏分散とカイ二乗分布
   * 標本平均と不偏分散の独立性
   * ヘルマート行列を用いた証明
   * t 分布の導出
   * 標本平均の標準化と Student 化

3. 確率変数列の収束と漸近理論

   * 確率収束
   * 大数の弱法則
   * 分布収束
   * 中心極限定理
   * 連続写像定理
   * スラツキーの定理
   * 順序統計量

## Motivation

機械学習や深層学習におけるモデル評価、不確実性推定、統計的推論を理解するためには、数理統計の基礎が重要だと考えています。

特に、標本平均や分散がどのような分布に従うのか、推定量の不偏性とは何か、正規母集団において標本平均と不偏分散がなぜ独立になるのか、といった内容は、統計的なモデル理解の土台になります。

また、確率収束、分布収束、中心極限定理、スラツキーの定理は、漸近理論や大標本に基づく推論を理解するうえで不可欠です。本ノートでは、これらを公式として暗記するのではなく、定義・定理・証明の流れを追いながら整理することを目的としています。

## Keywords

* Mathematical Statistics
* Random Sample
* Statistic
* Sampling Distribution
* Estimator
* Unbiasedness
* Sample Mean
* Sample Variance
* Unbiased Variance
* Normal Population
* Chi-square Distribution
* Student's t-distribution
* Helmert Matrix
* Convergence in Probability
* Weak Law of Large Numbers
* Convergence in Distribution
* Central Limit Theorem
* Continuous Mapping Theorem
* Slutsky's Theorem
* Order Statistics

## Structure

```text
.
├── main.tex
├── preamble.tex
├── section1.tex
├── section2.tex
├── section3.tex
└── pdf/
    ├── kubokawa-math-statistics-notes.pdf
    ├── 01-sampling-distribution-and-estimation.pdf
    ├── 02-normal-sample-distribution-and-independence.pdf
    └── 03-convergence-and-asymptotic-theory.pdf
```

* `main.tex`: 全体をコンパイルするメインファイル
* `preamble.tex`: パッケージ・定理環境などの設定
* `section1.tex`: 統計量・標本分布と推定量
* `section2.tex`: 標本平均・標本分散の独立性と t 分布
* `section3.tex`: 確率変数列の収束と漸近理論
* `pdf/`: コンパイル済みPDF

## Notes

この資料は個人の学習ノートです。内容には誤りを含む可能性があります。

公開にあたっては、教科書本文の転載ではなく、自分の理解に基づく行間補足・導出整理として作成しています。
