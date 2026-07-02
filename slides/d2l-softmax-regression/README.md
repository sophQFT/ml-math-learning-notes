# Softmax Regression: Linear Neural Networks for Classification

Dive into Deep Learning 輪読ゼミで作成した、ソフトマックス回帰に関する発表資料です。

分類問題において、線形モデルの出力をどのように確率として解釈し、交差エントロピー損失によって学習するのかを整理しています。また、ソフトマックス関数と統計力学におけるボルツマン分布の対応関係についても説明しています。

## PDF

* [View PDF](https://sophqft.github.io/ml-math-learning-notes/slides/d2l-softmax-regression/softmax-regression-d2l.pdf)

## Overview

この資料では、線形回帰から分類問題へ拡張する流れを出発点として、ソフトマックス回帰の考え方を説明しています。

前半では、分類問題におけるラベル表現、クラスごとのスコアを出す線形モデル、ソフトマックス関数による確率化を整理します。特に、線形モデルの出力をそのまま確率とみなせない理由を確認し、ソフトマックス関数がスコアを確率分布へ変換する役割を説明しています。

また、ソフトマックス関数と統計力学におけるボルツマン分布の形の類似性に注目し、スコア、エネルギー、分配関数の対応関係を整理しています。

後半では、最尤推定の観点から交差エントロピー損失を導入し、ソフトマックスと交差エントロピーを組み合わせることで、勾配が「予測確率 − 正解ラベル」という簡単な形になることを説明しています。さらに、Fashion-MNISTの読み込み、ミニバッチ処理、分類器の精度評価といった実装上の基礎も扱っています。

## Contents

1. ソフトマックス回帰の概要

   * 回帰から分類へ
   * ハードな分類とソフトな分類
   * one-hot エンコーディング
   * クラスごとのスコアを出す線形モデル
   * 線形モデルの行列表記

2. ソフトマックス関数

   * 線形モデルの出力をそのまま確率にできない理由
   * ソフトマックス関数による確率分布への変換
   * 最大クラスと argmax の関係
   * ミニバッチに対するベクトル化された計算

3. ソフトマックスとボルツマン分布

   * ボルツマン分布の形
   * スコアとエネルギーの対応
   * 分配関数との対応
   * 統計力学的な見方からのソフトマックスの理解

4. 損失関数と交差エントロピー

   * 正解クラスに高い確率を与える損失関数
   * 最尤推定からの導出
   * 負の対数尤度
   * one-hot ラベルと交差エントロピー損失
   * 正解クラスの確率と損失の関係

5. ソフトマックスと交差エントロピーの勾配

   * 交差エントロピーにソフトマックスを代入する計算
   * 勾配が「予測確率 − 正解ラベル」になる理由
   * 勾配降下法における更新の意味
   * 正解クラスのスコアを上げ、不正解クラスのスコアを下げる仕組み

6. 情報理論から見た交差エントロピー

   * 情報量
   * 情報エントロピー
   * 真の分布と予測分布
   * 交差エントロピー損失の再解釈

7. Fashion-MNIST による実践

   * Fashion-MNIST データセット
   * PyTorch によるデータセットの読み込み
   * 画像テンソルの形
   * ミニバッチ処理
   * 画像とラベルの可視化

8. 分類モデルの基礎

   * Classifier クラス
   * 検証ステップ
   * 分類精度 accuracy
   * argmax による予測クラスの決定
   * D2L における精度計算の実装

## Motivation

ソフトマックス回帰を、単なる多クラス分類の基本モデルとしてではなく、確率分布、最尤推定、交差エントロピー、情報理論、統計力学との関係から理解するために作成しました。

特に、ソフトマックス関数がスコアを確率分布に変換する仕組み、交差エントロピー損失が最尤推定から自然に導かれること、さらにソフトマックスとボルツマン分布の形が対応していることを重視しています。

この理解は、深層学習における分類モデルだけでなく、Transformer や大規模言語モデルにおける logit、確率分布、エネルギー的解釈を考えるための基礎にもなります。


## Keywords

* Softmax Regression
* Linear Neural Network
* Classification
* One-hot Encoding
* Softmax Function
* Cross-Entropy Loss
* Maximum Likelihood Estimation
* Negative Log-Likelihood
* Boltzmann Distribution
* Partition Function
* Information Entropy
* Fashion-MNIST
* PyTorch
* Mini-batch
* Accuracy



## Reference

* Dive into Deep Learning, Chapter 4: Linear Neural Networks for Classification
* Dive into Deep Learning Japanese version, Chapter 4.1--4.3

## Notes

この資料は個人の学習・発表資料として作成したものです。内容には誤りを含む可能性があります。
