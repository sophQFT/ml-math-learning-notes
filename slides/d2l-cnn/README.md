# CNN: From Fully Connected Layers to Convolution

Dive into Deep Learning 輪読ゼミで作成した、CNNの導入に関する発表資料です。

画像認識において、なぜ全結合層ではなく畳み込み層を用いるのかを、平行移動等価性、局所性、重み共有といった帰納バイアスの観点から整理しています。

## PDF

* [View PDF](https://sophqft.github.io/ml-math-learning-notes/slides/d2l-cnn/d2l-cnn.pdf)

## Overview

この資料では、画像を単なるベクトルではなく、空間構造を持つデータとして扱うことで、畳み込み層がどのように自然に導かれるかを説明しています。

前半では、全結合層を画像に適用した場合の問題点を確認し、画像認識に必要な性質として平行移動等価性、局所性、重み共有を導入します。これにより、CNNが単なる経験的なモデルではなく、画像データの性質に合った帰納バイアスを持つモデルであることを整理しています。

後半では、畳み込み層の具体的な計算として相互相関演算を扱い、手計算の例、`corr2d` の実装、エッジ検出、カーネルの学習、パディングとストライドによる出力サイズの調整について説明しています。

## Contents

1. 画像認識と全結合層の問題点

   * 画像をベクトルとして扱うことの問題
   * 全結合層におけるパラメータ数の増大
   * 画像の空間構造を利用する必要性

2. CNNを支える帰納バイアス

   * 平行移動等価性
   * 平行移動不変性
   * 局所性
   * 重み共有
   * 全結合層から畳み込み層への考え方

3. 畳み込み層の計算

   * 相互相関演算
   * カーネルと特徴マップ
   * チャネルを持つ画像の畳み込み
   * `corr2d` の実装

4. 畳み込みによる特徴抽出

   * エッジ検出
   * 垂直エッジ検出の例
   * 学習可能なカーネル
   * 畳み込み層による特徴抽出の意味

5. 出力サイズの調整

   * パディング
   * ストライド
   * 入力サイズ・カーネルサイズ・出力サイズの関係

## Motivation

CNNを単なる画像認識モデルとして覚えるのではなく、画像が持つ空間的な構造をニューラルネットワークにどのように組み込むかという観点から理解するために作成しました。

特に、全結合層から出発し、平行移動等価性、局所性、重み共有という制約を入れることで畳み込み層が自然に導かれる流れを重視しています。これにより、畳み込みを「画像に対する便利な演算」としてではなく、画像データに適した帰納バイアスとして理解することを目指しています。

## Keywords

* Convolutional Neural Network
* Inductive Bias
* Translation Equivariance
* Translation Invariance
* Locality
* Weight Sharing
* Cross-Correlation
* Feature Map
* Edge Detection
* Kernel Learning
* Padding
* Stride



## Reference

* Dive into Deep Learning, Chapter 7: Convolutional Neural Networks
* Dive into Deep Learning Japanese version, Chapter 7.1--7.3

## Notes

この資料は個人の学習・発表資料として作成したものです。内容には誤りを含む可能性があります。
