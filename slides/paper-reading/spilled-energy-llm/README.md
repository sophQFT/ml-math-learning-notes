# Spilled Energy in Large Language Models

LLMのハルシネーション検出を「生成過程におけるエネルギーのズレ」として読む論文紹介スライドです。

Adrian R. Minut, Hazem Dewidar, Iacopo Masi による *Spilled Energy in Large Language Models* を題材に、LLMの最終 softmax を Energy-Based Model として解釈し、追加学習なしで誤答・ハルシネーションを検出する考え方を整理しています。

## PDF

* [View PDF](https://sophqft.github.io/ml-math-learning-notes/slides/paper-reading/spilled-energy-llm/spilled-energy-llm.pdf)

## Overview

この資料では、LLMの回答が誤答やハルシネーションであるかを、外部検索や追加学習なしで検出できるかという問いを出発点にしています。

前半では、LLMが入力テキストをトークン列として処理し、Transformerを通じて各時刻の logit を出力し、softmax によって次トークン確率を与える流れを整理します。そのうえで、自己回帰モデルの chain rule と softmax 分類器の形を確認し、LLMの1-step予測を Energy-Based Model として読むための準備を行います。

中盤では、生成されたトークンの logit に由来する logit energy と、語彙全体の logsumexp に由来する marginal energy を導入します。そして、理論的には同じ prefix に対応して整合してほしい2つのエネルギーが、実際のLLMでは異なる時刻・異なるsoftmax成分から測られることに注目し、その差を spilled energy として定義します。

後半では、spilled energy を用いた誤答検出のパイプライン、exact answer token の重要性、token-level score から answer-level score への pooling、合成算術タスクや実世界ベンチマークでの実験結果を整理します。また、logit confidence、p(true)、probe classifier などの既存手法との違いや、white-box な logit アクセスが必要であるという実用上の限界についても扱っています。

## Contents

1. 論文の問題設定

   * LLMのハルシネーション検出
   * 外部検索や追加学習なしで検出するという設定
   * 最終 softmax を Energy-Based Model として読む着眼点
   * logit confidence だけでは捉えにくい誤答検出

2. LLMの自己回帰生成と softmax

   * 入力テキストからトークン列への変換
   * token ID、one-hot、埋め込み、Transformer、logit
   * 次トークン確率としての softmax
   * prefix から次トークンを予測する自己回帰モデル
   * 生成されたトークンの logit の読み方

3. 自己回帰分解とエネルギー解釈

   * 系列確率の chain rule
   * 条件付き確率を prefix 確率の比として見る考え方
   * LLMの1-step予測を softmax 分類器として表す
   * Energy-Based Model の形との対応
   * logit energy と marginal energy の導入

4. Spilled Energy の定義

   * token logit に由来する logit energy
   * 語彙全体の logsumexp に由来する marginal energy
   * 負の対数尤度とエネルギー差の関係
   * 理論的に整合してほしい2つのエネルギー
   * spilled energy を「エネルギーのズレ」として定義する流れ
   * Rome を例にした具体的な計算イメージ

5. 既存手法との比較

   * logit confidence
   * p(true)
   * probe classifier
   * hidden state を読む方法と softmax の数理構造を読む方法の違い
   * 追加学習なしで logit 列からスコアを計算する利点

6. 実用パイプライン

   * LLMによる回答生成と logit の保存
   * exact answer token の特定
   * token-level spilled energy の計算
   * min / max / mean pooling による answer-level score
   * 閾値やランキングによる誤答候補の検出
   * token-level heatmap と answer-level score の使い分け

7. 実験結果

   * 合成算術タスク
   * 正解と誤答の score 分布
   * 実世界9ベンチマークでの評価
   * LLaMA、Mistral、Gemma など複数モデルでの比較
   * cross-dataset generalization
   * AuROC による性能評価

8. 限界と実装上の注意

   * exact answer token に限定する必要性
   * 句読点・機能語による false positive
   * white-box な logit アクセスの必要性
   * API利用時の制約
   * しきい値の校正
   * 外部事実検証ではなく内部異常スコアとして使う位置づけ

9. 論文の貢献

   * 最終 softmax を EBM として読む数理的貢献
   * 追加訓練なしで token-local な hallucination score を計算する方法的貢献
   * 合成算術・実世界ベンチマーク・cross-dataset 転移で有効性を示した実験的貢献

## Motivation

LLMのハルシネーションを検出する方法には、外部検索、自己評価、probe classifier、logit confidence などがあります。しかし、これらの方法は外部情報に依存したり、追加学習を必要としたり、モデルの自信過剰を見逃したりする場合があります。

この論文では、LLMの最終 softmax が持つ数理構造に注目し、生成された答えトークンの「自信」だけでなく、そのトークンを含む prefix が次の時刻でどれだけ整合的な分布を作るかを見ることで、誤答検出に使える信号を取り出しています。

この資料は、softmax、logit、logsumexp、Energy-Based Model、自己回帰モデルの関係を整理し、LLMのハルシネーション検出を数理的に理解するために作成しました。特に、自分の研究関心である LLM の信頼性評価やエネルギーベースの解釈につなげることを意識しています。

## Keywords

* Large Language Model
* Hallucination Detection
* Spilled Energy
* Energy-Based Model
* Softmax
* Logit
* Logsumexp
* Autoregressive Model
* Chain Rule
* Negative Log-Likelihood
* Exact Answer Token
* Token-level Score
* Answer-level Score
* Pooling
* Probe Classifier
* p(true)
* Logit Confidence
* AuROC
* White-box Detection
* Uncertainty Estimation

## Reference

* Adrian R. Minut, Hazem Dewidar, Iacopo Masi, *Spilled Energy in Large Language Models*
* arXiv: https://arxiv.org/abs/2602.18671
* Code: https://github.com/OmnAI-Lab/spilled-energy/

## Notes

この資料は個人の学習・論文紹介資料として作成したものです。内容には誤りを含む可能性があります。

論文の内容をそのまま網羅することよりも、LLMの自己回帰生成、softmax、Energy-Based Model、ハルシネーション検出の関係を自分の理解に基づいて整理することを目的としています。
