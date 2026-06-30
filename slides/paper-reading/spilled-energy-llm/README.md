# Spilled Energy in Large Language Models

LLMのハルシネーション検出に関する論文  
*Spilled Energy in Large Language Models* の紹介資料です。

## Overview

LLMの最終softmaxをEnergy-Based Modelとして解釈し、連続する生成ステップ間で生じるエネルギーのズレを用いて、誤答やハルシネーションを検出する考え方を整理しました。

特に、logit confidence や probe classifier との違い、exact answer token の重要性、spilled energy の定義と実用パイプラインを中心にまとめています。

## Keywords

- Large Language Model
- Hallucination Detection
- Energy-Based Model
- Softmax
- Logit
- Spilled Energy
- Uncertainty Estimation

## PDF

- [Slide PDF](./spilled-energy-llm.pdf)

## Paper

- Adrian R. Minut, Hazem Dewidar, Iacopo Masi, *Spilled Energy in Large Language Models*
- arXiv:2602.18671
