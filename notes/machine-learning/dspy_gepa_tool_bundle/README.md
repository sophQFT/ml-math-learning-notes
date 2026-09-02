# DSPy / GEPA ツール利用最適化 実習パッケージ

## 推奨する読む順番

1. `dspy_gepa_complex_experiment_copy_paste.md`  
   実習中にJupyterへ上からコピーして実行する手順書です。

2. `dspy_gepa_zero_to_gepa_beginner_reference.md`  
   PythonとDSPyの用語・記法を0から確認する辞書兼教科書です。

3. `dspy_gepa_10min_slides_and_script.md`  
   10分発表のスライド内容、時間配分、読み上げ原稿です。

4. `dspy_gepa_internship_10min_presentation.pptx`  
   上記原稿に対応したPowerPointです。Slide 7の`XX.X`と`［入力］`を実測値へ置き換えてください。

## 補助ファイル

- `dspy_gepa_complex_experiment_code.py`  
  手順書内のコードを1ファイルへまとめたものです。メンター配布の3ツールを定義した後に使います。

- `dspy_gepa_internship_10min_montage.png`  
  PowerPoint全体のプレビューです。

## 公開前の注意

次をGitHubへ含めないでください。

- Azure OpenAI APIキー
- MCP APIキー
- Azure endpoint
- MCPサーバーURL・社内IP
- 公開許可のないログ、問題、画面写真

## 実験上の注意

新しい複雑問題でもGEPAの改善が必ず出るとは限りません。最初に決めたtrain / validation / testを維持し、良い結果が出るまでtestを変更しないでください。改善が小さい場合も、ツール順序、引数、最終回答、1ツール対照を分けて報告できます。
