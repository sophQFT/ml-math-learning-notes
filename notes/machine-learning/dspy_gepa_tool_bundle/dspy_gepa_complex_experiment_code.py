"""
DSPy + GEPA 複数ツール実験コード

前提:
- メンター配布コードを先に実行済み
- 次の3関数が定義済み
    calculate_expression
    analyze_numbers
    convert_units
- Azure OpenAI設定の変数が定義済み
    AZURE_OPENAI_API_KEY
    AZURE_OPENAI_ENDPOINT
    AZURE_OPENAI_API_VERSION
    AZURE_OPENAI_DEPLOYMENT

このファイルは、上記のコードの後ろへ貼り付ける部分だけを収録しています。
"""

import csv
import json
from statistics import mean

import dspy


# ReActが行動できる最大回数。
# 4個の外部ツール + finish を使えるよう、余裕を持って8にする。
MAX_ITERS = 8

# Judgeへ見せるツール説明。
AVAILABLE_TOOLS_TEXT = """
calculate_expression:
1つの数式を計算する。例: 128 * 1.08、(2000 + 350) / 3。

analyze_numbers:
数値配列を統計分析する。平均、中央値、標準偏差、分散などを求める。

convert_units:
長さ、重さ、温度、時間、データサイズなどの単位を変換する。
""".strip()


class ToolQA(dspy.Signature):
    """
    ユーザーの質問に正確に答えてください。
    必要な場合は、利用可能なツールを使ってください。
    ツールの実行結果を確認し、最終回答を作ってください。
    """

    question: str = dspy.InputField(
        desc="ユーザーが入力した質問"
    )

    answer: str = dspy.OutputField(
        desc="単位を含む、分かりやすい最終回答"
    )


def extract_tool_history(trajectory):
    """
    ReActのtrajectoryから、外部ツールの名前・引数・結果を取り出す。
    finishはReActの終了合図なので除外する。
    """

    trajectory = trajectory or {}

    tool_calls = []
    tool_results = []
    tool_names = []

    # tool_name_0、tool_name_1 ... の番号を集める。
    step_numbers = []

    for key in trajectory:
        if key.startswith("tool_name_"):
            step_text = key.removeprefix("tool_name_")

            try:
                step_numbers.append(int(step_text))
            except ValueError:
                # 想定外のキーは無視する。
                continue

    # 0、1、2 ... の順番に並べて履歴を取り出す。
    for step in sorted(step_numbers):
        tool_name = str(
            trajectory.get(f"tool_name_{step}", "")
        )

        if tool_name in ("", "finish", "submit"):
            continue

        tool_args = trajectory.get(
            f"tool_args_{step}",
            {},
        )

        observation = trajectory.get(
            f"observation_{step}"
        )

        tool_names.append(tool_name)

        tool_calls.append(
            {
                "step": step,
                "tool": tool_name,
                "arguments": tool_args,
            }
        )

        tool_results.append(
            {
                "step": step,
                "tool": tool_name,
                "result": observation,
            }
        )

    return tool_calls, tool_results, tool_names


class MCPToolAgent(dspy.Module):
    """メンター配布の3ツールを利用するReAct Agent。"""

    def __init__(self):
        super().__init__()

        self.agent = dspy.ReAct(
            ToolQA,
            tools=[
                calculate_expression,
                analyze_numbers,
                convert_units,
            ],
            max_iters=MAX_ITERS,
        )

    def forward(self, question: str):
        result = self.agent(
            question=question
        )

        trajectory = getattr(
            result,
            "trajectory",
            {},
        )

        tool_calls, tool_results, tool_names = (
            extract_tool_history(trajectory)
        )

        return dspy.Prediction(
            answer=str(result.answer),
            tool_calls=json.dumps(
                tool_calls,
                ensure_ascii=False,
                default=str,
            ),
            tool_results=json.dumps(
                tool_results,
                ensure_ascii=False,
                default=str,
            ),
            tool_names=tool_names,
            trajectory=trajectory,
        )


# GEPAで最適化する前のAgent。
agent = MCPToolAgent()


# ------------------------------------------------------------
# データセット
# ------------------------------------------------------------

def make_example(
    question,
    expected_tools,
    reference_answer,
    reference_plan,
    category,
):
    """1問分の入力と正解情報をdspy.Exampleへまとめる。"""

    return dspy.Example(
        question=question,
        user_query=question,
        available_tools=AVAILABLE_TOOLS_TEXT,
        expected_tools=expected_tools,
        reference_answer=reference_answer,
        reference_plan=reference_plan,
        category=category,
    ).with_inputs("question")


# GEPAが改善案を考えるために使う練習問題。
trainset = [
    make_example(
        question=(
            "1.2 km、800 m、1500 mの平均をmで求め、"
            "その値に250 mを足してください。"
        ),
        expected_tools=[
            "convert_units",
            "analyze_numbers",
            "calculate_expression",
        ],
        reference_answer="約1416.67 m",
        reference_plan=(
            "1.2 kmを1200 mへ変換する。"
            "[1200, 800, 1500]の平均1166.666...を求める。"
            "最後に250を足して1416.666... mとする。"
        ),
        category="3ツール: 変換→統計→計算",
    ),
    make_example(
        question=(
            "2.5 kg、750 g、1.25 kgの中央値をgで求め、"
            "そこから100 gを引いてください。"
        ),
        expected_tools=[
            "convert_units",
            "convert_units",
            "analyze_numbers",
            "calculate_expression",
        ],
        reference_answer="1150 g",
        reference_plan=(
            "2.5 kgを2500 g、1.25 kgを1250 gへ変換する。"
            "[2500, 750, 1250]の中央値1250を求め、"
            "100を引いて1150 gとする。"
        ),
        category="4ツール: 変換×2→統計→計算",
    ),
    make_example(
        question=(
            "1.5 kmと900 mの平均をmで求め、"
            "その平均値をkmへ変換してください。"
        ),
        expected_tools=[
            "convert_units",
            "analyze_numbers",
            "convert_units",
        ],
        reference_answer="1.2 km",
        reference_plan=(
            "1.5 kmを1500 mへ変換する。"
            "[1500, 900]の平均1200 mを求め、"
            "1200 mを1.2 kmへ変換する。"
        ),
        category="3ツール: 変換→統計→変換",
    ),
    make_example(
        question=(
            "20分、0.5時間、1800秒の平均を分で求めてください。"
        ),
        expected_tools=[
            "convert_units",
            "convert_units",
            "analyze_numbers",
        ],
        reference_answer="約26.67分",
        reference_plan=(
            "0.5時間を30分、1800秒を30分へ変換する。"
            "[20, 30, 30]の平均26.666...分を求める。"
        ),
        category="3ツール: 変換×2→統計",
    ),
    make_example(
        question=(
            "750 cm、8 m、0.095 kmの平均をmで求め、"
            "その値に1.5 mを足してください。"
        ),
        expected_tools=[
            "convert_units",
            "convert_units",
            "analyze_numbers",
            "calculate_expression",
        ],
        reference_answer="約38.33 m",
        reference_plan=(
            "750 cmを7.5 m、0.095 kmを95 mへ変換する。"
            "[7.5, 8, 95]の平均36.833...を求め、"
            "1.5を足して38.333... mとする。"
        ),
        category="4ツール: 変換×2→統計→計算",
    ),
    make_example(
        question=(
            "1.8 kg、650 g、1.05 kgの中央値をgで求め、"
            "その値を5で割ってください。"
        ),
        expected_tools=[
            "convert_units",
            "convert_units",
            "analyze_numbers",
            "calculate_expression",
        ],
        reference_answer="210 g",
        reference_plan=(
            "1.8 kgを1800 g、1.05 kgを1050 gへ変換する。"
            "[1800, 650, 1050]の中央値1050を求め、"
            "5で割って210 gとする。"
        ),
        category="4ツール: 変換×2→統計→計算",
    ),
    make_example(
        question=(
            "2.4 kmをmへ変換し、375 mを足した後、"
            "3で割ってください。"
        ),
        expected_tools=[
            "convert_units",
            "calculate_expression",
        ],
        reference_answer="925 m",
        reference_plan=(
            "2.4 kmを2400 mへ変換する。"
            "(2400 + 375) / 3を計算し、925 mとする。"
        ),
        category="2ツール: 変換→計算",
    ),
    make_example(
        question=(
            "1.75 kgをgへ変換し、325 gを引いた後、"
            "残りをkgへ変換してください。"
        ),
        expected_tools=[
            "convert_units",
            "calculate_expression",
            "convert_units",
        ],
        reference_answer="1.425 kg",
        reference_plan=(
            "1.75 kgを1750 gへ変換する。"
            "1750 - 325を計算して1425 gを得る。"
            "1425 gを1.425 kgへ変換する。"
        ),
        category="3ツール: 変換→計算→変換",
    ),
]


# GEPAが候補の良し悪しを比較する検証問題。
# trainsetとは別問題にしている。
valset = [
    make_example(
        question=(
            "2.3 km、600 m、1.1 kmの平均をmで求め、"
            "その値に125 mを足してください。"
        ),
        expected_tools=[
            "convert_units",
            "convert_units",
            "analyze_numbers",
            "calculate_expression",
        ],
        reference_answer="約1458.33 m",
        reference_plan=(
            "2.3 kmを2300 m、1.1 kmを1100 mへ変換する。"
            "[2300, 600, 1100]の平均1333.333...を求め、"
            "125を足して1458.333... mとする。"
        ),
        category="4ツール: 変換×2→統計→計算",
    ),
    make_example(
        question=(
            "45分、1.25時間、3600秒の平均を分で求め、"
            "その平均を時間へ変換してください。"
        ),
        expected_tools=[
            "convert_units",
            "convert_units",
            "analyze_numbers",
            "convert_units",
        ],
        reference_answer="1時間",
        reference_plan=(
            "1.25時間を75分、3600秒を60分へ変換する。"
            "[45, 75, 60]の平均60分を求め、"
            "60分を1時間へ変換する。"
        ),
        category="4ツール: 変換×2→統計→変換",
    ),
    make_example(
        question=(
            "500 cm、7 m、900 cmの中央値をcmで求め、"
            "その値に50 cmを足してください。"
        ),
        expected_tools=[
            "convert_units",
            "analyze_numbers",
            "calculate_expression",
        ],
        reference_answer="750 cm",
        reference_plan=(
            "7 mを700 cmへ変換する。"
            "[500, 700, 900]の中央値700を求め、"
            "50を足して750 cmとする。"
        ),
        category="3ツール: 変換→統計→計算",
    ),
    make_example(
        question=(
            "0.75 kg、1200 g、950 gの平均をgで求め、"
            "その平均をkgへ変換してください。"
        ),
        expected_tools=[
            "convert_units",
            "analyze_numbers",
            "convert_units",
        ],
        reference_answer="約0.9667 kg",
        reference_plan=(
            "0.75 kgを750 gへ変換する。"
            "[750, 1200, 950]の平均966.666... gを求め、"
            "kgへ変換して0.966666... kgとする。"
        ),
        category="3ツール: 変換→統計→変換",
    ),
]


# 最後の報告に使う未知問題。
# GEPAのcompileには渡さない。
testset = [
    make_example(
        question=(
            "3.4 km、1250 m、0.8 kmの平均をmで求め、"
            "その値に400 mを足してください。"
        ),
        expected_tools=[
            "convert_units",
            "convert_units",
            "analyze_numbers",
            "calculate_expression",
        ],
        reference_answer="約2216.67 m",
        reference_plan=(
            "3.4 kmを3400 m、0.8 kmを800 mへ変換する。"
            "[3400, 1250, 800]の平均1816.666...を求め、"
            "400を足して2216.666... mとする。"
        ),
        category="4ツール: 変換×2→統計→計算",
    ),
    make_example(
        question=(
            "1.6 kg、900 g、1.1 kgの中央値をgで求め、"
            "そこから75 gを引いてください。"
        ),
        expected_tools=[
            "convert_units",
            "convert_units",
            "analyze_numbers",
            "calculate_expression",
        ],
        reference_answer="1025 g",
        reference_plan=(
            "1.6 kgを1600 g、1.1 kgを1100 gへ変換する。"
            "[1600, 900, 1100]の中央値1100を求め、"
            "75を引いて1025 gとする。"
        ),
        category="4ツール: 変換×2→統計→計算",
    ),
    make_example(
        question=(
            "30分、0.75時間、2700秒の平均を分で求め、"
            "その平均を時間へ変換してください。"
        ),
        expected_tools=[
            "convert_units",
            "convert_units",
            "analyze_numbers",
            "convert_units",
        ],
        reference_answer="約0.6667時間",
        reference_plan=(
            "0.75時間を45分、2700秒を45分へ変換する。"
            "[30, 45, 45]の平均40分を求め、"
            "40分を0.666666...時間へ変換する。"
        ),
        category="4ツール: 変換×2→統計→変換",
    ),
    make_example(
        question=(
            "250 cm、4.5 m、0.006 kmの平均をmで求め、"
            "その値に0.5 mを足してください。"
        ),
        expected_tools=[
            "convert_units",
            "convert_units",
            "analyze_numbers",
            "calculate_expression",
        ],
        reference_answer="約4.8333 m",
        reference_plan=(
            "250 cmを2.5 m、0.006 kmを6 mへ変換する。"
            "[2.5, 4.5, 6]の平均4.333...を求め、"
            "0.5を足して4.833... mとする。"
        ),
        category="4ツール: 変換×2→統計→計算",
    ),
    make_example(
        question=(
            "2.2 kmをmへ変換し、480 mを足した後、"
            "4で割ってください。"
        ),
        expected_tools=[
            "convert_units",
            "calculate_expression",
        ],
        reference_answer="670 m",
        reference_plan=(
            "2.2 kmを2200 mへ変換する。"
            "(2200 + 480) / 4を計算して670 mとする。"
        ),
        category="2ツール: 変換→計算",
    ),
    make_example(
        question=(
            "2.4 kgをgへ変換し、600 gを引いた後、"
            "残りをkgへ変換してください。"
        ),
        expected_tools=[
            "convert_units",
            "calculate_expression",
            "convert_units",
        ],
        reference_answer="1.8 kg",
        reference_plan=(
            "2.4 kgを2400 gへ変換する。"
            "2400 - 600を計算して1800 gを得る。"
            "1800 gを1.8 kgへ変換する。"
        ),
        category="3ツール: 変換→計算→変換",
    ),
    make_example(
        question=(
            "1200 m、1.8 km、950 mの中央値をmで求め、"
            "その中央値をkmへ変換してください。"
        ),
        expected_tools=[
            "convert_units",
            "analyze_numbers",
            "convert_units",
        ],
        reference_answer="1.2 km",
        reference_plan=(
            "1.8 kmを1800 mへ変換する。"
            "[1200, 1800, 950]の中央値1200 mを求め、"
            "1200 mを1.2 kmへ変換する。"
        ),
        category="3ツール: 変換→統計→変換",
    ),
    make_example(
        question=(
            "15分、0.25時間、1200秒の平均を分で求め、"
            "その値に5分を足してください。"
        ),
        expected_tools=[
            "convert_units",
            "convert_units",
            "analyze_numbers",
            "calculate_expression",
        ],
        reference_answer="約21.67分",
        reference_plan=(
            "0.25時間を15分、1200秒を20分へ変換する。"
            "[15, 15, 20]の平均16.666...を求め、"
            "5を足して21.666...分とする。"
        ),
        category="4ツール: 変換×2→統計→計算",
    ),
]


# 1ツール問題で性能が悪化していないか確認する対照問題。
controlset = [
    make_example(
        question="37 * 19を計算してください。",
        expected_tools=["calculate_expression"],
        reference_answer="703",
        reference_plan="37 * 19をcalculate_expressionで計算する。",
        category="対照: 1ツール計算",
    ),
    make_example(
        question="3、7、14の平均を求めてください。",
        expected_tools=["analyze_numbers"],
        reference_answer="8",
        reference_plan="[3, 7, 14]をanalyze_numbersへ渡し、平均8を求める。",
        category="対照: 1ツール統計",
    ),
    make_example(
        question="3.5 kmは何mですか。",
        expected_tools=["convert_units"],
        reference_answer="3500 m",
        reference_plan="3.5 kmをconvert_unitsで3500 mへ変換する。",
        category="対照: 1ツール変換",
    ),
]


# ------------------------------------------------------------
# Judge（採点用LM）
# ------------------------------------------------------------

class ToolUseJudge(dspy.Signature):
    """
    AI Agentのツール利用を、参照計画と参照回答に照らして評価してください。

    評価観点:
    1. 期待されたツールを適切な順序で選んだか
    2. ツールを使う必要がある問題で、必要なツールを使ったか
    3. 各ツールへ渡した引数と、前段の結果の受け渡しが正しいか
    4. 最終回答が参照回答と一致し、単位も適切か
    5. 不要なツール呼び出しや重複がないか

    0.0は不適切、1.0は完全に適切です。
    feedbackには、良かった点と、次に改善すべき行動を具体的に書いてください。
    """

    user_query: str = dspy.InputField(
        desc="ユーザーの質問"
    )
    available_tools: str = dspy.InputField(
        desc="利用可能なツールの名前と説明"
    )
    expected_tools: str = dspy.InputField(
        desc="この実験で期待するツールの順序"
    )
    reference_plan: str = dspy.InputField(
        desc="期待する処理手順"
    )
    reference_answer: str = dspy.InputField(
        desc="期待する最終回答"
    )
    tool_calls: str = dspy.InputField(
        desc="実際のツール名、引数、実行順序"
    )
    tool_results: str = dspy.InputField(
        desc="実際のツール実行結果"
    )
    final_answer: str = dspy.InputField(
        desc="Agentが生成した最終回答"
    )

    tool_selection_score: float = dspy.OutputField(
        desc="ツール選択と順序の妥当性。0.0から1.0"
    )
    tool_necessity_score: float = dspy.OutputField(
        desc="ツール利用の必要性判断。0.0から1.0"
    )
    argument_score: float = dspy.OutputField(
        desc="引数と結果受け渡しの妥当性。0.0から1.0"
    )
    task_success_score: float = dspy.OutputField(
        desc="最終回答の正確さと要求達成度。0.0から1.0"
    )
    feedback: str = dspy.OutputField(
        desc="Agentの指示を改善するための具体的な日本語の助言"
    )


# まずは、すでに動いているAgent用deploymentをJudgeにも使う。
# メンターから別のJudge用deployment名を指定された場合だけ変更する。
JUDGE_DEPLOYMENT = AZURE_OPENAI_DEPLOYMENT

judge_lm = dspy.LM(
    f"azure/{JUDGE_DEPLOYMENT}",
    api_key=AZURE_OPENAI_API_KEY,
    api_base=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION,
    temperature=0.0,
)

judge = dspy.ChainOfThought(ToolUseJudge)


def clamp_score(value):
    """値をfloatへ変換し、0.0から1.0の範囲へ収める。"""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0

    return max(0.0, min(1.0, number))


def sequence_position_score(expected_tools, actual_tools):
    """
    期待順序と実際の順序を同じ位置どうしで比較し、0.0〜1.0で返す。

    例:
    期待 [convert, analyze, calculate]
    実際 [convert, calculate]
    なら、先頭の1個だけ一致するので 1 / 3。
    """

    if not expected_tools and not actual_tools:
        return 1.0

    denominator = max(
        len(expected_tools),
        len(actual_tools),
        1,
    )

    matched = 0

    for expected, actual in zip(
        expected_tools,
        actual_tools,
    ):
        if expected == actual:
            matched += 1

    return matched / denominator


def make_sequence_feedback(expected_tools, actual_tools):
    """期待順序と実際の順序をGEPAが読みやすい文章へ変換する。"""

    expected_text = " → ".join(expected_tools) or "ツールなし"
    actual_text = " → ".join(actual_tools) or "ツールなし"

    if expected_tools == actual_tools:
        return (
            "ツール順序は参照計画と一致しました。"
            f" 順序: {actual_text}"
        )

    return (
        "ツール順序が参照計画と一致していません。"
        f" 期待: {expected_text}。"
        f" 実際: {actual_text}。"
        "前段のツール結果を確認してから、"
        "次に必要なツールへその値を渡す指示を強めてください。"
    )


def run_judge(example, prediction):
    """1問の正解情報とAgentの実行結果をJudge LMへ渡す。"""

    expected_tools_text = " → ".join(
        example.expected_tools
    )

    with dspy.context(lm=judge_lm):
        return judge(
            user_query=example.user_query,
            available_tools=example.available_tools,
            expected_tools=expected_tools_text,
            reference_plan=example.reference_plan,
            reference_answer=example.reference_answer,
            tool_calls=getattr(
                prediction,
                "tool_calls",
                "[]",
            ),
            tool_results=getattr(
                prediction,
                "tool_results",
                "[]",
            ),
            final_answer=str(
                getattr(prediction, "answer", "")
            ),
        )


def score_prediction(example, prediction):
    """
    Judgeの評価と客観的な順序比較をまとめ、総合点を作る。

    メンターの大分類の重み:
    - ツール選択 40%
    - ツール必要性 20%
    - 引数 15%
    - タスク成功 25%
    """

    actual_tools = list(
        getattr(prediction, "tool_names", [])
    )
    expected_tools = list(example.expected_tools)

    judge_result = run_judge(
        example,
        prediction,
    )

    objective_sequence = sequence_position_score(
        expected_tools,
        actual_tools,
    )

    judge_selection = clamp_score(
        judge_result.tool_selection_score
    )

    # ツール選択は、機械的な順序比較とJudge判断を半分ずつ使う。
    tool_selection = (
        0.50 * objective_sequence
        + 0.50 * judge_selection
    )

    tool_necessity = clamp_score(
        judge_result.tool_necessity_score
    )
    argument = clamp_score(
        judge_result.argument_score
    )
    task_success = clamp_score(
        judge_result.task_success_score
    )

    total_score = (
        0.40 * tool_selection
        + 0.20 * tool_necessity
        + 0.15 * argument
        + 0.25 * task_success
    )

    sequence_exact = (
        expected_tools == actual_tools
    )

    feedback = (
        make_sequence_feedback(
            expected_tools,
            actual_tools,
        )
        + "\nJudgeの助言: "
        + str(judge_result.feedback)
    )

    return {
        "score": clamp_score(total_score),
        "tool_selection": tool_selection,
        "tool_necessity": tool_necessity,
        "argument": argument,
        "task_success": task_success,
        "sequence_score": objective_sequence,
        "sequence_exact": sequence_exact,
        "expected_tools": expected_tools,
        "actual_tools": actual_tools,
        "feedback": feedback,
    }


def tool_use_metric(
    example,
    prediction,
    trace=None,
    pred_name=None,
    pred_trace=None,
    program_trace=None,
):
    """GEPAへ総合scoreと文章feedbackを返す。"""

    details = score_prediction(
        example,
        prediction,
    )

    return dspy.Prediction(
        score=details["score"],
        feedback=details["feedback"],
    )


# ------------------------------------------------------------
# 評価・CSV保存
# ------------------------------------------------------------

def evaluate_program(
    program,
    dataset,
    label,
    csv_path,
):
    """複数問題を実行し、平均点とツール順序一致率を返す。"""

    rows = []

    print("\n" + "=" * 78)
    print(label)
    print("=" * 78)

    for number, example in enumerate(
        dataset,
        start=1,
    ):
        prediction = program(
            question=example.question
        )

        details = score_prediction(
            example,
            prediction,
        )

        row = {
            "number": number,
            "category": example.category,
            "question": example.question,
            "reference_answer": example.reference_answer,
            "agent_answer": prediction.answer,
            "expected_tools": " -> ".join(
                details["expected_tools"]
            ),
            "actual_tools": " -> ".join(
                details["actual_tools"]
            ),
            "sequence_exact": details["sequence_exact"],
            "sequence_score": details["sequence_score"],
            "tool_selection": details["tool_selection"],
            "tool_necessity": details["tool_necessity"],
            "argument": details["argument"],
            "task_success": details["task_success"],
            "total_score": details["score"],
            "feedback": details["feedback"],
        }

        rows.append(row)

        print(f"\n[{number}] {example.category}")
        print("質問:", example.question)
        print("期待順序:", row["expected_tools"])
        print("実際順序:", row["actual_tools"] or "ツールなし")
        print("回答:", prediction.answer)
        print("総合点:", f"{details['score'] * 100:.1f}")
        print(
            "順序完全一致:",
            "OK" if details["sequence_exact"] else "NG",
        )

    if not rows:
        raise ValueError("datasetが空です。")

    summary = {
        "label": label,
        "problem_count": len(rows),
        "average_score": 100 * mean(
            row["total_score"]
            for row in rows
        ),
        "exact_sequence_rate": 100 * mean(
            1.0 if row["sequence_exact"] else 0.0
            for row in rows
        ),
        "task_success_rate": 100 * mean(
            row["task_success"]
            for row in rows
        ),
    }

    fieldnames = list(rows[0].keys())

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\n--- 集計 ---")
    print("問題数:", summary["problem_count"])
    print(
        "平均総合点:",
        f"{summary['average_score']:.1f}",
    )
    print(
        "ツール順序完全一致率:",
        f"{summary['exact_sequence_rate']:.1f}%",
    )
    print(
        "タスク達成度の平均:",
        f"{summary['task_success_rate']:.1f}",
    )
    print("CSV保存先:", csv_path)

    return summary, rows


def compare_summaries(before, after):
    """最適化前後の3指標を見やすく表示する。"""

    print("\n" + "=" * 78)
    print("最適化前後の比較")
    print("=" * 78)

    items = [
        ("平均総合点", "average_score", "点"),
        ("ツール順序完全一致率", "exact_sequence_rate", "%"),
        ("タスク達成度の平均", "task_success_rate", "点"),
    ]

    for label, key, unit in items:
        before_value = before[key]
        after_value = after[key]
        difference = after_value - before_value

        print(
            f"{label}: "
            f"{before_value:.1f}{unit} -> "
            f"{after_value:.1f}{unit} "
            f"(差 {difference:+.1f}{unit})"
        )


def compare_each_problem(
    before_rows,
    after_rows,
    csv_path,
):
    """問題ごとの改善幅を計算し、大きい順にCSVへ保存する。"""

    comparison_rows = []

    for before_row, after_row in zip(
        before_rows,
        after_rows,
    ):
        comparison_rows.append(
            {
                "number": before_row["number"],
                "category": before_row["category"],
                "question": before_row["question"],
                "before_score": 100 * before_row["total_score"],
                "after_score": 100 * after_row["total_score"],
                "score_difference": 100 * (
                    after_row["total_score"]
                    - before_row["total_score"]
                ),
                "expected_tools": before_row["expected_tools"],
                "before_tools": before_row["actual_tools"],
                "after_tools": after_row["actual_tools"],
                "before_answer": before_row["agent_answer"],
                "after_answer": after_row["agent_answer"],
            }
        )

    comparison_rows.sort(
        key=lambda row: row["score_difference"],
        reverse=True,
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(comparison_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(comparison_rows)

    print("\n改善幅が大きい問題:")

    for row in comparison_rows[:3]:
        print("-", row["question"])
        print(
            "  score:",
            f"{row['before_score']:.1f}",
            "->",
            f"{row['after_score']:.1f}",
        )
        print("  before:", row["before_tools"] or "ツールなし")
        print("  after :", row["after_tools"] or "ツールなし")

    print("比較CSV保存先:", csv_path)

    return comparison_rows


# ------------------------------------------------------------
# 実行例
# ------------------------------------------------------------
# 以下はJupyterでは、区切りごとに別セルへコピーして実行する。

# 1問だけAgentを動かす。
example = trainset[0]
prediction = agent(
    question=example.question
)

print("質問:", example.question)
print("回答:", prediction.answer)
print("期待順序:", example.expected_tools)
print("実際順序:", prediction.tool_names)
print("tool_calls:", prediction.tool_calls)
print("tool_results:", prediction.tool_results)


# 1問だけJudgeで採点する。
details = score_prediction(
    example,
    prediction,
)

print("\n1問の採点結果:")
print("総合点:", details["score"] * 100)
print("順序スコア:", details["sequence_score"])
print("feedback:", details["feedback"])


# 最適化前の未知問題と1ツール対照問題を評価する。
baseline_test_summary, baseline_test_rows = evaluate_program(
    program=agent,
    dataset=testset,
    label="最適化前: 複雑な未知問題",
    csv_path="baseline_test_results.csv",
)

baseline_control_summary, baseline_control_rows = evaluate_program(
    program=agent,
    dataset=controlset,
    label="最適化前: 1ツール対照問題",
    csv_path="baseline_control_results.csv",
)


# Reflection LMはJudgeと同じdeploymentを使う。
reflection_lm = dspy.LM(
    f"azure/{JUDGE_DEPLOYMENT}",
    api_key=AZURE_OPENAI_API_KEY,
    api_base=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION,
    temperature=0.0,
)


# autoとmax_metric_callsは同時に指定しない。
optimizer = dspy.GEPA(
    metric=tool_use_metric,
    max_metric_calls=80,
    reflection_lm=reflection_lm,
    num_threads=1,
    seed=0,
)


# GEPAへ渡すのはtrainsetとvalsetだけ。
# testsetは最後の評価まで見せない。
optimized_agent = optimizer.compile(
    student=agent,
    trainset=trainset,
    valset=valset,
)


# 最適化後を、同じ未知問題・同じ採点方法で評価する。
optimized_test_summary, optimized_test_rows = evaluate_program(
    program=optimized_agent,
    dataset=testset,
    label="GEPA後: 複雑な未知問題",
    csv_path="optimized_test_results.csv",
)

optimized_control_summary, optimized_control_rows = evaluate_program(
    program=optimized_agent,
    dataset=controlset,
    label="GEPA後: 1ツール対照問題",
    csv_path="optimized_control_results.csv",
)


# 集計値と各問題の変化を表示・保存する。
compare_summaries(
    baseline_test_summary,
    optimized_test_summary,
)

comparison_rows = compare_each_problem(
    baseline_test_rows,
    optimized_test_rows,
    csv_path="before_after_comparison.csv",
)


# 1ツール性能が悪化していないかも確認する。
print("\n1ツール対照問題の比較:")
compare_summaries(
    baseline_control_summary,
    optimized_control_summary,
)


# GEPAが変更した可能性のある指示を確認する。
print("\n最適化後のpredictor:")

for name, predictor in optimized_agent.named_predictors():
    print("\npredictor名:", name)

    signature = getattr(
        predictor,
        "signature",
        None,
    )

    if signature is not None:
        print("instructions:")
        print(signature.instructions)
