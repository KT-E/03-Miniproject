"""Agent 1: the review-analysis multi-agent (Analyzer / Critic / Supervisor).

This is a LangGraph implementation of the "Step2" workflow: given a single
product review, extract which product aspects it discusses and whether the
sentiment for each aspect is positive (1) or negative (0), together with the
review's star rating. A Critic node checks the Analyzer's output against the
review text and a Supervisor node decides whether to accept the result or
send it back for another pass (up to ``max_num`` retries).

Ported from the final team notebook (notebooks/agent_v2_final.ipynb, the
Streamlit `app.py` cell) into a standalone, importable module. Logic and
prompts are unchanged from that notebook; only the surrounding plumbing
(imports, API-key loading) was adapted for local/script use instead of
Google Colab.
"""

from __future__ import annotations

import json
import re
from typing import Annotated, List, Literal, Optional, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from . import config

ANALYZER_SYSTEM_PROMPT = """
너는 화장품 리뷰 분석 전문가야.
다음 지침에 따라 입력받은 사용자의 리뷰를 분석해줘.
만약 [수정 요청 피드백]이 제공된다면, 이전의 실수를 교정하여 결과를 다시 작성해야 해.

[지침]
1. 제시된 [후보 속성] 중에서 리뷰 내용과 일치하는 것을 추출해.
   단, '성능', '효과'처럼 포괄적인 단어가 쓰였더라도 문맥상 화장품의 기능(예: 보습력)을 뜻한다면
   유연하게 해석해서 해당 후보 속성으로 매칭해줘.
2. 중요: 'aspect' 리스트에 담긴 속성의 순서는 반드시 [후보 속성]에 나열된 순서를 유지해야 해.
3. 각 속성에 대한 만족도를 판별해 (만족/긍정은 1, 불만족/부정은 0).
4. 'label' 리스트의 값은 'aspect' 리스트의 순서와 1:1로 정확히 매칭되어야 해.
5. 리뷰에 명시된 별점(score)을 추출해.
6. 리뷰에서 후보 속성과 매칭되는 내용이 전혀 없으면 aspect와 label은 빈 리스트([])로 반환해.
7. 결과는 반드시 아래의 JSON 형식으로만 출력하고, 다른 설명이나 코드펜스(```)는 절대 붙이지 마.

[출력 형식]
{
  "review": "리뷰 원문",
  "aspect": ["추출된 속성 리스트"],
  "label": [각 속성에 대응하는 1 또는 0 리스트],
  "score": 별점숫자
}
""".strip()

CRITIC_SYSTEM_PROMPT = """
너는 리뷰 분석 결과의 정확성을 검증하는 품질 관리(QA) 전문가야.
Analyzer가 추출한 결과가 리뷰 원문의 내용과 논리적으로 일치하는지 검토해줘.

[검토 지침]
1. 일관성: 추출된 'aspect'가 리뷰 원문에 직접 언급되었거나,
   문맥상 의미가 일치하는지 확인해 (예: '성능' -> '보습' 등 포괄적/유의어 매칭 허용).
2. 정확성: 각 'aspect'에 매칭된 'label'(1: 만족, 0: 불만족)이 리뷰의 맥락과 맞는지 확인해.
3. 형식: 결과가 약속된 JSON 형식을 유지하고 있는지 확인해.
4. 순서: [중요] 추출된 'aspect'들은 [후보 속성]에 나열된 상대적 순서를 유지해야 해.
   예: 후보가 [A, B, C, D]이고 추출이 [A, C]라면 순서가 맞는 것으로 간주함.

[판단 기준]
- 분석 결과에 논리적/형식적 오류가 없는지
- 문맥상 유추할 수 없는 완전히 엉뚱한 속성이 포함되어 있지는 않은지
- 존재하는 속성들끼리의 순서가 후보 리스트의 우선순위와 일치하는지

[출력]
[FEEDBACK]
- ...
[VERDICT]
VERDICT: OK
또는
VERDICT: REVISE
""".strip()


class ReviewState(TypedDict):
    messages: Annotated[list, add_messages]
    input_review: str
    max_num: int
    current_num: int
    score: int
    next_step: Optional[Literal["analyzer", "critic", "end"]]
    result: dict
    aspect_list: List[str]
    criti_score: str


def analyzer_node(state: ReviewState, analyzer_llm) -> dict:
    """Extract (aspect, sentiment-label) pairs and the star rating from the review."""
    review = state.get("input_review") or ""
    aspect_list = state.get("aspect_list") or []
    score = state.get("score") or 0
    feedback = state.get("criti_score") or "없음 (최초 분석)"

    human = (
        f"[리뷰 원문]\n{review}\n\n"
        f"[후보 속성]\n{aspect_list}\n\n"
        f"[별점 숫자]\n{score}\n\n"
        f"[수정 요청 피드백]\n{feedback}"
    )

    resp = analyzer_llm.invoke([
        SystemMessage(content=ANALYZER_SYSTEM_PROMPT),
        HumanMessage(content=human),
    ])

    raw = resp.content.strip()
    try:
        data_dict = json.loads(raw)
    except json.JSONDecodeError:
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
        try:
            data_dict = json.loads(cleaned)
        except json.JSONDecodeError:
            data_dict = {"review": review, "aspect": [], "label": [], "score": score}

    data_dict.setdefault("review", review)
    data_dict.setdefault("aspect", [])
    data_dict.setdefault("label", [])
    data_dict.setdefault("score", score)

    return {
        "messages": [AIMessage(content=f"[Analyzer Result]\n{resp.content}")],
        "result": data_dict,
        "criti_score": "",
    }


def critic_node(state: ReviewState, criti_llm) -> dict:
    """Check the Analyzer's output for consistency with the review text."""
    review = state.get("input_review") or ""
    aspect_list = state.get("aspect_list") or []
    result = state.get("result") or ""
    current_num = state.get("current_num") or 0

    human = (
        f"[리뷰 원문]\n{review}\n\n"
        f"[후보 속성]\n{aspect_list}\n\n"
        f"[Analyzer의 분석 결과]\n{result}"
    )

    resp = criti_llm.invoke([
        SystemMessage(content=CRITIC_SYSTEM_PROMPT),
        HumanMessage(content=human),
    ])

    return {
        "messages": [AIMessage(content=f"[CRITIC RESULT]\n{resp.content}")],
        "criti_score": resp.content,
        "current_num": current_num + 1,
    }


def supervisor_node(state: ReviewState) -> dict:
    """Decide whether to run Analyzer, Critic, or stop."""
    if not state.get("result"):
        next_step = "analyzer"
    elif not state.get("current_num"):
        next_step = "critic"
    else:
        criti = state.get("criti_score") or ""
        if "VERDICT: OK" in criti:
            next_step = "end"
        elif "VERDICT: REVISE" in criti and state.get("current_num") <= state.get("max_num"):
            next_step = "analyzer"
        else:
            next_step = "end"

    return {"next_step": next_step}


def route_next(state: ReviewState) -> str:
    return state["next_step"]


def build_review_graph(analyzer_temperature: float = 0.2, critic_temperature: float = 0.2):
    """Compile the Analyzer -> Critic -> Supervisor LangGraph app."""
    analyzer_llm = config.get_llm(temperature=analyzer_temperature)
    criti_llm = config.get_llm(temperature=critic_temperature)

    builder = StateGraph(ReviewState)
    builder.add_node("analyzer", lambda s: analyzer_node(s, analyzer_llm))
    builder.add_node("critic", lambda s: critic_node(s, criti_llm))
    builder.add_node("supervisor", supervisor_node)

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        route_next,
        {"analyzer": "analyzer", "critic": "critic", "end": END},
    )
    builder.add_edge("analyzer", "supervisor")
    builder.add_edge("critic", "supervisor")

    return builder.compile()


def analyze_review(app, review: str, score: int = 0, max_num: int = 3, aspect_list: Optional[List[str]] = None) -> dict:
    """Convenience wrapper: run one review through the compiled graph and
    return the Analyzer's final structured result."""
    initial_state: ReviewState = {
        "input_review": review,
        "score": score,
        "aspect_list": aspect_list or config.ASPECT_LIST,
        "max_num": max_num,
        "current_num": 0,
        "messages": [],
        "result": {},
        "criti_score": "",
        "next_step": None,
    }
    final_state = app.invoke(initial_state)
    return final_state["result"]
