"""Agent 2: the improvement-suggestion multi-agent (Router + 5 specialists + Aggregator).

Once Agent 1 (see ``agent1_review.py``) has labeled a review's aspects as
positive/negative, this second graph takes the negative aspects and routes
them to the specialist agent(s) responsible for that area of the product, so
each can propose a concrete improvement from their own point of view. A
final Aggregator node merges every specialist's suggestion into one report.

Routing is fan-out: a single review can be sent to multiple specialists in
parallel via LangGraph's ``Send`` API (e.g. a review complaining about both
price and packaging goes to both price_node and design_node).

Ported from the final team notebook (notebooks/agent_v2_final.ipynb).
"""

from __future__ import annotations

import json
from typing import Annotated, List, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Send

from . import config

ROUTER_SYSTEM_PROMPT = """
너는 고객 리뷰의 불만 사항을 파악하고 해결할 전문 부서를 배정하는 라우터(Router) 에이전트야.
입력된 [aspect] 리스트와 [label] 리스트의 규칙을 분석하여,
불만족으로 평가된 속성을 해결할 전문가 에이전트 목록을 출력해.

[전문가 그룹 및 담당 속성]
1. price_node:       가격, 할인, 가성비, 구성
2. performance_node: 보습, 커버력, 발색, 자외선 차단, 톤업, 진정, 세정력, 손상케어, 주름개선,
                     클렌징, 효과, 피부결, 피부, 품질, 성분, 소독효과
3. texture_node:     발림성, 흡수력, 사용감, 제형 밀도, 제형, 유분감, 밀착력, 번짐, 자극, 향
4. design_node:      디자인, 색감(색상), 사이즈(크기), 휴대성, 편의성, 위생, 포장
5. service_node:     배송, 만족도, 재구매, 피부타입, 사은품

[분석 및 라우팅 지침]
1. [aspect]와 [label]의 요소는 같은 인덱스로 1:1 매칭.
2. [label] 값이 0인 항목이 불만족.
3. 불만족 속성이 속한 전문가 노드 이름을 매칭.
4. 중복 없이 고유한 값만 출력.
5. performance_node나 texture_node가 포함되면 맨 앞에 정렬.
6. 불만족이 없으면 ["service_node"] 출력.

[출력 형식]
반드시 쌍따옴표가 포함된 올바른 JSON 리스트 형식으로만 응답해. 부가 설명 금지.
예) ["performance_node", "texture_node", "design_node"]
""".strip()

PERFORMANCE_SYSTEM_PROMPT = """
너는 제품의 '효능, 성분, 품질' 개선을 담당하는 제품 개발 전문가야.
[담당 속성] 중 불만족(0) 항목에 대한 원인 분석과 개선 방안을 제시해.

[담당 속성]
보습, 커버력, 발색, 자외선 차단, 톤업, 진정, 세정력, 손상케어, 주름개선,
클렌징, 효과, 피부결, 피부, 품질, 성분, 소독효과

[지침]
1. label=0인 항목 중 담당 속성만 필터링.
2. 리뷰 원문에서 구체적 불만 내용 파악.
3. 화장품 연구원 관점의 실질적 개선 방안 도출.
4. 담당 외 속성 언급 금지.

[출력 형식] — JSON·마크다운 금지, 인사말 금지

[개선 대상 속성]: 속성1, 속성2, ...

- 속성: 속성 이름
- 원인 분석: 리뷰 원문 기반 불만 원인 요약
- 개선 방안: 성분 변경, 배합 비율 조정 등 전문적인 개선 방안
""".strip()

PRICE_SYSTEM_PROMPT = """
너는 제품의 '가격 경쟁력 및 경제적 가치' 개선을 담당하는 가격 전략 전문가야.
[담당 속성] 중 불만족(0) 항목에 대한 원인 분석과 개선 방안을 제시해.

[담당 속성]
가격, 할인, 가성비, 구성

[지침]
1. label=0인 항목 중 담당 속성만 필터링.
2. 리뷰 원문에서 구체적 불만 내용 파악.
3. 마케팅 및 영업 기획자 관점의 실질적 개선 방안 도출.
4. 담당 외 속성 언급 금지.

[출력 형식] — JSON·마크다운 금지, 인사말 금지

[개선 대상 속성]: 속성1, 속성2, ...

- 속성: 속성 이름
- 원인 분석: 리뷰 원문 기반 불만 원인 요약
- 개선 방안: 프로모션 기획, 용량 리뉴얼, 결합 상품 구성 변경 등 마케팅 관점의 해결 방안
""".strip()

TEXTURE_SYSTEM_PROMPT = """
너는 제품의 '제형, 사용감, 피부 자극 및 향기' 개선을 담당하는 제형 개발 전문가야.
[담당 속성] 중 불만족(0) 항목에 대한 원인 분석과 개선 방안을 제시해.

[담당 속성]
발림성, 흡수력, 사용감, 제형 밀도, 제형, 유분감, 밀착력, 번짐, 자극, 향

[지침]
1. label=0인 항목 중 담당 속성만 필터링.
2. 리뷰 원문에서 구체적 불만 내용 파악.
3. 제형 연구원·향료 전문가 관점의 실질적 개선 방안 도출.
4. 담당 외 속성 언급 금지.

[출력 형식] — JSON·마크다운 금지, 인사말 금지

[개선 대상 속성]: 속성1, 속성2, ...

- 속성: 속성 이름
- 원인 분석: 리뷰 원문 기반 불만 원인 요약
- 개선 방안: 점증제 변경, 알러지 유발 성분 배제, 향료 배합 변경 등 전문적인 개선 방안
""".strip()

DESIGN_SYSTEM_PROMPT = """
너는 제품의 '외형 디자인, 패키징 및 사용 편의성' 개선을 담당하는 제품 디자이너야.
[담당 속성] 중 불만족(0) 항목에 대한 원인 분석과 개선 방안을 제시해.

[담당 속성]
디자인, 색감(색상), 사이즈(크기), 휴대성, 편의성, 위생, 포장

[지침]
1. label=0인 항목 중 담당 속성만 필터링.
2. 리뷰 원문에서 구체적 불만 내용 파악.
3. 제품 디자이너·패키징 전문가 관점의 실질적 개선 방안 도출.
4. 담당 외 속성 언급 금지.

[출력 형식] — JSON·마크다운 금지, 인사말 금지

[개선 대상 속성]: 속성1, 속성2, ...

- 속성: 속성 이름
- 원인 분석: 리뷰 원문 기반 불만 원인 요약
- 개선 방안: 용기 구조 변경, 색상 라인업 수정, 패키지 내구성 보강 등 디자이너 관점의 해결 방안
""".strip()

SERVICE_SYSTEM_PROMPT = """
너는 제품의 '배송 서비스 및 고객 경험 관리'를 담당하는 CS 전략 전문가야.
[담당 속성] 중 불만족(0) 항목에 대한 원인 분석과 개선 방안을 제시해.

[담당 속성]
배송, 만족도, 재구매, 피부타입, 사은품

[지침]
1. label=0인 항목 중 담당 속성만 필터링.
2. 리뷰 원문에서 구체적 불만 내용 파악.
3. CS·브랜드 매니저 관점의 실질적 개선 방안 도출.
4. 담당 외 속성 언급 금지.

[출력 형식] — JSON·마크다운 금지, 인사말 금지

[개선 대상 속성]: 속성1, 속성2, ...

- 속성: 속성 이름
- 원인 분석: 리뷰 원문 기반 불만 원인 요약
- 개선 방안: 배송 프로세스 점검, 피부타입별 가이드라인 세분화, 보상 정책 수립 등 CS 관점의 해결 방안
""".strip()

VALID_NODES = {"performance_node", "price_node", "texture_node", "design_node", "service_node"}


class InsightState(TypedDict):
    messages: Annotated[list, add_messages]
    review: str
    aspect: List[str]
    label: List[int]
    next_node: List[str]
    result_list: List[str]
    final_result: str


def _build_expert_user_msg(review: str, aspect: list, label: list) -> str:
    return f"[review]\n{review}\n\n[aspect]\n{aspect}\n\n[label]\n{label}"


def _expert_node(state: InsightState, llm: ChatOpenAI, system_prompt: str) -> dict:
    review = state.get("review") or ""
    aspect = state.get("aspect") or []
    label = state.get("label") or []
    results = state.get("result_list") or []

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=_build_expert_user_msg(review, aspect, label)),
    ])

    results.append(response.content)
    return {"messages": [response], "result_list": results}


def router_node(state: InsightState, router_llm: ChatOpenAI) -> dict:
    """Decide which specialist(s) should handle this review's negative aspects."""
    aspect = state.get("aspect") or []
    label = state.get("label") or []

    user_msg = f"[aspect]\n{aspect}\n\n[label]\n{label}"

    response = router_llm.invoke([
        SystemMessage(content=ROUTER_SYSTEM_PROMPT),
        HumanMessage(content=user_msg),
    ])

    content = response.content.strip()
    if content.startswith("```"):
        content = content.replace("```json", "").replace("```", "").strip()

    try:
        destination = json.loads(content)
    except json.JSONDecodeError:
        destination = ["service_node"]

    return {"next_node": destination}


def route_to_experts(state: InsightState) -> list:
    """Fan out to every specialist node the router selected."""
    next_nodes = state.get("next_node") or []

    payload = {
        "review": state.get("review"),
        "aspect": state.get("aspect"),
        "label": state.get("label"),
    }

    sends = []
    for node_name in next_nodes:
        target = node_name if node_name in VALID_NODES else "service_node"
        sends.append(Send(target, payload))

    if not sends:
        sends.append(Send("service_node", payload))

    return sends


def aggregator_node(state: InsightState) -> dict:
    """Merge every specialist's suggestion into a single improvement report."""
    results = state.get("result_list") or []

    if not results:
        final_answer = "분석 결과가 없습니다. 모든 항목에 만족하셨거나 처리 중 오류가 발생했습니다."
    else:
        combined = "\n\n".join(results)
        final_answer = (
            "[종합 제품 개선 제안서]\n\n"
            "고객님의 리뷰를 바탕으로 각 분야 전문가들이 분석한 개선 방안입니다.\n\n"
            f"{combined}\n\n"
            "---\n위 개선 방안이 제품 품질 향상에 도움이 되기를 바랍니다."
        )

    return {"final_result": final_answer}


def build_insight_graph():
    """Compile the Router -> {specialists} -> Aggregator LangGraph app."""
    router_llm = config.get_llm(temperature=0.0)
    price_llm = config.get_llm(temperature=0.7)
    performance_llm = config.get_llm(temperature=0.3)
    texture_llm = config.get_llm(temperature=0.2)
    design_llm = config.get_llm(temperature=0.7)
    service_llm = config.get_llm(temperature=1.0)

    builder = StateGraph(InsightState)

    builder.add_node("router_node", lambda s: router_node(s, router_llm))
    builder.add_node("performance_node", lambda s: _expert_node(s, performance_llm, PERFORMANCE_SYSTEM_PROMPT))
    builder.add_node("price_node", lambda s: _expert_node(s, price_llm, PRICE_SYSTEM_PROMPT))
    builder.add_node("texture_node", lambda s: _expert_node(s, texture_llm, TEXTURE_SYSTEM_PROMPT))
    builder.add_node("design_node", lambda s: _expert_node(s, design_llm, DESIGN_SYSTEM_PROMPT))
    builder.add_node("service_node", lambda s: _expert_node(s, service_llm, SERVICE_SYSTEM_PROMPT))
    builder.add_node("aggregator_node", aggregator_node)

    builder.add_edge(START, "router_node")
    builder.add_conditional_edges("router_node", route_to_experts, list(VALID_NODES))
    for expert in VALID_NODES:
        builder.add_edge(expert, "aggregator_node")
    builder.add_edge("aggregator_node", END)

    return builder.compile()


def suggest_improvements(app, review: str, aspects: List[str], labels: List[int]) -> str:
    """Convenience wrapper: run the negative aspects of one review through the
    insight graph and return the combined improvement report."""
    state: InsightState = {
        "review": review,
        "aspect": aspects,
        "label": labels,
        "next_node": [],
        "result_list": [],
        "final_result": "",
        "messages": [],
    }
    final_state = app.invoke(state)
    return final_state["final_result"]
