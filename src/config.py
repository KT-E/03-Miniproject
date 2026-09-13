"""Shared configuration for the review-analysis multi-agent system.

Originally the notebooks loaded the OpenAI key from a Google-Drive-hosted
``api_key.txt`` (a Colab convention). For a standalone script we simply read
standard environment variables instead, so this can run anywhere:

    export OPENAI_API_KEY="sk-..."
    # optional, only needed if you want LangSmith tracing (see Step2/Mission3)
    export LANGSMITH_TRACING="true"
    export LANGSMITH_PROJECT="proj2_agent"
    export LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
    export LANGSMITH_API_KEY="ls-..."
"""

from __future__ import annotations

import os

from langchain_openai import ChatOpenAI

# The model used for every agent in the notebooks was gpt-4.1-mini.
DEFAULT_MODEL = "gpt-4.1-mini"

# Full candidate aspect list used by the final (AI_12조) notebook's Analyzer.
# The original dataset (data/reviews.csv) only carries ground-truth labels
# for the first four of these (보습/가격/향/포장) - the rest were added by
# the team as the system was generalized beyond the original labeled set.
ASPECT_LIST = [
    "가격", "구성", "제형 밀도", "향", "사용감", "지속력", "디자인", "만족도", "보습", "휴대성",
    "배송", "발림성", "용량", "사은품", "포장", "커버력", "피부표현", "세팅력", "발색", "소독효과",
    "흡수력", "유통기한", "유통 기한", "사이즈", "편의성", "품질", "색감", "제형", "피부타입",
    "유분감", "유효기간", "자외선 차단", "톤업", "진정", "크기", "피부결", "세정력", "번짐",
    "밀착력", "색상", "위생", "손상케어", "주름개선", "자극", "클렌징", "효과", "할인",
    "재구매", "성분", "피부",
]

DB_PATH = os.environ.get("REVIEW_DB_PATH", "user_review_db.db")


def require_api_key() -> None:
    """Fail fast with a clear message if OPENAI_API_KEY is missing."""
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Export it before running, e.g.\n"
            '  export OPENAI_API_KEY="sk-..."'
        )


def get_llm(temperature: float = 0.2, model: str = DEFAULT_MODEL) -> ChatOpenAI:
    """Build a ChatOpenAI client. Each agent in the notebooks used its own
    instance (same model, different temperature) so nodes can be tuned
    independently."""
    require_api_key()
    return ChatOpenAI(model=model, temperature=temperature)
