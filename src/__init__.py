"""Multi-agent product review sentiment analysis system.

- ``agent1_review``: Analyzer / Critic / Supervisor loop that turns one raw
  review into structured aspect/sentiment/rating data.
- ``agent2_insight``: Router + 5 domain-specialist agents + Aggregator that
  turns negative aspects into improvement suggestions.
- ``db``: SQLite persistence for analyzed reviews.
- ``main``: CLI entry point.
- ``dashboard``: Streamlit UI (run with ``streamlit run src/dashboard.py``).
"""
