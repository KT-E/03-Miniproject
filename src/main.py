"""Command-line entry point that ties the two agents and the DB together.

Usage
-----
    export OPENAI_API_KEY="sk-..."

    # analyze one ad-hoc review and print the structured result
    python -m src.main demo --review "보습력이 정말 좋아요. 향도 괜찮지만 가격은 조금 비싸요." --score 4

    # batch-analyze every review in data/reviews.csv into a local SQLite DB
    python -m src.main batch --csv data/reviews.csv --concurrency 6

    # like `batch`, but also runs Agent 2 on the negative aspects of the
    # first few analyzed reviews and prints improvement suggestions
    python -m src.main batch --csv data/reviews.csv --with-insights --limit 5

This mirrors what the notebooks did across three separate, order-dependent
Colab cells (single-review test -> DB batch fill-in -> Streamlit dashboard);
here it is a single, repeatable script.
"""

from __future__ import annotations

import argparse
import asyncio
import time

import pandas as pd

from . import config, db
from .agent1_review import analyze_review, build_review_graph
from .agent2_insight import build_insight_graph, suggest_improvements


def cmd_demo(args: argparse.Namespace) -> None:
    app = build_review_graph()
    result = analyze_review(app, args.review, score=args.score)
    print("===== Agent 1: 분석 결과 =====")
    print(result)

    negative = [a for a, l in zip(result.get("aspect", []), result.get("label", [])) if l == 0]
    if negative and args.with_insights:
        insight_app = build_insight_graph()
        report = suggest_improvements(insight_app, result["review"], result["aspect"], result["label"])
        print("\n===== Agent 2: 개선 제안 =====")
        print(report)


async def _analyze_all_async(app, reviews: list[str], concurrency: int) -> list[dict]:
    semaphore = asyncio.Semaphore(concurrency)

    async def analyze_one(review: str) -> dict:
        async with semaphore:
            initial_state = {
                "input_review": review,
                "score": 0,
                "aspect_list": config.ASPECT_LIST,
                "max_num": 3,
                "current_num": 0,
                "messages": [],
                "result": {},
                "criti_score": "",
                "next_step": None,
            }
            final_state = await app.ainvoke(initial_state)
            return final_state["result"]

    return await asyncio.gather(*(analyze_one(r) for r in reviews))


def cmd_batch(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.csv)
    reviews = df["review"].dropna().tolist()
    if args.limit:
        reviews = reviews[: args.limit]

    db.init_db(args.db)
    if args.reset:
        db.clear_all(args.db)
    db.insert_pending_reviews(reviews, args.db)

    pending = db.select_pending_reviews(args.db)
    if not pending:
        print("분석할 리뷰가 없습니다 (이미 모두 분석됨).")
        return

    print(f"-분석할 리뷰 {len(pending)}개 감지됐습니다- (concurrency={args.concurrency})")
    app = build_review_graph()

    start = time.time()
    results = asyncio.run(_analyze_all_async(app, pending, args.concurrency))
    elapsed = time.time() - start
    print(f"-분석 완료- ({elapsed:.1f}초)")

    for review, result in zip(pending, results):
        db.save_result(review, result.get("aspect", []), result.get("label", []), result.get("score", 0), args.db)

    analyzed = db.load_reviews(args.db)
    print(f"\n총 {len(analyzed)}건 저장 완료.")
    summary = db.build_aspect_summary(analyzed, config.ASPECT_LIST)
    summary = summary[summary["언급횟수"] > 0].sort_values("언급횟수", ascending=False)
    print("\n===== 속성별 집계 =====")
    print(summary.to_string(index=False))

    if args.with_insights:
        insight_app = build_insight_graph()
        shown = 0
        for review, result in zip(pending, results):
            negative = [a for a, l in zip(result.get("aspect", []), result.get("label", [])) if l == 0]
            if not negative:
                continue
            report = suggest_improvements(insight_app, result["review"], result["aspect"], result["label"])
            print(f"\n===== 개선 제안 (리뷰: {review[:30]}...) =====")
            print(report)
            shown += 1
            if shown >= args.insight_limit:
                break


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="analyze a single review")
    demo.add_argument("--review", required=True, help="review text to analyze")
    demo.add_argument("--score", type=int, default=0, help="star rating if known (0 = let the agent infer it)")
    demo.add_argument("--with-insights", action="store_true", help="also run Agent 2 on negative aspects")
    demo.set_defaults(func=cmd_demo)

    batch = sub.add_parser("batch", help="analyze every review in a CSV file into a SQLite DB")
    batch.add_argument("--csv", default="data/reviews.csv", help="path to the reviews CSV (must have a 'review' column)")
    batch.add_argument("--db", default=None, help="SQLite file path (default: %s)" % config.DB_PATH)
    batch.add_argument("--concurrency", type=int, default=6, help="max reviews analyzed in parallel")
    batch.add_argument("--limit", type=int, default=None, help="only analyze the first N reviews")
    batch.add_argument("--reset", action="store_true", help="wipe the DB before inserting")
    batch.add_argument("--with-insights", action="store_true", help="also run Agent 2 on a few negative results")
    batch.add_argument("--insight-limit", type=int, default=3, help="max reviews to run Agent 2 on")
    batch.set_defaults(func=cmd_batch)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
