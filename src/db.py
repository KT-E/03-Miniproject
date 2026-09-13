"""SQLite storage for analyzed reviews.

Ported from the notebooks' DB helper cells (Step2 Mission 4's batch
processing section and the final notebook's dashboard). A single table,
``user_reviews``, stores the review text plus the Agent 1 output (aspect
list, label list, star rating).
"""

from __future__ import annotations

import json
import sqlite3
from typing import Iterable, List, Optional

import pandas as pd

from . import config


def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    return sqlite3.connect(db_path or config.DB_PATH)


def init_db(db_path: Optional[str] = None) -> None:
    """Create the user_reviews table if it doesn't exist yet."""
    conn = get_connection(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS user_reviews (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            review     TEXT,
            aspect     TEXT,
            label      TEXT,
            score      INTEGER,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    conn.close()


def clear_all(db_path: Optional[str] = None) -> None:
    conn = get_connection(db_path)
    conn.execute("DELETE FROM user_reviews")
    conn.commit()
    conn.close()


def insert_pending_reviews(reviews: Iterable[str], db_path: Optional[str] = None) -> None:
    """Insert raw review text rows with no analysis yet (aspect/label = NULL)."""
    conn = get_connection(db_path)
    conn.executemany(
        "INSERT INTO user_reviews (review) VALUES (?)",
        [(r,) for r in reviews],
    )
    conn.commit()
    conn.close()


def select_pending_reviews(db_path: Optional[str] = None) -> List[str]:
    """Return review texts that have not been analyzed yet."""
    conn = get_connection(db_path)
    rows = conn.execute("SELECT review FROM user_reviews WHERE aspect IS NULL").fetchall()
    conn.close()
    return [r[0] for r in rows]


def save_result(review: str, aspect: List[str], label: List[int], score: int, db_path: Optional[str] = None) -> None:
    """Write one Agent 1 result back onto its review row."""
    conn = get_connection(db_path)
    conn.execute(
        "UPDATE user_reviews SET aspect = ?, label = ?, score = ? WHERE review = ?",
        (json.dumps(aspect, ensure_ascii=False), json.dumps(label), score, review.strip()),
    )
    conn.commit()
    conn.close()


def insert_analyzed_review(result: dict, db_path: Optional[str] = None) -> None:
    """Insert a fully-analyzed review (used by the interactive dashboard path,
    where a single new review is analyzed and stored in one step)."""
    conn = get_connection(db_path)
    conn.execute(
        "INSERT INTO user_reviews (review, aspect, label, score, updated_at)"
        " VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)",
        (
            result["review"].strip(),
            json.dumps(result["aspect"], ensure_ascii=False),
            json.dumps(result["label"]),
            result["score"],
        ),
    )
    conn.commit()
    conn.close()


def load_reviews(db_path: Optional[str] = None) -> pd.DataFrame:
    """Load every stored review as a DataFrame, with aspect/label parsed back
    into Python lists."""
    conn = get_connection(db_path)
    try:
        df = pd.read_sql_query(
            "SELECT id, review, aspect, label, score, updated_at FROM user_reviews ORDER BY id ASC",
            conn,
        )
    finally:
        conn.close()

    if df.empty:
        return df

    df["aspect_parsed"] = df["aspect"].apply(lambda x: json.loads(x) if pd.notna(x) and x else [])
    df["label_parsed"] = df["label"].apply(lambda x: json.loads(x) if pd.notna(x) and x else [])
    return df


def build_aspect_summary(df: pd.DataFrame, aspect_list: List[str]) -> pd.DataFrame:
    """Aggregate positive/negative counts per aspect across a set of reviews."""
    rows = []
    for aspect in aspect_list:
        pos = neg = 0
        for _, r in df.iterrows():
            aspects = r["aspect_parsed"]
            labels = r["label_parsed"]
            if not aspects or not labels or len(aspects) != len(labels):
                continue
            if aspect in aspects:
                idx = aspects.index(aspect)
                if labels[idx] == 1:
                    pos += 1
                elif labels[idx] == 0:
                    neg += 1
        total = pos + neg
        rows.append(
            {
                "aspect": aspect,
                "긍정": pos,
                "부정": neg,
                "언급횟수": total,
                "긍정비율(%)": round((pos / total * 100) if total > 0 else 0.0, 1),
            }
        )
    return pd.DataFrame(rows)
