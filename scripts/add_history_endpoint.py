from pathlib import Path

api_file = Path("api.py")
source = api_file.read_text(encoding="utf-8")

if "import sqlite3" in source or "/history\")" in source:
    raise SystemExit(
        "Patch may already be applied. No changes made; inspect api.py first."
    )

if "from pathlib import Path" not in source:
    raise SystemExit("Could not find expected import section.")

if "RESULTS = ROOT / \"results\"" not in source:
    raise SystemExit("Could not find expected RESULTS configuration.")

updated = source.replace(
    "from pathlib import Path\n",
    "from pathlib import Path\nimport sqlite3\n",
    1,
)

updated = updated.replace(
    'RESULTS = ROOT / "results"\n',
    'RESULTS = ROOT / "results"\nHISTORY_DB = RESULTS / "historical_reviews.sqlite3"\n',
    1,
)

endpoint = '''

@app.get("/api/reviewers/{user_id}/history")
def get_reviewer_history(
    user_id: str,
    before_date: str | None = None,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    """Return original historical reviews for a reviewer, newest first."""
    from datetime import date

    user_id = user_id.strip()

    if not user_id:
        raise HTTPException(status_code=400, detail="Reviewer ID is required.")

    if before_date:
        try:
            date.fromisoformat(before_date)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="before_date must use YYYY-MM-DD format.",
            )

    if not HISTORY_DB.exists():
        raise HTTPException(
            status_code=503,
            detail="Historical review database is not available.",
        )

    where_clause = "WHERE user_id = ?"
    params = [user_id]

    if before_date:
        where_clause += " AND date < ?"
        params.append(before_date)

    with sqlite3.connect(HISTORY_DB) as conn:
        conn.row_factory = sqlite3.Row

        total = conn.execute(
            f"SELECT COUNT(*) FROM historical_reviews {where_clause}",
            params,
        ).fetchone()[0]

        rows = conn.execute(
            f"""
            SELECT row_id, user_id, product_id, date, rating, review_text
            FROM historical_reviews
            {where_clause}
            ORDER BY date DESC, row_id ASC
            LIMIT ? OFFSET ?
            """,
            params + [limit, offset],
        ).fetchall()

    return {
        "user_id": user_id,
        "count": total,
        "limit": limit,
        "offset": offset,
        "items": [
            {
                "row_id": row["row_id"],
                "user_id": row["user_id"],
                "product_id": row["product_id"],
                "date": row["date"],
                "rating": row["rating"],
                "review_text": row["review_text"],
            }
            for row in rows
        ],
        "before_date": before_date,
        "notice": (
            "These are historical dataset reviews, not necessarily model-scored "
            "predictions. When before_date is supplied, only strictly earlier "
            "calendar dates are included; same-day ordering is not inferred. "
            "Dataset reference labels are not independent proof of fraud."
        ),
    }
'''

updated += endpoint
api_file.write_text(updated, encoding="utf-8")
print("Historical-review endpoint added to api.py")
