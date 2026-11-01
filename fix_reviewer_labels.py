from pathlib import Path

path = Path("dashboard.py")
source = path.read_text(encoding="utf-8")

old = '''            eligible = eligible.reset_index(drop=True)
            eligible["selector_label"] = eligible.apply(
                lambda r: (
                    f"Reviewer {eligible.index[eligible['user_id'].eq(r['user_id'])[0]] + 1:03d}"
                    f" · {int(r['review_count']):,} reviews"
                ),
                axis=1,
            )'''

new = '''            eligible = eligible.reset_index(drop=True)
            eligible["selector_label"] = [
                f"Reviewer {i + 1:03d} · {int(row['review_count']):,} reviews"
                for i, (_, row) in enumerate(eligible.iterrows())
            ]'''

if old not in source:
    raise SystemExit(
        "Could not find the exact faulty reviewer-label code. "
        "No changes were made."
    )

source = source.replace(old, new, 1)
path.write_text(source, encoding="utf-8")
print("Reviewer labels fixed.")
