from pathlib import Path

path = Path("dashboard.py")
source = path.read_text(encoding="utf-8")

# Ensure the threshold exists before the selector's format function is used.
marker = '            st.markdown("### Find a review")'
if marker not in source:
    raise SystemExit("Could not find the review selector section. No changes made.")

review_start = source.index(marker)
review_end = source.find('elif section == "Reviewer Analysis":', review_start)
if review_end == -1:
    raise SystemExit("Could not find the end of Review Analyzer. No changes made.")

section = source[review_start:review_end]

# Add the threshold before case_label/selectbox.
if '            threshold = 0.20' in section:
    section = section.replace('            threshold = 0.20\n', '', 1)

section = section.replace(
    marker,
    '            threshold = 0.20\n' + marker,
    1,
)

# Remove the internal row ID from the visible details.
section = section.replace(
    '                st.write("Review row ID:", int(review["row_id"]))\n',
    '',
    1,
)

source = source[:review_start] + section + source[review_end:]
path.write_text(source, encoding="utf-8")
print("Threshold moved before selector; visible review row ID removed.")
