
from core.router import route_query, load_book_index
index = load_book_index("coa")
results = route_query(
  "difference between 1s and 2s complement",
  index,
  top_k=3,
  )
for r in results:
  print(r["section_id"], r["confidence"], r["reason"])

  # in phase-0 folder run this python3 -m core.tester