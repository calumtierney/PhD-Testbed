"""Plan review: score a measurement plan (an engineer's, an algorithm's) on the testbed's chain and compare
plans in decision terms.

Pure Python: no Streamlit, Qt or matplotlib imports; every result is JSON-serialisable. A front end calls
`review_plans` / `compare_plans` / `algorithm_counterparts` and `review.log`; it does no calculation of its own.
See docs/plan_review.md.
"""
