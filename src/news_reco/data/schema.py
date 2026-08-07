"""Column definitions for the project's common, dataset-agnostic schema.

Every dataset adapter (news_reco.data.mind_adapter, and later an EB-NeRD
adapter) must produce tables with exactly these columns, in this order, so
every downstream module -- temporal splitting, the feature store, retrieval,
evaluation -- can be written once against this schema instead of once per
dataset. This is what makes the MIND-first/EB-NeRD-second plan a config +
adapter change rather than a rewrite.
"""

# One row per article in the catalog. published_time is nullable (NaT):
# MIND has no publish-date field at all (flagged since Milestone 2), so its
# adapter fills NaT for every article -- EB-NeRD's adapter fills real
# timestamps. Keeping this in the shared schema (rather than EB-NeRD-only)
# means one freshness-gating function can serve both datasets, correctly
# reporting "cannot gate, no data" for MIND instead of silently skipping it.
ARTICLES_COLUMNS = ["article_id", "category", "subcategory", "title", "abstract", "text", "published_time"]

# One row per impression (a user being shown a slate of candidates at a point in time).
IMPRESSIONS_COLUMNS = ["impression_id", "user_id", "time", "history"]

# One row per (impression, candidate article) pair, with the click label.
INTERACTIONS_COLUMNS = ["impression_id", "article_id", "clicked"]
