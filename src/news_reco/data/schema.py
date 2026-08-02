"""Column definitions for the project's common, dataset-agnostic schema.

Every dataset adapter (news_reco.data.mind_adapter, and later an EB-NeRD
adapter) must produce tables with exactly these columns, in this order, so
every downstream module -- temporal splitting, the feature store, retrieval,
evaluation -- can be written once against this schema instead of once per
dataset. This is what makes the MIND-first/EB-NeRD-second plan a config +
adapter change rather than a rewrite.
"""

# One row per article in the catalog.
ARTICLES_COLUMNS = ["article_id", "category", "subcategory", "title", "abstract", "text"]

# One row per impression (a user being shown a slate of candidates at a point in time).
IMPRESSIONS_COLUMNS = ["impression_id", "user_id", "time", "history"]

# One row per (impression, candidate article) pair, with the click label.
INTERACTIONS_COLUMNS = ["impression_id", "article_id", "clicked"]
