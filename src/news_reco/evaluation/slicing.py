"""Slicing utilities: partition impressions into head/tail by clicked-article popularity.

Aggregate metrics can hide a real difference between methods that only
shows up on a subset of traffic -- e.g. one method might win overall by
being better at recommending already-popular articles, while doing worse
on obscure ones. Slicing by "was the article the user clicked popular or
not" surfaces that, where a single blended number wouldn't.
"""

from __future__ import annotations


def classify_head_tail(clicked_article_ids: set[str], train_click_counts: dict[str, int], threshold: int) -> str:
    """Return "head" if any clicked article's train-split popularity >= threshold, else "tail".

    Uses the *maximum* popularity among an impression's clicked articles
    (an impression can have more than one click) -- if any one of them is
    a popular "head" article, the impression counts as head, since the
    method only needed to surface one popular hit to succeed there.

    train_click_counts must come from the train split only (same principle
    as NoveltyModel): using val/test popularity to slice val/test
    evaluation would be circular -- classifying an impression using
    information about how it (or its outcome) behaves in the very data
    being evaluated.
    """
    max_popularity = max((train_click_counts.get(aid, 0) for aid in clicked_article_ids), default=0)
    return "head" if max_popularity >= threshold else "tail"
