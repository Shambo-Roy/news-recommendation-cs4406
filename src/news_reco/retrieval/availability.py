"""Freshness gating: which articles were actually available at a given time.

The assignment's anti-gaming clause asks to "report metrics with and
without features unavailable at serving time." For candidate generation,
the clearest instance of that is time itself: retrieving an article that
hadn't been published yet is something a real production system could
never do, no matter how textually or semantically relevant it looks.
MIND has no publish-date field at all (flagged since Milestone 2), so this
gate simply cannot be built for it -- EB-NeRD's real published_time makes
it possible here for the first time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class AvailabilityGate:
    """Determines which articles were published at or before a given time.

    Raises immediately if constructed from a dataset with no published_time
    data (e.g. MIND) -- refusing loudly is safer than silently gating
    nothing (which would look like gating happened when it didn't) or
    gating everything (which would look like a bug, not a limitation).
    """

    def __init__(self, articles: pd.DataFrame) -> None:
        if articles["published_time"].isna().all():
            raise ValueError(
                "This dataset has no published_time data -- freshness gating is not possible "
                "(see the Milestone 2 limitation for MIND)."
            )
        known = articles.dropna(subset=["published_time"]).sort_values("published_time")
        self._sorted_times = known["published_time"].to_numpy()
        self._sorted_ids = known["article_id"].to_numpy()

    def eligible_at(self, time: pd.Timestamp) -> set[str]:
        """Return the set of article_ids published at or before `time`.

        Uses binary search (np.searchsorted) over the pre-sorted publish
        times rather than filtering the whole articles table per call --
        this runs once per impression, so an O(n) filter every time would
        needlessly dominate an otherwise-fast evaluation loop.
        """
        cutoff = np.searchsorted(self._sorted_times, np.datetime64(time), side="right")
        return set(self._sorted_ids[:cutoff])
