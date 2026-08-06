"""BM25 lexical retrieval over article text, via a hand-built inverted index.

An inverted index maps each term to the list of documents containing it
(its "postings"), with how often it occurs in each -- the opposite
direction from a document listing its words, hence "inverted". It's what
makes search fast: scoring a query only touches documents that share at
least one term with it, never the whole corpus.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable

# A small, standard English stopword list -- common function words that
# carry no topical signal and would otherwise dominate postings lists
# (nearly every document contains "the"), diluting real content terms.
STOPWORDS = frozenset(
    """
    a an the and or but is are was were be been being of to in on at for
    with by from as that this these those it its into than then so such
    not no nor do does did has have had will would can could should
    i you he she we they
    """.split()
)

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase, extract alphanumeric tokens, drop stopwords."""
    tokens = _TOKEN_PATTERN.findall(text.lower())
    return [t for t in tokens if t not in STOPWORDS]


def build_query_from_history(history: list[str], get_title: Callable[[str], str]) -> str:
    """Concatenate the titles of a user's clicked-history articles into a BM25 query.

    This is the query-construction method the assignment specifies:
    "concatenate titles of recently clicked articles." Article order within
    history doesn't matter for BM25 (term order isn't part of the scoring),
    so no ordering logic is needed here -- and an empty history correctly
    produces an empty string, which tokenizes to no terms and yields no
    retrieval results (there's nothing to build a query from for a
    cold-start user with no click history).
    """
    titles = [get_title(article_id) for article_id in history]
    return " ".join(titles)


class BM25Index:
    """Okapi BM25 with the standard (always-nonnegative) smoothed IDF term.

    Standard defaults k1=1.5, b=0.75 (Robertson & Zaragoza) are used unless
    overridden; these aren't tuned for this project, just the well-known
    starting point from the literature.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._doc_lengths: dict[str, int] = {}
        self._avg_doc_length: float = 0.0
        self._inverted_index: dict[str, dict[str, int]] = {}  # term -> {doc_id: term_freq}
        self._doc_freq: dict[str, int] = {}  # term -> number of documents containing it
        self._n_docs: int = 0

    def fit(self, doc_ids: list[str], texts: list[str]) -> None:
        """Build the inverted index from a corpus of (doc_id, text) pairs."""
        self._n_docs = len(doc_ids)
        total_length = 0

        for doc_id, text in zip(doc_ids, texts):
            terms = tokenize(text)
            self._doc_lengths[doc_id] = len(terms)
            total_length += len(terms)

            term_counts: dict[str, int] = {}
            for term in terms:
                term_counts[term] = term_counts.get(term, 0) + 1

            for term, count in term_counts.items():
                self._inverted_index.setdefault(term, {})[doc_id] = count
                self._doc_freq[term] = self._doc_freq.get(term, 0) + 1

        self._avg_doc_length = total_length / self._n_docs if self._n_docs > 0 else 0.0

        # IDF depends only on a term's document frequency, which is fixed
        # once fit() finishes -- precomputing it here (rather than calling
        # math.log() inside search(), potentially thousands of times for
        # the same common term across many queries) turns repeated log()
        # calls into one dict lookup per query term.
        self._idf_cache: dict[str, float] = {
            term: self._compute_idf(n_t) for term, n_t in self._doc_freq.items()
        }

    def _compute_idf(self, n_t: int) -> float:
        """Smoothed IDF: log((N - n_t + 0.5) / (n_t + 0.5) + 1).

        The "+1" inside the log keeps this always >= 0 (it's 0 only when a
        term appears in literally every document), unlike the classic
        Robertson-Sparck Jones formula, which can go negative for very
        common terms. This is the variant modern search engines use.
        """
        return math.log((self._n_docs - n_t + 0.5) / (n_t + 0.5) + 1)

    def _idf(self, term: str) -> float:
        """Look up a term's precomputed IDF; 0.0 for a term never seen during fit()."""
        return self._idf_cache.get(term, 0.0)

    def search(self, query_text: str, top_k: int) -> list[tuple[str, float]]:
        """Return up to top_k (doc_id, score) pairs, ranked by descending BM25 score.

        Only documents sharing at least one term with the query are ever
        scored -- that's the point of the inverted index -- so a query with
        no recognized terms (e.g. an empty string) correctly returns [].
        """
        query_terms = set(tokenize(query_text))
        scores: dict[str, float] = {}

        for term in query_terms:
            idf = self._idf(term)
            if idf <= 0:
                continue
            postings = self._inverted_index.get(term, {})
            for doc_id, term_freq in postings.items():
                doc_length = self._doc_lengths[doc_id]
                denom = term_freq + self.k1 * (1 - self.b + self.b * doc_length / self._avg_doc_length)
                term_score = idf * (term_freq * (self.k1 + 1)) / denom
                scores[doc_id] = scores.get(doc_id, 0.0) + term_score

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[:top_k]

    def score_documents(self, query_text: str, doc_ids: list[str]) -> dict[str, float]:
        """Score a specific, fixed set of documents against a query.

        Unlike search(), this never discovers new candidates -- it answers
        "how would BM25 rank exactly these documents", which is what's
        needed to evaluate AUC/MRR/nDCG over an impression's *actual shown*
        candidates (a small, fixed set with known labels), as opposed to
        open-corpus retrieval (search()) where most candidates have no
        ground-truth label at all. Iterates query terms x requested doc_ids
        (both typically small) rather than full postings lists, since the
        candidate set here is usually much smaller than a term's postings.
        """
        query_terms = set(tokenize(query_text))
        scores: dict[str, float] = {}

        for doc_id in doc_ids:
            doc_length = self._doc_lengths.get(doc_id)
            if doc_length is None:  # doc_id unknown to this index -- score 0
                scores[doc_id] = 0.0
                continue

            total = 0.0
            for term in query_terms:
                term_freq = self._inverted_index.get(term, {}).get(doc_id, 0)
                if term_freq == 0:
                    continue
                idf = self._idf(term)
                if idf <= 0:
                    continue
                denom = term_freq + self.k1 * (1 - self.b + self.b * doc_length / self._avg_doc_length)
                total += idf * (term_freq * (self.k1 + 1)) / denom
            scores[doc_id] = total

        return scores
