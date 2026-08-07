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

# The same idea for Danish (EB-NeRD), a standard list matching common
# Danish NLP resources (e.g. the Snowball stemmer's Danish stopword list) --
# not exhaustive, just the common function words most likely to dominate
# postings lists the way "the"/"a" do in English. Written with real Danish
# characters (æ/ø/å), not romanized approximations -- those wouldn't match
# anything real Danish text tokenizes to.
DANISH_STOPWORDS = frozenset(
    """
    og i jeg det at en den til er som på de med han af for ikke der var
    mig sig men et har om vi min havde ham hun nu over da fra du ud sin
    dem os op man hans hvor eller hvad skal selv her alle vil blev kunne
    ind når være dog noget ville jo deres efter ned skulle denne thi
    jer sådan
    """.split()
)

# [a-z0-9] alone is ASCII-only and would silently mangle Danish text: "på"
# (very common -- "on"/"at") would tokenize to just "p", since "å" isn't in
# the character class and breaks the match rather than being included or
# transliterated. Explicitly including æøå (post lower()) fixes this for
# Danish while staying a strict superset of the English-only pattern.
_TOKEN_PATTERN = re.compile(r"[a-zæøå0-9]+")


def tokenize(text: str, stopwords: frozenset[str] = STOPWORDS) -> list[str]:
    """Lowercase, extract alphanumeric tokens, drop stopwords.

    `stopwords` defaults to English but is a plain parameter, not a
    hardcoded assumption -- BM25Index passes through whichever set matches
    the corpus's language (see DANISH_STOPWORDS for EB-NeRD).
    """
    tokens = _TOKEN_PATTERN.findall(text.lower())
    return [t for t in tokens if t not in stopwords]


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

    def __init__(self, k1: float = 1.5, b: float = 0.75, stopwords: frozenset[str] = STOPWORDS) -> None:
        self.k1 = k1
        self.b = b
        self._stopwords = stopwords
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
            terms = tokenize(text, self._stopwords)
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

    def search(
        self, query_text: str, top_k: int, eligible_doc_ids: set[str] | None = None
    ) -> list[tuple[str, float]]:
        """Return up to top_k (doc_id, score) pairs, ranked by descending BM25 score.

        Only documents sharing at least one term with the query are ever
        scored -- that's the point of the inverted index -- so a query with
        no recognized terms (e.g. an empty string) correctly returns [].

        Args:
            eligible_doc_ids: If given, only these documents can ever be
                returned -- used for freshness gating (see
                news_reco.retrieval.availability.AvailabilityGate): a
                document that hadn't been published yet at the query's
                time is excluded here, not filtered out after the fact,
                so it never displaces a genuinely-eligible document from
                the top_k. None (the default) means no gating, matching
                every previous caller's behavior unchanged.
        """
        query_terms = set(tokenize(query_text, self._stopwords))
        scores: dict[str, float] = {}

        for term in query_terms:
            idf = self._idf(term)
            if idf <= 0:
                continue
            postings = self._inverted_index.get(term, {})
            for doc_id, term_freq in postings.items():
                if eligible_doc_ids is not None and doc_id not in eligible_doc_ids:
                    continue
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
        query_terms = set(tokenize(query_text, self._stopwords))
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
