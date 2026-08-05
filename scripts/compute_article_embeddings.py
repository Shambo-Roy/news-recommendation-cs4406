"""Compute and cache sentence embeddings for the full article catalog.

Encodes every article's text (title + abstract -- the same field BM25
indexes, for a fair later comparison) with a pretrained sentence encoder,
and saves the result to data/interim/<dataset>/article_embeddings.npz.
This is the expensive, one-time step Milestone 7 depends on: everything
downstream (EmbeddingIndex, recall@K evaluation) loads this cached file
instead of re-running the encoder.

Usage:
    python scripts/compute_article_embeddings.py --config configs/mind_small.yaml
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Import sentence_transformers (which pulls in torch) before anything that
# imports pandas/pyarrow (FeatureStore, below): on Windows, torch's and
# pyarrow's bundled native libraries conflict if pyarrow loads first,
# crashing with a DLL initialization error. See
# news_reco.retrieval.semantic.embeddings for the full explanation -- this
# import order requirement was found empirically, not documented anywhere.
import sentence_transformers  # noqa: E402,F401

import argparse  # noqa: E402

import numpy as np  # noqa: E402

from news_reco.features.store import FeatureStore  # noqa: E402
from news_reco.retrieval.semantic.embeddings import compute_embeddings
from news_reco.utils.config import PipelineConfig, load_config

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a dataset YAML config.")
    args = parser.parse_args()
    config: PipelineConfig = load_config(args.config)

    store = FeatureStore(config)
    articles = store.articles

    print(f"Encoding {len(articles):,} articles with {MODEL_NAME} (CPU) ...")
    embeddings = compute_embeddings(articles["text"].tolist(), model_name=MODEL_NAME)
    print(f"Done. Embedding matrix shape: {embeddings.shape}")

    out_path = config.interim_dir / "article_embeddings.npz"
    np.savez(
        out_path,
        # .astype(str) forces a fixed-width numpy unicode dtype rather than
        # pandas' default object dtype (an array of Python str objects) --
        # object arrays require allow_pickle=True to load back, which
        # np.load refuses by default for good reason (arbitrary pickle
        # deserialization is a real security risk for untrusted files).
        article_ids=articles["article_id"].to_numpy().astype(str),
        embeddings=embeddings.astype(np.float32),
    )
    print(f"Wrote embeddings to {out_path}")


if __name__ == "__main__":
    main()
