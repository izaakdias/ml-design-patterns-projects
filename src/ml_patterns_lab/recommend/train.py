"""Create item and user embeddings, evaluate retrieval, and version the store."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ml_patterns_lab.recommend.data import make_catalog
from ml_patterns_lab.recommend.engine import (
    build_feature_store,
    recommend_items,
    split_interactions,
)
from ml_patterns_lab.shared import save_bundle


def train() -> tuple[Path, dict]:
    catalog, users, interactions = make_catalog()
    train_interactions, test_interactions = split_interactions(interactions)
    store = build_feature_store(catalog, users, train_interactions)

    hits = 0
    discounted_gains = []
    baseline_hits = 0
    popularity = train_interactions["item_id"].value_counts()
    for row in test_interactions.itertuples(index=False):
        top_items = recommend_items(store, row.user_id, limit=10)
        ranked_ids = [item["item_id"] for item in top_items]
        hit = row.item_id in ranked_ids
        hits += hit
        discounted_gains.append(1 / np.log2(ranked_ids.index(row.item_id) + 2) if hit else 0.0)
        seen = store.seen_items.get(row.user_id, set())
        baseline_ids = [item_id for item_id in popularity.index if item_id not in seen][:10]
        baseline_hits += row.item_id in baseline_ids
    hit_rate = hits / max(len(test_interactions), 1)
    metrics = {
        "users": int(len(users)),
        "catalog_items": int(len(catalog)),
        "training_interactions": int(len(train_interactions)),
        "held_out_interactions": int(len(test_interactions)),
        "hit_rate_at_10": round(float(hit_rate), 4),
        "ndcg_at_10": round(float(np.mean(discounted_gains)), 4),
        "popularity_baseline_hit_rate_at_10": round(baseline_hits / len(test_interactions), 4),
    }
    version_path, metadata = save_bundle(
        "recommend",
        store,
        {
            "metrics": metrics,
            "data_seed": 61,
            "embedding_dimensions": int(store.item_vectors.shape[1]),
            "user_key": "user_id",
            "item_key": "item_id",
        },
    )
    return version_path, metadata


def main() -> None:
    version_path, metadata = train()
    print(
        json.dumps(
            {"artifact": str(version_path), "metadata": metadata}, indent=2, ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()
