import pytest

from ml_patterns_lab.recommend.data import make_catalog
from ml_patterns_lab.recommend.engine import (
    build_feature_store,
    recommend_items,
    split_interactions,
)


def test_holdout_and_recommendations_exclude_seen_items():
    catalog, users, interactions = make_catalog()
    train, test = split_interactions(interactions)
    assert len(test) == len(users)
    assert len(train) + len(test) == len(interactions)
    assert set(map(tuple, train.to_numpy())).isdisjoint(set(map(tuple, test.to_numpy())))
    store = build_feature_store(catalog, users, train)
    results = recommend_items(store, "user-001")
    assert {row["item_id"] for row in results}.isdisjoint(store.seen_items["user-001"])
    assert results == recommend_items(store, "user-001")
    assert all(0 <= row["score"] <= 1 for row in results)


def test_store_rejects_unknown_references():
    catalog, users, interactions = make_catalog()
    interactions.loc[0, "item_id"] = "missing"
    with pytest.raises(ValueError, match="unknown"):
        build_feature_store(catalog, users, interactions)


def test_duplicate_interactions_do_not_leak_into_holdout():
    _, _, interactions = make_catalog()
    interactions.loc[1] = interactions.loc[0]
    with pytest.raises(ValueError, match="unique"):
        split_interactions(interactions)
