"""Build shared features and serve keyed search and recommendation results."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


@dataclass(frozen=True)
class FeatureStore:
    """One versioned snapshot shared by batch and HTTP scoring."""

    catalog: list[dict]
    user_languages: dict[str, str]
    user_to_index: dict[str, int]
    user_vectors: np.ndarray
    item_vectors: np.ndarray
    text_vectorizer: TfidfVectorizer
    item_text_matrix: csr_matrix
    category_affinity: dict[str, dict[str, float]]
    seen_items: dict[str, set[str]]
    popularity: dict[str, float]


def split_interactions(
    interactions: pd.DataFrame,
    seed: int = 62,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if interactions.empty or interactions.duplicated(["user_id", "item_id"]).any():
        raise ValueError("Interactions must be nonempty unique user-item pairs.")
    if not interactions.index.is_unique:
        raise ValueError("Interaction index must be unique.")
    rng = np.random.default_rng(seed)
    held_out = []
    for _, group in interactions.groupby("user_id", sort=True):
        if len(group) < 2:
            raise ValueError("At least two interactions per user are required for holdout.")
        held_out.append(rng.choice(group.index.to_numpy()))
    test = interactions.loc[held_out].reset_index(drop=True)
    train = interactions.drop(index=held_out).reset_index(drop=True)
    return train, test


def build_feature_store(
    catalog: pd.DataFrame,
    users: pd.DataFrame,
    interactions: pd.DataFrame,
    seed: int = 63,
) -> FeatureStore:
    if len(catalog) < 2 or len(users) < 2 or interactions.empty:
        raise ValueError("At least two items, two users, and interactions are required.")
    if catalog["item_id"].duplicated().any() or users["user_id"].duplicated().any():
        raise ValueError("Item and user IDs must be unique.")
    if (
        not interactions["user_id"].isin(users["user_id"]).all()
        or not interactions["item_id"].isin(catalog["item_id"]).all()
    ):
        raise ValueError("Interactions reference unknown users or items.")
    if not catalog["quality_score"].between(0, 1).all():
        raise ValueError("Quality scores must be between zero and one.")
    item_ids = catalog["item_id"].tolist()
    user_ids = users["user_id"].tolist()
    item_to_index = {item_id: index for index, item_id in enumerate(item_ids)}
    user_to_index = {user_id: index for index, user_id in enumerate(user_ids)}

    rows = interactions["user_id"].map(user_to_index).to_numpy()
    columns = interactions["item_id"].map(item_to_index).to_numpy()
    interaction_matrix = csr_matrix(
        (np.ones(len(interactions), dtype=np.float32), (rows, columns)),
        shape=(len(user_ids), len(item_ids)),
    )
    components = min(16, len(user_ids) - 1, len(item_ids) - 1)
    decomposition = TruncatedSVD(n_components=components, random_state=seed)
    user_vectors = normalize(decomposition.fit_transform(interaction_matrix))
    item_vectors = normalize(decomposition.components_.T)

    documents = (
        catalog["title"].astype(str)
        + " "
        + catalog["description"].astype(str)
        + " "
        + catalog["category"].astype(str)
    ).tolist()
    text_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), max_features=3000, strip_accents="unicode"
    )
    item_text_matrix = text_vectorizer.fit_transform(documents)

    enriched = interactions.merge(catalog[["item_id", "category"]], on="item_id", how="left")
    category_affinity = {}
    for user_id, group in enriched.groupby("user_id", sort=True):
        category_affinity[str(user_id)] = group["category"].value_counts(normalize=True).to_dict()
    seen_items = {
        str(user_id): set(group["item_id"].astype(str))
        for user_id, group in interactions.groupby("user_id", sort=True)
    }
    counts = interactions["item_id"].value_counts().to_dict()
    max_count = max(counts.values(), default=1)

    return FeatureStore(
        catalog=catalog.to_dict(orient="records"),
        user_languages=users.set_index("user_id")["language"].to_dict(),
        user_to_index=user_to_index,
        user_vectors=user_vectors,
        item_vectors=item_vectors,
        text_vectorizer=text_vectorizer,
        item_text_matrix=item_text_matrix,
        category_affinity=category_affinity,
        seen_items=seen_items,
        popularity={
            str(item_id): float(np.log1p(count) / np.log1p(max_count))
            for item_id, count in counts.items()
        },
    )


def recommend_items(
    store: FeatureStore,
    user_id: str,
    limit: int = 10,
) -> list[dict]:
    if user_id not in store.user_to_index:
        raise KeyError(f"Unknown user_id: {user_id}")
    if limit < 1 or limit > 50:
        raise ValueError("limit must be between 1 and 50.")

    user_index = store.user_to_index[user_id]
    user_language = store.user_languages[user_id]
    affinity = store.category_affinity.get(user_id, {})
    collaborative = store.item_vectors @ store.user_vectors[user_index]
    seen = set(store.seen_items.get(user_id, []))

    ranked = []
    for index, item in enumerate(store.catalog):
        item_id = item["item_id"]
        if item_id in seen:
            continue
        collaborative_score = float(np.clip((collaborative[index] + 1.0) / 2.0, 0.0, 1.0))
        topic_match = float(affinity.get(item["category"], 0.0))
        language_match = float(item["language"] == user_language)
        quality = float(item["quality_score"])
        popularity = float(store.popularity.get(item_id, 0.0))
        topic_quality_cross = topic_match * quality
        language_topic_cross = language_match * topic_match
        score = (
            0.48 * collaborative_score
            + 0.22 * topic_match
            + 0.10 * language_match
            + 0.13 * topic_quality_cross
            + 0.04 * language_topic_cross
            + 0.03 * popularity
        )
        ranked.append(
            {
                "user_id": user_id,
                "item_id": item_id,
                "title": item["title"],
                "category": item["category"],
                "score": float(score),
                "feature_crosses": {
                    "topic_x_quality": round(topic_quality_cross, 4),
                    "language_x_topic": round(language_topic_cross, 4),
                },
            }
        )
    selected = sorted(ranked, key=lambda item: (-item["score"], item["item_id"]))[:limit]
    return [{**item, "score": round(item["score"], 4)} for item in selected]


def search_items(store: FeatureStore, query: str, limit: int = 10) -> list[dict]:
    if not query.strip():
        raise ValueError("query cannot be empty.")
    if limit < 1 or limit > 50:
        raise ValueError("limit must be between 1 and 50.")
    query_vector = store.text_vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, store.item_text_matrix).ravel()
    ranked = []
    indices = sorted(
        range(len(similarities)),
        key=lambda index: (-similarities[index], store.catalog[index]["item_id"]),
    )
    for index in indices[:limit]:
        if similarities[index] <= 0:
            continue
        item = store.catalog[int(index)]
        ranked.append(
            {
                "item_id": item["item_id"],
                "title": item["title"],
                "category": item["category"],
                "score": round(float(np.clip(similarities[index], 0, 1)), 4),
            }
        )
    return ranked
