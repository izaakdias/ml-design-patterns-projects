"""Generate an item catalog, user profiles, and implicit interactions."""

from __future__ import annotations

import numpy as np
import pandas as pd

CATEGORY_TERMS = {
    "ciencia": ["astronomia", "biologia", "clima", "experimento", "laboratorio"],
    "culinaria": ["receita", "cozinha", "ingrediente", "forno", "sabor"],
    "arte": ["desenho", "fotografia", "museu", "pintura", "criatividade"],
    "tecnologia": ["programacao", "dados", "robotica", "seguranca", "software"],
    "historia": ["arquivo", "civilizacao", "memoria", "periodo", "documento"],
}


def make_catalog(seed: int = 61) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    categories = list(CATEGORY_TERMS)
    formats = ["guia", "curso", "artigo", "colecao", "introducao"]
    languages = ["pt", "en"]
    items = []
    for category_index, category in enumerate(categories):
        terms = CATEGORY_TERMS[category]
        for offset in range(30):
            term_a = terms[offset % len(terms)]
            term_b = terms[(offset + 2) % len(terms)]
            format_name = formats[offset % len(formats)]
            language = languages[(offset + category_index) % len(languages)]
            title = f"{format_name.title()} de {term_a}"
            description = (
                f"Material de {category} sobre {term_a}, {term_b}, "
                f"aprendizado pratico e ideias para explorar"
            )
            items.append(
                {
                    "item_id": f"item-{len(items):04d}",
                    "title": title,
                    "description": description,
                    "category": category,
                    "language": language,
                    "quality_score": round(float(rng.uniform(0.55, 1.0)), 3),
                }
            )
    catalog = pd.DataFrame(items)

    user_rows = []
    interaction_rows = []
    item_ids_by_category = {
        category: catalog.loc[catalog["category"] == category, "item_id"].to_numpy()
        for category in categories
    }
    for user_index in range(60):
        user_id = f"user-{user_index:03d}"
        preferred_category = categories[user_index % len(categories)]
        language = languages[(user_index // 2) % len(languages)]
        user_rows.append(
            {
                "user_id": user_id,
                "preferred_category": preferred_category,
                "language": language,
            }
        )
        favorite_items = rng.choice(
            item_ids_by_category[preferred_category],
            size=11,
            replace=False,
        )
        other_categories = [category for category in categories if category != preferred_category]
        other_items = np.concatenate(
            [item_ids_by_category[category] for category in other_categories]
        )
        exploration_items = rng.choice(other_items, size=4, replace=False)
        for item_id in np.concatenate([favorite_items, exploration_items]):
            interaction_rows.append({"user_id": user_id, "item_id": str(item_id)})

    return (
        catalog,
        pd.DataFrame(user_rows),
        pd.DataFrame(interaction_rows),
    )
