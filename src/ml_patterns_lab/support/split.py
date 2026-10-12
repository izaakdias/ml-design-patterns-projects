"""Split phrase families, never their near-duplicate generated variants."""

import numpy as np
import pandas as pd


def split_phrase_families(
    data: pd.DataFrame, seed: int = 41
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    train_groups, validation_groups, test_groups = [], [], []
    for _, group in data.groupby("intent", sort=True):
        families = group["phrase_group"].unique()
        if len(families) < 3:
            raise ValueError("At least three phrase families per intent are required.")
        families = rng.permutation(families)
        holdout_size = max(1, int(len(families) * 0.20))
        test_groups.extend(families[:holdout_size])
        validation_groups.extend(families[holdout_size : 2 * holdout_size])
        train_groups.extend(families[2 * holdout_size :])
    return tuple(
        data.loc[data["phrase_group"].isin(groups)].copy()
        for groups in (train_groups, validation_groups, test_groups)
    )
