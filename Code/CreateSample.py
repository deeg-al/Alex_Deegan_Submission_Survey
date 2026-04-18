'''
Script for generating our sample dataframe.
'''

import numpy as np
import pandas as pd
from itertools import product


def make_nationally_representative_dataset(
    n_respondents: int = 50000,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Create a synthetic nationally representative respondent dataset.

    Output columns:
        respondent_id
        gender
        age_band
        region
        quota_cell
        base_weight

    Notes:
    - The marginal distributions below are for testing purposes.
    - This creates a respondent-level dataset sampled from a joint quota-cell distribution.
    """

    rng = np.random.default_rng(random_state)

    # 1) Define national marginals

    gender_dist = {
        "Male": 0.49,
        "Female": 0.51,
    }

    age_dist = {
        "18-24": 0.12,
        "25-34": 0.18,
        "35-44": 0.17,
        "45-54": 0.17,
        "55-64": 0.16,
        "65+": 0.20,
    }

    region_dist = {
        "North": 0.22,
        "South": 0.19,
        "East": 0.18,
        "West": 0.16,
        "Central": 0.15,
        "Rural/Other": 0.10,
    }

    # Validate inputs sum to 1
    for name, dist in [("gender_dist", gender_dist), ("age_dist", age_dist), ("region_dist", region_dist)]:
        total = sum(dist.values())
        if not np.isclose(total, 1.0):
            raise ValueError(f"{name} must sum to 1. Got {total:.6f}")

    # -------------------------------------------------------
    # 2) Create a joint quota-cell distribution from marginals
    # -------------------------------------------------------
    # Assumption: independence across quota dimensions.
    cells = []
    for gender, age_band, region in product(gender_dist, age_dist, region_dist):
        p = gender_dist[gender] * age_dist[age_band] * region_dist[region]
        cells.append(
            {
                "gender": gender,
                "age_band": age_band,
                "region": region,
                "cell_prob": p,
            }
        )

    cell_df = pd.DataFrame(cells)
    cell_df["quota_cell"] = (
        cell_df["gender"] + " | " + cell_df["age_band"] + " | " + cell_df["region"]
    )

    # Normalize just in case of floating-point drift
    cell_df["cell_prob"] = cell_df["cell_prob"] / cell_df["cell_prob"].sum()

    # --------------------------------------
    # 3) Sample respondents from quota cells
    # --------------------------------------
    sampled_cells = rng.choice(
        cell_df.index,
        size=n_respondents,
        replace=True,
        p=cell_df["cell_prob"].to_numpy(),
    )

    respondents = cell_df.loc[sampled_cells, ["gender", "age_band", "region", "quota_cell"]].reset_index(drop=True)
    respondents.insert(0, "respondent_id", np.arange(1, n_respondents + 1))

    # Optional base weight for downstream weighted checks
    # Since we sampled exactly from the target distribution, weight = 1 for all rows.
    respondents["base_weight"] = 1.0

    return respondents


def summarize_representation(df: pd.DataFrame) -> None:
    """
    Print simple checks showing achieved vs target distributions.
    """
    print("\nGender distribution")
    print((df["gender"].value_counts(normalize=True).sort_index() * 100).round(2))

    print("\nAge distribution")
    print((df["age_band"].value_counts(normalize=True).sort_index() * 100).round(2))

    print("\nRegion distribution")
    print((df["region"].value_counts(normalize=True).sort_index() * 100).round(2))

    print("\nTop quota cells")
    print((df["quota_cell"].value_counts(normalize=True).head(10) * 100).round(2))


if __name__ == "__main__":
    df = make_nationally_representative_dataset(
        n_respondents=200000,
        random_state=42,
    )

    print(df.head())
    summarize_representation(df)

    df.to_csv("../Data/synthetic_national_sample.csv", index=False)
    print("\nSaved to synthetic_national_sample.csv")
