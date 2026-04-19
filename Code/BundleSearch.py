"""
bundle_search.py

Purpose
-------
Search over different ways of grouping categories into bundles, while keeping the
bundle size roughly fixed, to identify bundle compositions that minimize the total
number of respondents required.

Why this script exists
----------------------
Earlier experiments established:
1. Allowing multiple category surveys per respondent is far more efficient than
   one-survey-per-respondent.
2. Adding qualifier time creates a more realistic baseline and raises the required
   respondents versus the idealized upper bound.
3. Bundle size matters a lot. In this project, the promising range appears to be
   about 20-30 qualifiers per respondent.

The next question is not "how many qualifiers should be shown?" but:
"which categories should be shown together?"

That is what this script explores.

Bundle strategies implemented
-----------------------------

1. random
   - Shuffle all categories randomly, then split them into bundles of roughly
     equal size.
   - This is the simplest baseline for bundle composition.
   - Good for estimating how much performance varies just from grouping choices.

2. difficulty_balanced
   - Sort categories by "difficulty" and assign them greedily to the currently
     lightest bundle.
   - Difficulty is approximated using:
       (1 / incidence_rate) + (category_length_seconds / 120)
   - Intuition:
       * Low-incidence categories are harder to fill.
       * Longer categories consume more respondent time.
       * Spreading hard categories across bundles avoids creating one impossible
         bundle and tends to stabilize fieldwork.

3. similarity_clustered
   - Build a demographic profile vector for each category using the category-by-cell
     qualification probabilities, then cluster categories by profile similarity.
   - Categories with similar qualification patterns across demographic cells tend
     to qualify among similar types of respondents.
   - Intuition:
       * Grouping similar categories may increase multi-survey reuse within a bundle.
       * This can reduce the total respondents needed.

4. hybrid_similarity_balanced
   - First form similarity-based groups, then spread those groups across bundles
     in a load-balanced way.
   - Intuition:
       * Preserve some within-bundle similarity to improve reuse.
       * Avoid concentrating all hard or all easy categories in the same bundle.
   - In practice this is often a strong candidate because it balances overlap
     efficiency and bundle feasibility.

Search approach
---------------
For a fixed target bundle size, this script:
- generates many candidate bundle layouts using one or more strategies,
- simulates each candidate across several random seeds,
- aggregates metrics,
- ranks candidates by respondent cost subject to feasibility.

Primary optimization target
---------------------------
Minimize:
- total respondents used

Subject to:
- all categories hitting target completes,
- mean interview length remaining below the time cap,
- acceptable qualifier exposure representativeness.

Key outputs
-----------
1. candidate-level results:
   One row per candidate x seed
2. candidate summary:
   Aggregated results for each candidate layout
3. top bundle assignments:
   The best-performing bundle layouts, saved so they can be evaluated again
   on more seeds or used in the final policy evaluation.

Laptop-friendliness
-------------------
This script is designed to stay manageable on a laptop by:
- fixing bundle size instead of jointly optimizing size and composition,
- using a moderate number of candidates,
- using a moderate number of seeds,
- stopping simulation once all categories hit target,
- optionally subsampling respondents.

Recommended workflow
--------------------
1. Run this script for one promising bundle size (e.g. 25).
2. Inspect the best-performing strategies and bundle layouts.
3. Re-run the top few layouts with more seeds or at neighboring bundle sizes
   (e.g. 20 and 30).
4. Choose the final policy for holdout evaluation.
"""

import json
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

DATA_CATEGORIES = "../Data/fake_category_data.csv"
DATA_CELL_PROBS = "../Data/category_cell_probabilities.csv"
DATA_RESPONDENTS = "../Data/synthetic_national_sample.csv"

TARGET_COMPLETES = 200
BASE_TIME_CAP_SECONDS = 480
QUALIFIER_SECONDS_PER_QUESTION = 1.5

# Fixed bundle size for search.
# Change this to 20 or 30 for sensitivity checks after first run.
TARGET_BUNDLE_SIZE = 25

# Laptop-friendly defaults
N_RANDOM_CANDIDATES = 20
N_DIFFICULTY_CANDIDATES = 5
N_SIMILARITY_CANDIDATES = 10
N_HYBRID_CANDIDATES = 10

SEARCH_SEEDS = [1, 2, 3]
TOP_K_TO_SAVE = 10

RESPONDENT_SUBSAMPLE_N = 100_000
MAX_RESPONDENTS_TO_PROCESS = 100_000

DEMOGRAPHIC_COLS = ["gender", "age_band", "region"]


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class SearchPolicy:
    bundle_size: int = TARGET_BUNDLE_SIZE
    max_surveys_per_respondent: int = 99
    base_time_cap_seconds: int = BASE_TIME_CAP_SECONDS
    qualifier_seconds_per_question: float = QUALIFIER_SECONDS_PER_QUESTION
    priority_rule: str = "lowest_incidence_first"


# ============================================================
# LOADERS
# ============================================================

def load_inputs() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    categories = pd.read_csv(DATA_CATEGORIES)
    cell_probs = pd.read_csv(DATA_CELL_PROBS)
    respondents = pd.read_csv(DATA_RESPONDENTS)

    required_cat_cols = {
        "category_id", "category_name", "incidence_rate", "category_length_seconds"
    }
    required_cell_cols = {
        "category_id", "gender", "age_band", "region", "calibrated_probability"
    }
    required_resp_cols = {"respondent_id", "gender", "age_band", "region"}

    missing_cat = required_cat_cols - set(categories.columns)
    missing_cell = required_cell_cols - set(cell_probs.columns)
    missing_resp = required_resp_cols - set(respondents.columns)

    if missing_cat:
        raise ValueError(f"Missing category columns: {missing_cat}")
    if missing_cell:
        raise ValueError(f"Missing category_cell_prob columns: {missing_cell}")
    if missing_resp:
        raise ValueError(f"Missing respondent columns: {missing_resp}")

    return categories, cell_probs, respondents


# ============================================================
# PREP
# ============================================================

def maybe_subsample_respondents(
    respondents: pd.DataFrame,
    n: int = RESPONDENT_SUBSAMPLE_N,
) -> pd.DataFrame:
    if len(respondents) <= n:
        return respondents.copy().reset_index(drop=True)

    return (
        respondents
        .sample(n=n, random_state=42)
        .sort_values("respondent_id")
        .reset_index(drop=True)
    )


def build_respondent_category_prob_table(
    respondents: pd.DataFrame,
    cell_probs: pd.DataFrame,
    categories: pd.DataFrame,
) -> pd.DataFrame:
    rc = respondents.merge(
        cell_probs[
            ["category_id", "gender", "age_band", "region", "calibrated_probability"]
        ],
        on=["gender", "age_band", "region"],
        how="left",
    ).merge(
        categories[
            ["category_id", "category_name", "incidence_rate", "category_length_seconds"]
        ],
        on="category_id",
        how="left",
    )

    rc = rc.rename(columns={"calibrated_probability": "qualify_probability"})
    return rc


def make_category_priority(categories: pd.DataFrame, rule: str) -> Dict[int, int]:
    if rule != "lowest_incidence_first":
        raise ValueError(f"Unsupported priority rule: {rule}")

    ordered = categories.sort_values(
        ["incidence_rate", "category_length_seconds", "category_id"],
        ascending=[True, False, True]
    )["category_id"].tolist()

    return {cat_id: rank for rank, cat_id in enumerate(ordered)}


def compute_category_profiles(
    cell_probs: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a category x demographic-cell matrix of qualification probabilities.
    This is used for similarity-based bundle strategies.
    """
    cp = cell_probs.copy()
    cp["cell_key"] = (
        cp["gender"].astype(str) + " | "
        + cp["age_band"].astype(str) + " | "
        + cp["region"].astype(str)
    )

    wide = cp.pivot_table(
        index="category_id",
        columns="cell_key",
        values="calibrated_probability",
        aggfunc="mean",
        fill_value=0.0,
    )

    wide = wide.sort_index()
    return wide


def normalize_rows(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return matrix / norms


# ============================================================
# BUNDLE STRATEGIES
# ============================================================

def chunk_list(items: List[int], chunk_size: int) -> Dict[int, List[int]]:
    return {
        i: items[i * chunk_size:(i + 1) * chunk_size]
        for i in range(math.ceil(len(items) / chunk_size))
    }


def difficulty_score(categories: pd.DataFrame) -> pd.Series:
    return (1.0 / categories["incidence_rate"]) + (categories["category_length_seconds"] / 120.0)


def build_random_bundle_map(
    categories: pd.DataFrame,
    bundle_size: int,
    rng: np.random.Generator,
) -> Dict[int, List[int]]:
    cat_ids = categories["category_id"].tolist()
    rng.shuffle(cat_ids)
    return chunk_list(cat_ids, bundle_size)


def build_difficulty_balanced_bundle_map(
    categories: pd.DataFrame,
    bundle_size: int,
    rng: Optional[np.random.Generator] = None,
) -> Dict[int, List[int]]:
    work = categories.copy()
    work["difficulty"] = difficulty_score(work)

    # Add slight randomness for multiple candidates
    if rng is not None:
        work["noise"] = rng.normal(0, 1e-6, size=len(work))
    else:
        work["noise"] = 0.0

    work = work.sort_values(
        ["difficulty", "noise"],
        ascending=[False, False]
    )

    n_bundles = math.ceil(len(work) / bundle_size)
    bundle_map = {i: [] for i in range(n_bundles)}
    bundle_load = {i: 0.0 for i in range(n_bundles)}

    for _, row in work.iterrows():
        eligible = [b for b in bundle_map if len(bundle_map[b]) < bundle_size]
        target_bundle = min(eligible, key=lambda b: bundle_load[b])
        bundle_map[target_bundle].append(int(row["category_id"]))
        bundle_load[target_bundle] += float(row["difficulty"])

    return bundle_map


def greedy_similarity_path_order(
    profile_df: pd.DataFrame,
    start_idx: int = 0,
) -> List[int]:
    """
    Create an ordering by greedily walking to the most similar unused category.
    This is a lightweight alternative to full clustering.
    """
    cat_ids = profile_df.index.to_list()
    X = normalize_rows(profile_df.to_numpy())
    sim = X @ X.T

    n = len(cat_ids)
    used = np.zeros(n, dtype=bool)

    order = []
    current = start_idx
    for _ in range(n):
        order.append(cat_ids[current])
        used[current] = True

        candidate_indices = np.where(~used)[0]
        if len(candidate_indices) == 0:
            break

        next_idx = candidate_indices[np.argmax(sim[current, candidate_indices])]
        current = next_idx

    return order


def build_similarity_clustered_bundle_map(
    categories: pd.DataFrame,
    profile_df: pd.DataFrame,
    bundle_size: int,
    rng: np.random.Generator,
) -> Dict[int, List[int]]:
    cat_ids = categories["category_id"].tolist()
    profiles = profile_df.loc[cat_ids]

    start_idx = int(rng.integers(0, len(profiles)))
    ordered_cat_ids = greedy_similarity_path_order(profiles, start_idx=start_idx)

    return chunk_list(ordered_cat_ids, bundle_size)


def build_hybrid_similarity_balanced_bundle_map(
    categories: pd.DataFrame,
    profile_df: pd.DataFrame,
    bundle_size: int,
    rng: np.random.Generator,
) -> Dict[int, List[int]]:
    """
    Hybrid strategy:
    1. Create similarity-neighbor groups.
    2. Spread those groups across bundles while balancing difficulty.
    """
    cat_ids = categories["category_id"].tolist()
    profiles = profile_df.loc[cat_ids]

    start_idx = int(rng.integers(0, len(profiles)))
    ordered_cat_ids = greedy_similarity_path_order(profiles, start_idx=start_idx)

    # Create small similarity blocks, then distribute them across bundles
    block_size = max(2, min(5, bundle_size // 5))
    blocks = [
        ordered_cat_ids[i:i + block_size]
        for i in range(0, len(ordered_cat_ids), block_size)
    ]

    cat_meta = categories.set_index("category_id")
    block_scores = []
    for block in blocks:
        score = float(
            ((1.0 / cat_meta.loc[block, "incidence_rate"]) + (cat_meta.loc[block, "category_length_seconds"] / 120.0)).sum()
        )
        block_scores.append(score)

    block_order = np.argsort(block_scores)[::-1]
    blocks = [blocks[i] for i in block_order]

    n_bundles = math.ceil(len(categories) / bundle_size)
    bundle_map = {i: [] for i in range(n_bundles)}
    bundle_load = {i: 0.0 for i in range(n_bundles)}

    for block, block_score in zip(blocks, [block_scores[i] for i in block_order]):
        eligible = [b for b in bundle_map if len(bundle_map[b]) + len(block) <= bundle_size]
        if not eligible:
            # fallback: place items one-by-one if exact block placement fails
            for cat_id in block:
                eligible_single = [b for b in bundle_map if len(bundle_map[b]) < bundle_size]
                target_bundle = min(eligible_single, key=lambda b: bundle_load[b])
                bundle_map[target_bundle].append(int(cat_id))
                row = cat_meta.loc[cat_id]
                bundle_load[target_bundle] += float((1.0 / row["incidence_rate"]) + (row["category_length_seconds"] / 120.0))
        else:
            target_bundle = min(eligible, key=lambda b: bundle_load[b])
            bundle_map[target_bundle].extend([int(x) for x in block])
            bundle_load[target_bundle] += block_score

    return bundle_map


def serialize_bundle_map(bundle_map: Dict[int, List[int]]) -> str:
    clean = {int(k): [int(x) for x in v] for k, v in bundle_map.items()}
    return json.dumps(clean, sort_keys=True)


def validate_bundle_map(bundle_map: Dict[int, List[int]], categories: pd.DataFrame) -> None:
    flat = [cat for cats in bundle_map.values() for cat in cats]
    expected = sorted(categories["category_id"].tolist())

    if sorted(flat) != expected:
        raise ValueError("Bundle map does not contain exactly the full set of categories.")

    sizes = [len(v) for v in bundle_map.values()]
    if min(sizes) <= 0:
        raise ValueError("Bundle map contains an empty bundle.")


# ============================================================
# CANDIDATE GENERATION
# ============================================================

def generate_candidates(
    categories: pd.DataFrame,
    profile_df: pd.DataFrame,
    bundle_size: int,
) -> pd.DataFrame:
    rows = []
    candidate_id = 0

    # Random
    for i in range(N_RANDOM_CANDIDATES):
        rng = np.random.default_rng(1000 + i)
        bundle_map = build_random_bundle_map(categories, bundle_size, rng)
        validate_bundle_map(bundle_map, categories)
        rows.append({
            "candidate_id": candidate_id,
            "strategy": "random",
            "bundle_map_json": serialize_bundle_map(bundle_map),
        })
        candidate_id += 1

    # Difficulty-balanced
    for i in range(N_DIFFICULTY_CANDIDATES):
        rng = np.random.default_rng(2000 + i)
        bundle_map = build_difficulty_balanced_bundle_map(categories, bundle_size, rng)
        validate_bundle_map(bundle_map, categories)
        rows.append({
            "candidate_id": candidate_id,
            "strategy": "difficulty_balanced",
            "bundle_map_json": serialize_bundle_map(bundle_map),
        })
        candidate_id += 1

    # Similarity-clustered
    for i in range(N_SIMILARITY_CANDIDATES):
        rng = np.random.default_rng(3000 + i)
        bundle_map = build_similarity_clustered_bundle_map(categories, profile_df, bundle_size, rng)
        validate_bundle_map(bundle_map, categories)
        rows.append({
            "candidate_id": candidate_id,
            "strategy": "similarity_clustered",
            "bundle_map_json": serialize_bundle_map(bundle_map),
        })
        candidate_id += 1

    # Hybrid
    for i in range(N_HYBRID_CANDIDATES):
        rng = np.random.default_rng(4000 + i)
        bundle_map = build_hybrid_similarity_balanced_bundle_map(categories, profile_df, bundle_size, rng)
        validate_bundle_map(bundle_map, categories)
        rows.append({
            "candidate_id": candidate_id,
            "strategy": "hybrid_similarity_balanced",
            "bundle_map_json": serialize_bundle_map(bundle_map),
        })
        candidate_id += 1

    return pd.DataFrame(rows)


def deserialize_bundle_map(bundle_map_json: str) -> Dict[int, List[int]]:
    raw = json.loads(bundle_map_json)
    return {int(k): [int(x) for x in v] for k, v in raw.items()}


# ============================================================
# SIMULATION CORE
# ============================================================

def simulate_qualifications(
    respondent_category_probs: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = respondent_category_probs.copy()
    df["qualified"] = (
        rng.random(len(df)) < df["qualify_probability"].to_numpy()
    ).astype(int)
    return df


def choose_bundle_for_respondent(
    respondent_id: int,
    n_bundles: int,
    seed: int,
) -> int:
    return hash((respondent_id, seed)) % n_bundles


def compute_population_distribution(respondents: pd.DataFrame) -> pd.DataFrame:
    pop = (
        respondents.groupby(DEMOGRAPHIC_COLS, as_index=False)
        .size()
        .rename(columns={"size": "n_population"})
    )
    pop["population_share"] = pop["n_population"] / pop["n_population"].sum()
    return pop


def compute_max_exposure_deviation(
    exposure_df: pd.DataFrame,
    respondents: pd.DataFrame,
) -> float:
    if exposure_df.empty:
        return np.nan

    pop = compute_population_distribution(respondents)

    exp = (
        exposure_df.groupby(["category_id"] + DEMOGRAPHIC_COLS, as_index=False)
        .size()
        .rename(columns={"size": "n_exposed"})
    )

    totals = (
        exp.groupby("category_id", as_index=False)["n_exposed"]
        .sum()
        .rename(columns={"n_exposed": "n_exposed_total"})
    )

    exp = exp.merge(totals, on="category_id", how="left")
    exp["exposure_share"] = exp["n_exposed"] / exp["n_exposed_total"]

    merged = exp.merge(pop, on=DEMOGRAPHIC_COLS, how="left")
    merged["abs_dev"] = (merged["exposure_share"] - merged["population_share"]).abs()

    return float(merged["abs_dev"].max())


def assign_surveys_for_respondent(
    respondent_rows: pd.DataFrame,
    fills: Dict[int, int],
    categories_by_id: Dict[int, dict],
    priority_rank: Dict[int, int],
    policy: SearchPolicy,
    qualifier_count: int,
) -> Tuple[List[int], float]:
    qualifier_time = qualifier_count * policy.qualifier_seconds_per_question
    effective_time_cap = max(policy.base_time_cap_seconds - qualifier_time, 0.0)

    qualified = respondent_rows.loc[respondent_rows["qualified"] == 1].copy()
    if qualified.empty:
        return [], qualifier_time

    qualified = qualified[qualified["category_id"].map(lambda x: fills[x] < TARGET_COMPLETES)]
    if qualified.empty:
        return [], qualifier_time

    qualified["priority_rank"] = qualified["category_id"].map(priority_rank)
    qualified = qualified.sort_values("priority_rank")

    assigned = []
    category_time = 0.0

    for _, row in qualified.iterrows():
        cat_id = int(row["category_id"])
        length_sec = float(categories_by_id[cat_id]["category_length_seconds"])

        if len(assigned) >= policy.max_surveys_per_respondent:
            break

        if fills[cat_id] >= TARGET_COMPLETES:
            continue

        if category_time + length_sec > effective_time_cap:
            continue

        assigned.append(cat_id)
        category_time += length_sec

    total_time = qualifier_time + category_time
    return assigned, total_time


def simulate_month_for_candidate(
    candidate_id: int,
    strategy: str,
    bundle_map: Dict[int, List[int]],
    policy: SearchPolicy,
    respondents: pd.DataFrame,
    categories: pd.DataFrame,
    respondent_category_probs: pd.DataFrame,
    seed: int,
) -> dict:
    categories_by_id = categories.set_index("category_id").to_dict(orient="index")
    priority_rank = make_category_priority(categories, policy.priority_rule)

    sim = simulate_qualifications(respondent_category_probs, seed=seed)
    grouped = sim.groupby("respondent_id", sort=False)

    fills = {int(cat_id): 0 for cat_id in categories["category_id"]}
    respondent_times = []
    respondent_num_surveys = []
    exposure_records = []

    respondent_ids = respondents["respondent_id"].sample(frac=1.0, random_state=seed).tolist()
    processed_respondents = 0

    n_bundles = len(bundle_map)
    avg_qualifiers_shown = float(np.mean([len(v) for v in bundle_map.values()]))

    for respondent_id in respondent_ids:
        if processed_respondents >= MAX_RESPONDENTS_TO_PROCESS:
            break

        if all(v >= TARGET_COMPLETES for v in fills.values()):
            break

        bundle_id = choose_bundle_for_respondent(respondent_id, n_bundles, seed)
        exposed_categories = bundle_map[bundle_id]

        respondent_profile = respondents.loc[
            respondents["respondent_id"] == respondent_id,
            ["gender", "age_band", "region"]
        ].iloc[0]

        respondent_rows = grouped.get_group(respondent_id).copy()
        respondent_rows = respondent_rows[
            respondent_rows["category_id"].isin(exposed_categories)
        ].copy()

        for cat_id in exposed_categories:
            exposure_records.append({
                "respondent_id": respondent_id,
                "category_id": cat_id,
                "gender": respondent_profile["gender"],
                "age_band": respondent_profile["age_band"],
                "region": respondent_profile["region"],
            })

        assigned, total_time = assign_surveys_for_respondent(
            respondent_rows=respondent_rows,
            fills=fills,
            categories_by_id=categories_by_id,
            priority_rank=priority_rank,
            policy=policy,
            qualifier_count=len(exposed_categories),
        )

        for cat_id in assigned:
            fills[cat_id] += 1

        respondent_times.append(total_time)
        respondent_num_surveys.append(len(assigned))
        processed_respondents += 1

    exposure_df = pd.DataFrame(exposure_records)
    fills_df = categories[["category_id", "category_name"]].copy()
    fills_df["completes"] = fills_df["category_id"].map(fills)

    return {
        "candidate_id": candidate_id,
        "strategy": strategy,
        "seed": seed,
        "bundle_size_target": policy.bundle_size,
        "n_bundles": n_bundles,
        "avg_qualifiers_shown": avg_qualifiers_shown,
        "avg_qualifier_time_seconds": avg_qualifiers_shown * policy.qualifier_seconds_per_question,
        "total_respondents_used": processed_respondents,
        "all_categories_hit_target": bool((fills_df["completes"] >= TARGET_COMPLETES).all()),
        "min_completes": int(fills_df["completes"].min()),
        "mean_interview_seconds": float(np.mean(respondent_times)) if respondent_times else 0.0,
        "p95_interview_seconds": float(np.percentile(respondent_times, 95)) if respondent_times else 0.0,
        "avg_surveys_per_respondent": float(np.mean(respondent_num_surveys)) if respondent_num_surveys else 0.0,
        "share_2plus_surveys": float(np.mean(np.array(respondent_num_surveys) >= 2)) if respondent_num_surveys else 0.0,
        "share_3plus_surveys": float(np.mean(np.array(respondent_num_surveys) >= 3)) if respondent_num_surveys else 0.0,
        "max_exposure_deviation": compute_max_exposure_deviation(exposure_df, respondents),
    }


# ============================================================
# EVALUATION
# ============================================================

def aggregate_candidate_results(results_df: pd.DataFrame) -> pd.DataFrame:
    summary = results_df.groupby(["candidate_id", "strategy"], as_index=False).agg(
        bundle_size_target=("bundle_size_target", "first"),
        n_bundles=("n_bundles", "mean"),
        avg_qualifiers_shown=("avg_qualifiers_shown", "mean"),
        avg_qualifier_time_seconds=("avg_qualifier_time_seconds", "mean"),
        avg_respondents_used=("total_respondents_used", "mean"),
        p95_respondents_used=("total_respondents_used", lambda x: np.percentile(x, 95)),
        prob_all_categories_hit_target=("all_categories_hit_target", "mean"),
        avg_min_completes=("min_completes", "mean"),
        avg_mean_interview_seconds=("mean_interview_seconds", "mean"),
        avg_p95_interview_seconds=("p95_interview_seconds", "mean"),
        avg_surveys_per_respondent=("avg_surveys_per_respondent", "mean"),
        avg_share_2plus_surveys=("share_2plus_surveys", "mean"),
        avg_share_3plus_surveys=("share_3plus_surveys", "mean"),
        avg_max_exposure_deviation=("max_exposure_deviation", "mean"),
    )

    summary = summary.sort_values(
        ["prob_all_categories_hit_target", "avg_respondents_used"],
        ascending=[False, True]
    )
    return summary


def aggregate_strategy_results(candidate_summary_df: pd.DataFrame) -> pd.DataFrame:
    strategy_summary = candidate_summary_df.groupby("strategy", as_index=False).agg(
        n_candidates=("candidate_id", "count"),
        best_respondents_used=("avg_respondents_used", "min"),
        median_respondents_used=("avg_respondents_used", "median"),
        mean_respondents_used=("avg_respondents_used", "mean"),
        best_hit_prob=("prob_all_categories_hit_target", "max"),
    )
    return strategy_summary.sort_values("best_respondents_used")


def attach_bundle_maps(
    candidate_summary_df: pd.DataFrame,
    candidates_df: pd.DataFrame,
) -> pd.DataFrame:
    return candidate_summary_df.merge(
        candidates_df,
        on=["candidate_id", "strategy"],
        how="left",
    )


# ============================================================
# MAIN
# ============================================================

def main():
    categories, cell_probs, respondents = load_inputs()
    respondents = maybe_subsample_respondents(respondents, RESPONDENT_SUBSAMPLE_N)

    respondent_category_probs = build_respondent_category_prob_table(
        respondents=respondents,
        cell_probs=cell_probs,
        categories=categories,
    )

    profile_df = compute_category_profiles(cell_probs)

    candidates_df = generate_candidates(
        categories=categories,
        profile_df=profile_df,
        bundle_size=TARGET_BUNDLE_SIZE,
    )

    print(f"Generated {len(candidates_df)} candidate bundle layouts.")

    policy = SearchPolicy(bundle_size=TARGET_BUNDLE_SIZE)

    result_rows = []
    for _, candidate in candidates_df.iterrows():
        candidate_id = int(candidate["candidate_id"])
        strategy = candidate["strategy"]
        bundle_map = deserialize_bundle_map(candidate["bundle_map_json"])

        print(f"Evaluating candidate {candidate_id} ({strategy})...")

        for seed in SEARCH_SEEDS:
            result = simulate_month_for_candidate(
                candidate_id=candidate_id,
                strategy=strategy,
                bundle_map=bundle_map,
                policy=policy,
                respondents=respondents,
                categories=categories,
                respondent_category_probs=respondent_category_probs,
                seed=seed,
            )
            result_rows.append(result)

    results_df = pd.DataFrame(result_rows)
    candidate_summary_df = aggregate_candidate_results(results_df)
    strategy_summary_df = aggregate_strategy_results(candidate_summary_df)

    top_candidates_df = attach_bundle_maps(
        candidate_summary_df.head(TOP_K_TO_SAVE),
        candidates_df,
    )

    print("\n=== STRATEGY SUMMARY ===")
    print(strategy_summary_df)

    print("\n=== TOP CANDIDATES ===")
    print(
        top_candidates_df[
            [
                "candidate_id",
                "strategy",
                "bundle_size_target",
                "n_bundles",
                "avg_qualifiers_shown",
                "avg_respondents_used",
                "prob_all_categories_hit_target",
                "avg_mean_interview_seconds",
                "avg_surveys_per_respondent",
                "avg_max_exposure_deviation",
            ]
        ]
    )

    results_df.to_csv("../Reports/bundle_search_results.csv", index=False)
    candidate_summary_df.to_csv("../Reports/bundle_search_candidate_summary.csv", index=False)
    strategy_summary_df.to_csv("../Reports/bundle_search_strategy_summary.csv", index=False)
    top_candidates_df.to_csv("../Reports/bundle_search_top_candidates.csv", index=False)


if __name__ == "__main__":
    main()