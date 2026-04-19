import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple


# ============================================================
# CONFIG
# ============================================================

DATA_CATEGORIES = "../Data/fake_category_data.csv"
DATA_CELL_PROBS = "../Data/category_cell_probabilities.csv"
DATA_RESPONDENTS = "../Data/synthetic_national_sample.csv"

TARGET_COMPLETES = 200
BASE_TIME_CAP_SECONDS = 480
QUALIFIER_SECONDS_PER_QUESTION = 1.5

SWEEP_BUNDLE_SIZES = [15, 20, 25, 30, 35]
DEV_SEEDS = [1, 2, 3, 4, 5]

MAX_RESPONDENTS_TO_PROCESS = 100_000
RESPONDENT_SUBSAMPLE_N = 100_000

DEMOGRAPHIC_COLS = ["gender", "age_band", "region"]


# ============================================================
# POLICY
# ============================================================

@dataclass
class SweepPolicy:
    bundle_size: int
    max_surveys_per_respondent: int = 99
    base_time_cap_seconds: int = 480
    qualifier_seconds_per_question: float = 1.5
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


# ============================================================
# BUNDLING
# ============================================================

def build_bundles_for_size(categories: pd.DataFrame, bundle_size: int) -> Dict[int, List[int]]:
    """
    Greedy balanced bundling with approximately fixed bundle size.
    """
    work = categories.sort_values(
        ["incidence_rate", "category_length_seconds"],
        ascending=[True, False]
    ).copy()

    work["difficulty_score"] = (
        (1.0 / work["incidence_rate"]) + (work["category_length_seconds"] / 120.0)
    )

    n_categories = len(work)
    n_bundles = int(np.ceil(n_categories / bundle_size))

    bundle_map = {i: [] for i in range(n_bundles)}
    bundle_load = {i: 0.0 for i in range(n_bundles)}

    for _, row in work.iterrows():
        eligible = [b for b in bundle_map if len(bundle_map[b]) < bundle_size]
        target_bundle = min(eligible, key=lambda b: bundle_load[b])
        bundle_map[target_bundle].append(int(row["category_id"]))
        bundle_load[target_bundle] += float(row["difficulty_score"])

    return bundle_map


# ============================================================
# SIMULATION
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


def assign_surveys_for_respondent(
    respondent_rows: pd.DataFrame,
    fills: Dict[int, int],
    categories_by_id: Dict[int, dict],
    priority_rank: Dict[int, int],
    policy: SweepPolicy,
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


def simulate_month_for_bundle_size(
    policy: SweepPolicy,
    respondents: pd.DataFrame,
    categories: pd.DataFrame,
    respondent_category_probs: pd.DataFrame,
    seed: int,
) -> dict:
    categories_by_id = categories.set_index("category_id").to_dict(orient="index")
    priority_rank = make_category_priority(categories, policy.priority_rule)
    bundle_map = build_bundles_for_size(categories, policy.bundle_size)

    sim = simulate_qualifications(respondent_category_probs, seed=seed)
    grouped = sim.groupby("respondent_id", sort=False)

    fills = {int(cat_id): 0 for cat_id in categories["category_id"]}
    respondent_times = []
    respondent_num_surveys = []
    exposure_records = []

    respondent_ids = respondents["respondent_id"].sample(
        frac=1.0, random_state=seed
    ).tolist()

    processed_respondents = 0

    for respondent_id in respondent_ids:
        if processed_respondents >= MAX_RESPONDENTS_TO_PROCESS:
            break

        if all(v >= TARGET_COMPLETES for v in fills.values()):
            break

        bundle_id = choose_bundle_for_respondent(
            respondent_id=respondent_id,
            n_bundles=len(bundle_map),
            seed=seed,
        )
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
    max_exposure_deviation = compute_max_exposure_deviation(
        exposure_df=exposure_df,
        respondents=respondents,
    )

    fills_df = categories[["category_id", "category_name"]].copy()
    fills_df["completes"] = fills_df["category_id"].map(fills)

    return {
        "bundle_size": policy.bundle_size,
        "seed": seed,
        "n_bundles": len(bundle_map),
        "avg_qualifiers_shown": float(np.mean([len(v) for v in bundle_map.values()])),
        "avg_qualifier_time_seconds": float(np.mean([len(v) for v in bundle_map.values()]) * policy.qualifier_seconds_per_question),
        "total_respondents_used": processed_respondents,
        "all_categories_hit_target": bool((fills_df["completes"] >= TARGET_COMPLETES).all()),
        "min_completes": int(fills_df["completes"].min()),
        "mean_interview_seconds": float(np.mean(respondent_times)) if respondent_times else 0.0,
        "p95_interview_seconds": float(np.percentile(respondent_times, 95)) if respondent_times else 0.0,
        "avg_surveys_per_respondent": float(np.mean(respondent_num_surveys)) if respondent_num_surveys else 0.0,
        "share_2plus_surveys": float(np.mean(np.array(respondent_num_surveys) >= 2)) if respondent_num_surveys else 0.0,
        "share_3plus_surveys": float(np.mean(np.array(respondent_num_surveys) >= 3)) if respondent_num_surveys else 0.0,
        "max_exposure_deviation": max_exposure_deviation,
    }


# ============================================================
# EVALUATION
# ============================================================

def aggregate_results(results_df: pd.DataFrame) -> pd.DataFrame:
    summary = results_df.groupby("bundle_size", as_index=False).agg(
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
    return summary.sort_values("bundle_size")


# ============================================================
# MAIN
# ============================================================

def main():
    categories, cell_probs, respondents = load_inputs()

    # lightweight subsample, reduce compute time
    if len(respondents) > RESPONDENT_SUBSAMPLE_N:
        respondents = respondents.sample(
            n=RESPONDENT_SUBSAMPLE_N,
            random_state=42
        ).sort_values("respondent_id").reset_index(drop=True)

    respondent_category_probs = build_respondent_category_prob_table(
        respondents=respondents,
        cell_probs=cell_probs,
        categories=categories,
    )

    rows = []

    for bundle_size in SWEEP_BUNDLE_SIZES:
        print(f"Running bundle size {bundle_size}...")

        policy = SweepPolicy(
            bundle_size=bundle_size,
            max_surveys_per_respondent=99,
            base_time_cap_seconds=BASE_TIME_CAP_SECONDS,
            qualifier_seconds_per_question=QUALIFIER_SECONDS_PER_QUESTION,
            priority_rule="lowest_incidence_first",
        )

        for seed in DEV_SEEDS:
            result = simulate_month_for_bundle_size(
                policy=policy,
                respondents=respondents,
                categories=categories,
                respondent_category_probs=respondent_category_probs,
                seed=seed,
            )
            rows.append(result)

    results_df = pd.DataFrame(rows)
    summary_df = aggregate_results(results_df)

    print("\n=== BUNDLE SIZE SWEEP SUMMARY ===")
    print(summary_df)

    results_df.to_csv("../Reports/bundle_size_sweep_results.csv", index=False)
    summary_df.to_csv("../Reports/bundle_size_sweep_summary.csv", index=False)


if __name__ == "__main__":
    main()