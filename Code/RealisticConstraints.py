'''
After getting our baseline approach.
We now want to account for the realistic burden of exhausting time constraints,
qualifier count time expense, & bundle size optimization
'''
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple


# ============================================================
# CONFIG
# ============================================================

DATA_CATEGORIES = "./Data/fake_category_data.csv"
DATA_CELL_PROBS = "./Data/category_cell_probabilities.csv"
DATA_RESPONDENTS = "./Data/synthetic_national_sample.csv"

TARGET_COMPLETES = 200
BASE_TIME_CAP_SECONDS = 480

DEV_SEEDS = list(range(1, 11))
TEST_SEEDS = list(range(11, 16))
HOLDOUT_SEEDS = list(range(16, 21))

MAX_RESPONDENTS_TO_PROCESS = 200_000
DEMOGRAPHIC_COLS = ["gender", "age_band", "region"]


# ============================================================
# POLICY DEFINITIONS
# ============================================================

@dataclass
class Policy:
    name: str
    bundle_strategy: str               # "giant" or "incidence_balanced"
    max_surveys_per_respondent: int
    base_time_cap_seconds: int
    qualifier_seconds_per_question: float
    max_qualifiers_per_respondent: int
    priority_rule: str                 # "lowest_incidence_first"
    n_bundles: int


def get_realistic_policies() -> List[Policy]:
    return [
        Policy(
            name="giant_bundle_multi_with_qualifier_cost",
            bundle_strategy="giant",
            max_surveys_per_respondent=99,
            base_time_cap_seconds=480,
            qualifier_seconds_per_question=1.5,
            max_qualifiers_per_respondent=999,
            priority_rule="lowest_incidence_first",
            n_bundles=1,
        ),
        Policy(
            name="bundled_one_survey_15q",
            bundle_strategy="incidence_balanced",
            max_surveys_per_respondent=1,
            base_time_cap_seconds=480,
            qualifier_seconds_per_question=1.5,
            max_qualifiers_per_respondent=15,
            priority_rule="lowest_incidence_first",
            n_bundles=6,
        ),
        Policy(
            name="bundled_multi_survey_15q",
            bundle_strategy="incidence_balanced",
            max_surveys_per_respondent=99,
            base_time_cap_seconds=480,
            qualifier_seconds_per_question=1.5,
            max_qualifiers_per_respondent=15,
            priority_rule="lowest_incidence_first",
            n_bundles=6,
        ),
        Policy(
            name="bundled_multi_survey_10q",
            bundle_strategy="incidence_balanced",
            max_surveys_per_respondent=99,
            base_time_cap_seconds=480,
            qualifier_seconds_per_question=1.5,
            max_qualifiers_per_respondent=10,
            priority_rule="lowest_incidence_first",
            n_bundles=8,
        ),
    ]


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
        raise ValueError(f"Missing category_cell_probs columns: {missing_cell}")
    if missing_resp:
        raise ValueError(f"Missing respondent columns: {missing_resp}")

    return categories, cell_probs, respondents


# ============================================================
# PREP TABLES
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
# BUNDLE CONSTRUCTION
# ============================================================

def build_bundles(categories: pd.DataFrame, policy: Policy) -> Dict[int, List[int]]:
    """
    Returns:
        bundle_id -> list of category_ids
    """
    if policy.bundle_strategy == "giant":
        return {0: categories["category_id"].tolist()}

    if policy.bundle_strategy != "incidence_balanced":
        raise ValueError(f"Unsupported bundle strategy: {policy.bundle_strategy}")

    # Greedy balanced bundling:
    # sort hard categories first (low incidence first, then long surveys)
    work = categories.sort_values(
        ["incidence_rate", "category_length_seconds"],
        ascending=[True, False]
    ).copy()

    bundle_map = {i: [] for i in range(policy.n_bundles)}
    bundle_load = {i: 0.0 for i in range(policy.n_bundles)}

    # expected "difficulty load": lower incidence categories are harder to fill
    # combine qualification difficulty and interview burden
    work["difficulty_score"] = (
        (1.0 / work["incidence_rate"]) + (work["category_length_seconds"] / 120.0)
    )

    for _, row in work.iterrows():
        # put the next category in the currently lightest bundle
        target_bundle = min(bundle_load, key=bundle_load.get)
        bundle_map[target_bundle].append(int(row["category_id"]))
        bundle_load[target_bundle] += float(row["difficulty_score"])

    # Optional safety check for qualifier count
    for bundle_id, cats in bundle_map.items():
        if len(cats) > policy.max_qualifiers_per_respondent:
            raise ValueError(
                f"Bundle {bundle_id} has {len(cats)} categories, exceeding "
                f"max_qualifiers_per_respondent={policy.max_qualifiers_per_respondent}. "
                f"Increase n_bundles or reduce bundle size."
            )

    return bundle_map


def make_category_to_bundle(bundle_map: Dict[int, List[int]]) -> Dict[int, int]:
    out = {}
    for bundle_id, cats in bundle_map.items():
        for cat_id in cats:
            out[cat_id] = bundle_id
    return out


# ============================================================
# QUALIFICATION SIMULATION
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


# ============================================================
# ROUTING
# ============================================================

def choose_bundle_for_respondent(
    respondent_id: int,
    n_bundles: int,
    seed: int,
) -> int:
    """
    Deterministic pseudo-random assignment of respondent to a bundle.
    Keeps bundle exposure nationally representative by randomization.
    """
    return (hash((respondent_id, seed)) % n_bundles)


def assign_surveys_for_respondent(
    respondent_rows: pd.DataFrame,
    fills: Dict[int, int],
    categories_by_id: Dict[int, dict],
    priority_rank: Dict[int, int],
    policy: Policy,
    qualifier_count: int,
) -> Tuple[List[int], float]:
    """
    Assign completed category surveys for one respondent.

    Returns:
        assigned_category_ids, total_time_including_qualifiers
    """
    qualifier_time = qualifier_count * policy.qualifier_seconds_per_question
    effective_time_cap = max(policy.base_time_cap_seconds - qualifier_time, 0.0)

    qualified = respondent_rows.loc[respondent_rows["qualified"] == 1].copy()
    if qualified.empty:
        # Still spends qualifier time
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


# ============================================================
# SIMULATION CORE
# ============================================================

def simulate_month(
    policy: Policy,
    respondents: pd.DataFrame,
    categories: pd.DataFrame,
    respondent_category_probs: pd.DataFrame,
    seed: int,
) -> dict:
    if policy.name == "naive_separate":
        raise ValueError("naive_separate is not included in this realistic script.")

    rng = np.random.default_rng(seed)
    categories_by_id = categories.set_index("category_id").to_dict(orient="index")
    priority_rank = make_category_priority(categories, policy.priority_rule)

    bundle_map = build_bundles(categories, policy)
    category_to_bundle = make_category_to_bundle(bundle_map)

    sim = simulate_qualifications(respondent_category_probs, seed=seed)

    fills = {int(cat_id): 0 for cat_id in categories["category_id"]}

    # exposure tracking for representativeness
    exposure_records = []

    respondent_times = []
    respondent_num_surveys = []
    processed_respondents = 0

    respondent_ids = respondents["respondent_id"].sample(
        frac=1.0, random_state=seed
    ).tolist()

    grouped = sim.groupby("respondent_id", sort=False)

    for respondent_id in respondent_ids:
        if processed_respondents >= MAX_RESPONDENTS_TO_PROCESS:
            break

        if all(v >= TARGET_COMPLETES for v in fills.values()):
            break

        respondent_profile = respondents.loc[
            respondents["respondent_id"] == respondent_id,
            ["respondent_id", "gender", "age_band", "region"]
        ].iloc[0]

        bundle_id = choose_bundle_for_respondent(
            respondent_id=respondent_id,
            n_bundles=len(bundle_map),
            seed=seed,
        )
        exposed_categories = bundle_map[bundle_id]

        respondent_rows = grouped.get_group(respondent_id).copy()
        respondent_rows = respondent_rows[
            respondent_rows["category_id"].isin(exposed_categories)
        ].copy()

        # exposure records
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

    results = summarize_month_results(
        policy=policy,
        seed=seed,
        fills=fills,
        categories=categories,
        respondent_times=respondent_times,
        respondent_num_surveys=respondent_num_surveys,
        total_respondents_used=processed_respondents,
        exposure_df=exposure_df,
        respondents=respondents,
    )
    return results


# ============================================================
# EXPOSURE REPRESENTATIVENESS
# ============================================================

def compute_population_distribution(
    respondents: pd.DataFrame,
) -> pd.DataFrame:
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
    """
    For each category, compare exposure cell shares to population cell shares.
    Return the max absolute deviation across all categories and cells.
    """
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


# ============================================================
# METRICS
# ============================================================

def summarize_month_results(
    policy: Policy,
    seed: int,
    fills: Dict[int, int],
    categories: pd.DataFrame,
    respondent_times: List[float],
    respondent_num_surveys: List[int],
    total_respondents_used: int,
    exposure_df: pd.DataFrame,
    respondents: pd.DataFrame,
) -> dict:
    fills_df = categories[["category_id", "category_name"]].copy()
    fills_df["completes"] = fills_df["category_id"].map(fills)

    all_hit_target = bool((fills_df["completes"] >= TARGET_COMPLETES).all())
    min_completes = int(fills_df["completes"].min())

    mean_interview_seconds = float(np.mean(respondent_times)) if respondent_times else 0.0
    p95_interview_seconds = float(np.percentile(respondent_times, 95)) if respondent_times else 0.0
    avg_surveys_per_respondent = float(np.mean(respondent_num_surveys)) if respondent_num_surveys else 0.0
    share_2plus = float(np.mean(np.array(respondent_num_surveys) >= 2)) if respondent_num_surveys else 0.0
    share_3plus = float(np.mean(np.array(respondent_num_surveys) >= 3)) if respondent_num_surveys else 0.0

    max_exposure_deviation = compute_max_exposure_deviation(
        exposure_df=exposure_df,
        respondents=respondents,
    )

    avg_qualifiers_shown = (
        categories.shape[0] if policy.bundle_strategy == "giant"
        else np.mean([len(v) for v in build_bundles(categories, policy).values()])
    )

    qualifier_time_seconds = avg_qualifiers_shown * policy.qualifier_seconds_per_question

    return {
        "policy_name": policy.name,
        "seed": seed,
        "total_respondents_used": total_respondents_used,
        "all_categories_hit_target": all_hit_target,
        "min_completes": min_completes,
        "mean_interview_seconds": mean_interview_seconds,
        "p95_interview_seconds": p95_interview_seconds,
        "avg_surveys_per_respondent": avg_surveys_per_respondent,
        "share_2plus_surveys": share_2plus,
        "share_3plus_surveys": share_3plus,
        "avg_qualifiers_shown": avg_qualifiers_shown,
        "avg_qualifier_time_seconds": qualifier_time_seconds,
        "max_exposure_deviation": max_exposure_deviation,
    }


def aggregate_policy_results(results_df: pd.DataFrame) -> pd.DataFrame:
    grouped = results_df.groupby("policy_name", as_index=False).agg(
        avg_respondents_used=("total_respondents_used", "mean"),
        p95_respondents_used=("total_respondents_used", lambda x: np.percentile(x, 95)),
        prob_all_categories_hit_target=("all_categories_hit_target", "mean"),
        avg_min_completes=("min_completes", "mean"),
        avg_mean_interview_seconds=("mean_interview_seconds", "mean"),
        p95_mean_interview_seconds=("mean_interview_seconds", lambda x: np.percentile(x, 95)),
        avg_p95_interview_seconds=("p95_interview_seconds", "mean"),
        avg_surveys_per_respondent=("avg_surveys_per_respondent", "mean"),
        avg_share_2plus_surveys=("share_2plus_surveys", "mean"),
        avg_share_3plus_surveys=("share_3plus_surveys", "mean"),
        avg_qualifiers_shown=("avg_qualifiers_shown", "mean"),
        avg_qualifier_time_seconds=("avg_qualifier_time_seconds", "mean"),
        avg_max_exposure_deviation=("max_exposure_deviation", "mean"),
    )

    return grouped.sort_values(
        ["prob_all_categories_hit_target", "avg_respondents_used"],
        ascending=[False, True]
    )


def choose_best_feasible_policy(summary_df: pd.DataFrame) -> pd.DataFrame:
    feasible = summary_df[
        (summary_df["prob_all_categories_hit_target"] >= 0.95) &
        (summary_df["avg_mean_interview_seconds"] < 480)
    ].copy()

    if feasible.empty:
        return summary_df.sort_values(
            ["prob_all_categories_hit_target", "avg_respondents_used"],
            ascending=[False, True]
        ).head(1)

    return feasible.sort_values("avg_respondents_used", ascending=True).head(1)


# ============================================================
# EVALUATION HARNESS
# ============================================================

def run_policy_set(
    policies: List[Policy],
    seeds: List[int],
    respondents: pd.DataFrame,
    categories: pd.DataFrame,
    respondent_category_probs: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for policy in policies:
        for seed in seeds:
            result = simulate_month(
                policy=policy,
                respondents=respondents,
                categories=categories,
                respondent_category_probs=respondent_category_probs,
                seed=seed,
            )
            rows.append(result)
    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main():
    categories, cell_probs, respondents = load_inputs()

    respondent_category_probs = build_respondent_category_prob_table(
        respondents=respondents,
        cell_probs=cell_probs,
        categories=categories,
    )

    policies = get_realistic_policies()

    # Development
    dev_results = run_policy_set(
        policies=policies,
        seeds=DEV_SEEDS,
        respondents=respondents,
        categories=categories,
        respondent_category_probs=respondent_category_probs,
    )
    dev_summary = aggregate_policy_results(dev_results)
    print("\n=== DEVELOPMENT SUMMARY ===")
    print(dev_summary)

    best_policy_df = choose_best_feasible_policy(dev_summary)
    best_policy_name = best_policy_df.iloc[0]["policy_name"]
    print("\nSelected policy from development:")
    print(best_policy_df)

    selected_policies = [p for p in policies if p.name == best_policy_name]

    # Test
    test_results = run_policy_set(
        policies=selected_policies,
        seeds=TEST_SEEDS,
        respondents=respondents,
        categories=categories,
        respondent_category_probs=respondent_category_probs,
    )
    test_summary = aggregate_policy_results(test_results)
    print("\n=== TEST SUMMARY ===")
    print(test_summary)

    # Holdout
    holdout_results = run_policy_set(
        policies=selected_policies,
        seeds=HOLDOUT_SEEDS,
        respondents=respondents,
        categories=categories,
        respondent_category_probs=respondent_category_probs,
    )
    holdout_summary = aggregate_policy_results(holdout_results)
    print("\n=== HOLDOUT SUMMARY ===")
    print(holdout_summary)

    # Save outputs
    dev_results.to_csv("./Reports/realistic_dev_results.csv", index=False)
    test_results.to_csv("./Reports/realistic_test_results.csv", index=False)
    holdout_results.to_csv("./Reports/realistic_holdout_results.csv", index=False)

    dev_summary.to_csv("./Reports/realistic_dev_summary.csv", index=False)
    test_summary.to_csv("./Reports/realistic_test_summary.csv", index=False)
    holdout_summary.to_csv("./Reports/realistic_holdout_summary.csv", index=False)


if __name__ == "__main__":
    main()