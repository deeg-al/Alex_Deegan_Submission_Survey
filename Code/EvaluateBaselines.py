'''
We are going to be evaluating a models to predict survey qualification based on demographic features. 
To do this, we need a dataset that is nationally representative across key demographic dimensions.

'''
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
TIME_CAP_SECONDS = 480

DEV_SEEDS = list(range(1, 11))
TEST_SEEDS = list(range(11, 16))
HOLDOUT_SEEDS = list(range(16, 21))

N_RESPONDENTS_POOL_PER_WORLD = 200_000
MAX_RESPONDENTS_TO_PROCESS = 200_000  # safety guard

DEMOGRAPHIC_COLS = ["gender", "age_band", "region"]


# ============================================================
# POLICY DEFINITIONS
# ============================================================

@dataclass
class Policy:
    name: str
    max_surveys_per_respondent: int
    time_cap_seconds: int
    priority_rule: str  # currently supports: "lowest_incidence_first"


def get_baseline_policies() -> List[Policy]:
    return [
        Policy(
            name="naive_separate",
            max_surveys_per_respondent=1,
            time_cap_seconds=480,
            priority_rule="lowest_incidence_first",
        ),
        Policy(
            name="giant_bundle_one_survey",
            max_surveys_per_respondent=1,
            time_cap_seconds=480,
            priority_rule="lowest_incidence_first",
        ),
        Policy(
            name="giant_bundle_multi_survey_capped",
            max_surveys_per_respondent=99,
            time_cap_seconds=480,
            priority_rule="lowest_incidence_first",
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
# PREP
# ============================================================

def build_respondent_category_prob_table(
    respondents: pd.DataFrame,
    cell_probs: pd.DataFrame,
    categories: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create respondent x category probabilities by joining respondent demographic cell
    to category-cell calibrated probabilities.
    """
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


def assign_surveys_for_respondent(
    respondent_rows: pd.DataFrame,
    fills: Dict[int, int],
    categories_by_id: Dict[int, dict],
    priority_rank: Dict[int, int],
    policy: Policy,
) -> List[int]:
    """
    Given all qualified categories for a respondent, assign completed category surveys
    according to the policy.
    """
    qualified = respondent_rows.loc[respondent_rows["qualified"] == 1].copy()
    if qualified.empty:
        return []

    # Skip categories already filled
    qualified = qualified[qualified["category_id"].map(lambda x: fills[x] < TARGET_COMPLETES)]
    if qualified.empty:
        return []

    qualified["priority_rank"] = qualified["category_id"].map(priority_rank)
    qualified = qualified.sort_values("priority_rank")

    assigned = []
    total_time = 0.0

    for _, row in qualified.iterrows():
        cat_id = int(row["category_id"])
        length_sec = float(categories_by_id[cat_id]["category_length_seconds"])

        if len(assigned) >= policy.max_surveys_per_respondent:
            break

        if total_time + length_sec > policy.time_cap_seconds:
            continue

        if fills[cat_id] >= TARGET_COMPLETES:
            continue

        assigned.append(cat_id)
        total_time += length_sec

    return assigned


def simulate_month(
    policy: Policy,
    respondents: pd.DataFrame,
    categories: pd.DataFrame,
    respondent_category_probs: pd.DataFrame,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)

    categories_by_id = categories.set_index("category_id").to_dict(orient="index")
    priority_rank = make_category_priority(categories, policy.priority_rule)

    # For giant bundle policies, every respondent is exposed to all qualifiers.
    # For naive_separate, simulate category-by-category independent fieldwork.
    if policy.name == "naive_separate":
        return simulate_month_naive_separate(
            respondents=respondents,
            categories=categories,
            respondent_category_probs=respondent_category_probs,
            seed=seed,
        )

    sim = simulate_qualifications(respondent_category_probs, seed=seed)

    fills = {int(cat_id): 0 for cat_id in categories["category_id"]}
    exposure_counts = {int(cat_id): 0 for cat_id in categories["category_id"]}

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

        respondent_rows = grouped.get_group(respondent_id).copy()

        # Giant bundle exposure: all categories shown to every respondent
        for cat_id in exposure_counts:
            exposure_counts[cat_id] += 1

        assigned = assign_surveys_for_respondent(
            respondent_rows=respondent_rows,
            fills=fills,
            categories_by_id=categories_by_id,
            priority_rank=priority_rank,
            policy=policy,
        )

        total_time = 0.0
        for cat_id in assigned:
            fills[cat_id] += 1
            total_time += float(categories_by_id[cat_id]["category_length_seconds"])

        respondent_times.append(total_time)
        respondent_num_surveys.append(len(assigned))
        processed_respondents += 1

    results = summarize_month_results(
        policy_name=policy.name,
        seed=seed,
        fills=fills,
        categories=categories,
        respondent_times=respondent_times,
        respondent_num_surveys=respondent_num_surveys,
        total_respondents_used=processed_respondents,
        exposure_counts=exposure_counts,
        respondents=respondents,
    )

    return results


def simulate_month_naive_separate(
    respondents: pd.DataFrame,
    categories: pd.DataFrame,
    respondent_category_probs: pd.DataFrame,
    seed: int,
) -> dict:
    """
    Benchmark: each category is filled independently.
    This will double-count respondents across categories, which is fine for the benchmark
    because it represents separate fielding cost.
    """
    sim = simulate_qualifications(respondent_category_probs, seed=seed)

    fills = {}
    exposure_counts = {}
    total_respondents_used = 0
    respondent_times = []
    respondent_num_surveys = []

    for _, cat in categories.iterrows():
        cat_id = int(cat["category_id"])
        cat_length = float(cat["category_length_seconds"])

        cat_rows = sim[sim["category_id"] == cat_id].copy()
        cat_rows = cat_rows.sample(frac=1.0, random_state=seed + cat_id)

        qualified_cumsum = cat_rows["qualified"].cumsum()
        hit_idx = np.argmax((qualified_cumsum >= TARGET_COMPLETES).to_numpy())

        # If never reaches target, use whole pool
        if qualified_cumsum.iloc[hit_idx] < TARGET_COMPLETES:
            n_used = len(cat_rows)
            completes = int(cat_rows["qualified"].sum())
        else:
            n_used = hit_idx + 1
            completes = TARGET_COMPLETES

        fills[cat_id] = completes
        exposure_counts[cat_id] = n_used
        total_respondents_used += n_used

        # In naive separate, only qualified respondents do the category survey
        respondent_times.extend([cat_length] * completes)
        respondent_num_surveys.extend([1] * completes)

        # Non-qualified respondents also count as surveyed, with 0 full-survey time
        non_qualified_count = n_used - completes
        respondent_times.extend([0.0] * non_qualified_count)
        respondent_num_surveys.extend([0] * non_qualified_count)

    results = summarize_month_results(
        policy_name="naive_separate",
        seed=seed,
        fills=fills,
        categories=categories,
        respondent_times=respondent_times,
        respondent_num_surveys=respondent_num_surveys,
        total_respondents_used=total_respondents_used,
        exposure_counts=exposure_counts,
        respondents=respondents,
    )

    return results


# ============================================================
# METRICS
# ============================================================

def summarize_month_results(
    policy_name: str,
    seed: int,
    fills: Dict[int, int],
    categories: pd.DataFrame,
    respondent_times: List[float],
    respondent_num_surveys: List[int],
    total_respondents_used: int,
    exposure_counts: Dict[int, int],
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

    # For current giant-bundle setup, exposure is identical for all categories and therefore
    # perfectly nationally representative by construction if respondent order is random.
    # Placeholder metric for future bundle-specific exposure validation.
    max_exposure_deviation = 0.0

    return {
        "policy_name": policy_name,
        "seed": seed,
        "total_respondents_used": total_respondents_used,
        "all_categories_hit_target": all_hit_target,
        "min_completes": min_completes,
        "mean_interview_seconds": mean_interview_seconds,
        "p95_interview_seconds": p95_interview_seconds,
        "avg_surveys_per_respondent": avg_surveys_per_respondent,
        "share_2plus_surveys": share_2plus,
        "share_3plus_surveys": share_3plus,
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
        avg_max_exposure_deviation=("max_exposure_deviation", "mean"),
    )
    return grouped.sort_values(
        ["prob_all_categories_hit_target", "avg_respondents_used"],
        ascending=[False, True]
    )


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
# MAIN
# ============================================================

def main():
    categories, cell_probs, respondents = load_inputs()

    respondent_category_probs = build_respondent_category_prob_table(
        respondents=respondents,
        cell_probs=cell_probs,
        categories=categories,
    )

    policies = get_baseline_policies()

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

    # Save
    dev_results.to_csv("../Reports/dev_results.csv", index=False)
    test_results.to_csv("../Reports/test_results.csv", index=False)
    holdout_results.to_csv("../Reports/holdout_results.csv", index=False)

    dev_summary.to_csv("../Reports/dev_summary.csv", index=False)
    test_summary.to_csv("../Reports/test_summary.csv", index=False)
    holdout_summary.to_csv("../Reports/holdout_summary.csv", index=False)


if __name__ == "__main__":
    main()
