'''
If every respondent in the synthetic population has the same qualification probability for a category,
then demographics only matter for exposure, not for who qualifies.
That misses an important part of the problem.

synthetic setup is:

keep nationally representative exposure
allow qualification probabilities to vary by demographic cell
calibrate those cell-level probabilities so the overall category incidence still matches the CSV
'''

import numpy as np
import pandas as pd

# -----------------------------
# 1) Archetype multiplier rules
# -----------------------------
ARCHETYPE_RULES = {
    "broad": {
        "gender": {"Male": 1.00, "Female": 1.00},
        "age_band": {
            "18-24": 1.00,
            "25-34": 1.00,
            "35-44": 1.00,
            "45-54": 1.00,
            "55-64": 1.00,
            "65+": 1.00,
        },
        "region": {
            "North": 1.00,
            "South": 1.00,
            "East": 1.00,
            "West": 1.00,
            "Central": 1.00,
            "Rural/Other": 1.00,
        },
    },
    "female_skew": {
        "gender": {"Male": 0.35, "Female": 1.65},
        "age_band": {
            "18-24": 1.20,
            "25-34": 1.15,
            "35-44": 1.05,
            "45-54": 0.95,
            "55-64": 0.85,
            "65+": 0.75,
        },
        "region": {
            "North": 1.00,
            "South": 1.00,
            "East": 1.00,
            "West": 1.00,
            "Central": 1.00,
            "Rural/Other": 0.95,
        },
    },
    "male_skew": {
        "gender": {"Male": 1.55, "Female": 0.45},
        "age_band": {
            "18-24": 0.95,
            "25-34": 1.10,
            "35-44": 1.10,
            "45-54": 1.00,
            "55-64": 0.95,
            "65+": 0.85,
        },
        "region": {
            "North": 1.00,
            "South": 1.00,
            "East": 1.00,
            "West": 1.00,
            "Central": 1.00,
            "Rural/Other": 1.05,
        },
    },
    "young_skew": {
        "gender": {"Male": 1.00, "Female": 1.00},
        "age_band": {
            "18-24": 1.35,
            "25-34": 1.25,
            "35-44": 1.00,
            "45-54": 0.80,
            "55-64": 0.65,
            "65+": 0.50,
        },
        "region": {
            "North": 1.00,
            "South": 1.00,
            "East": 1.05,
            "West": 1.00,
            "Central": 1.00,
            "Rural/Other": 0.95,
        },
    },
    "older_skew": {
        "gender": {"Male": 1.00, "Female": 1.00},
        "age_band": {
            "18-24": 0.45,
            "25-34": 0.60,
            "35-44": 0.85,
            "45-54": 1.05,
            "55-64": 1.25,
            "65+": 1.45,
        },
        "region": {
            "North": 1.00,
            "South": 1.00,
            "East": 1.00,
            "West": 1.00,
            "Central": 1.00,
            "Rural/Other": 1.05,
        },
    },
    "parent_skew": {
        "gender": {"Male": 0.95, "Female": 1.05},
        "age_band": {
            "18-24": 0.50,
            "25-34": 1.55,
            "35-44": 1.45,
            "45-54": 0.90,
            "55-64": 0.50,
            "65+": 0.25,
        },
        "region": {
            "North": 1.00,
            "South": 1.00,
            "East": 1.00,
            "West": 1.00,
            "Central": 1.00,
            "Rural/Other": 1.05,
        },
    },
    "beauty_wellness_female": {
        "gender": {"Male": 0.50, "Female": 1.50},
        "age_band": {
            "18-24": 1.20,
            "25-34": 1.25,
            "35-44": 1.10,
            "45-54": 0.95,
            "55-64": 0.80,
            "65+": 0.65,
        },
        "region": {
            "North": 1.00,
            "South": 1.05,
            "East": 1.05,
            "West": 1.00,
            "Central": 1.00,
            "Rural/Other": 0.90,
        },
    },
    "auto_male": {
        "gender": {"Male": 1.45, "Female": 0.55},
        "age_band": {
            "18-24": 0.70,
            "25-34": 1.15,
            "35-44": 1.20,
            "45-54": 1.05,
            "55-64": 0.90,
            "65+": 0.75,
        },
        "region": {
            "North": 1.00,
            "South": 1.00,
            "East": 1.00,
            "West": 1.00,
            "Central": 1.00,
            "Rural/Other": 1.10,
        },
    },
    "finance_older": {
        "gender": {"Male": 1.10, "Female": 0.90},
        "age_band": {
            "18-24": 0.50,
            "25-34": 0.80,
            "35-44": 1.00,
            "45-54": 1.15,
            "55-64": 1.25,
            "65+": 1.20,
        },
        "region": {
            "North": 1.00,
            "South": 1.00,
            "East": 1.05,
            "West": 1.00,
            "Central": 1.00,
            "Rural/Other": 0.95,
        },
    },
    "outdoor_regional": {
        "gender": {"Male": 1.10, "Female": 0.90},
        "age_band": {
            "18-24": 0.90,
            "25-34": 1.00,
            "35-44": 1.05,
            "45-54": 1.05,
            "55-64": 1.00,
            "65+": 0.90,
        },
        "region": {
            "North": 0.95,
            "South": 1.00,
            "East": 0.95,
            "West": 1.05,
            "Central": 1.00,
            "Rural/Other": 1.20,
        },
    },
}

def assign_archetype(category_name: str) -> str:
    name = category_name.lower()

    if "female only" in name or "women" in name or "female" in name or "fertility" in name:
        if any(term in name for term in ["skincare", "self tan", "activewear", "fashion", "beauty", "wellness"]):
            return "beauty_wellness_female"
        return "female_skew"

    if "men" in name or "male only" in name:
        return "male_skew"

    if any(term in name for term in ["baby", "nappies", "child", "children"]):
        return "parent_skew"

    if any(term in name for term in ["funeral", "wealth", "term deposits"]):
        return "older_skew"

    if any(term in name for term in ["car ", "car&", "motorhome", "insurance", "modification", "rental"]):
        return "auto_male"

    if any(term in name for term in ["shares", "brokers", "funds", "wealth"]):
        return "finance_older"

    if any(term in name for term in ["camping", "outdoor", "blinds"]):
        return "outdoor_regional"

    if any(term in name for term in ["fast food", "betting", "rtd", "pre-mixed", "meal kit"]):
        return "young_skew"

    if any(term in name for term in ["skincare", "massage", "beauty", "wellness", "suncare"]):
        return "beauty_wellness_female"

    return "broad"

def soften_multiplier(raw_multiplier, incidence_rate, alpha=0.5):
    """
    Pull demographic multipliers toward 1 as category incidence rises.

    Parameters
    ----------
    raw_multiplier : float or np.ndarray
        Original demographic multiplier(s).
    incidence_rate : float
        Overall category incidence rate in [0, 1].
    alpha : float
        Controls how aggressively skew is softened for high-incidence categories.

    Returns
    -------
    float or np.ndarray
        Softened multiplier(s).
    """
    shrink = (1.0 - incidence_rate) ** alpha
    return 1.0 + shrink * (raw_multiplier - 1.0)


def calibrate_probabilities_to_target(raw_multiplier, cell_weight, target_incidence):
    """
    Convert multipliers into calibrated cell probabilities that preserve
    the population-weighted target incidence exactly, before any clipping.
    """
    weighted_mean_multiplier = np.sum(raw_multiplier * cell_weight)
    if weighted_mean_multiplier <= 0:
        raise ValueError("Weighted mean multiplier must be positive.")

    calibrated_prob = target_incidence * raw_multiplier / weighted_mean_multiplier
    return calibrated_prob

def build_cell_table(respondents: pd.DataFrame) -> pd.DataFrame:
    cell_table = (
        respondents.groupby(["gender", "age_band", "region"], as_index=False)
        .size()
        .rename(columns={"size": "n_cell"})
    )
    cell_table["cell_weight"] = cell_table["n_cell"] / cell_table["n_cell"].sum()
    return cell_table


def create_category_cell_probabilities(
    respondents: pd.DataFrame,
    categories: pd.DataFrame,
    max_prob: float = 0.995,
    alpha: float = 1.0,
) -> pd.DataFrame:
    cell_table = build_cell_table(respondents)
    outputs = []

    for _, row in categories.iterrows():
        category_id = row["category_id"]
        category_name = row["category_name"]
        incidence = row["incidence_rate"]

        archetype = assign_archetype(category_name)
        rules = ARCHETYPE_RULES[archetype]

        tmp = cell_table.copy()
        tmp["category_id"] = category_id
        tmp["category_name"] = category_name
        tmp["incidence_rate"] = incidence
        tmp["archetype"] = archetype

        # Raw multiplier from gender x age x region
        tmp["raw_multiplier"] = (
            tmp["gender"].map(rules["gender"])
            * tmp["age_band"].map(rules["age_band"])
            * tmp["region"].map(rules["region"])
        )

        # Soften skew for high-incidence categories
        tmp["soft_multiplier"] = soften_multiplier(
            raw_multiplier=tmp["raw_multiplier"].to_numpy(),
            incidence_rate=incidence,
            alpha=alpha,
        )

        # Calibrate so weighted average hits target incidence
        tmp["calibrated_probability"] = calibrate_probabilities_to_target(
            raw_multiplier=tmp["soft_multiplier"].to_numpy(),
            cell_weight=tmp["cell_weight"].to_numpy(),
            target_incidence=incidence,
        )

        # Optional safety clip
        tmp["calibrated_probability"] = tmp["calibrated_probability"].clip(0, max_prob)

        # Useful diagnostics
        tmp["max_probability_flag"] = (
            tmp["calibrated_probability"] >= max_prob
        ).astype(int)

        outputs.append(tmp)

    return pd.concat(outputs, ignore_index=True)

def attach_respondent_category_probabilities(
    respondents: pd.DataFrame,
    category_cell_probs: pd.DataFrame,
) -> pd.DataFrame:
    """
    Returns a long respondent-category table with respondent-specific qualification probability.
    """
    respondent_probs = respondents.merge(
        category_cell_probs[
            [
                "category_id",
                "category_name",
                "gender",
                "age_band",
                "region",
                "calibrated_probability",
            ]
        ],
        on=["gender", "age_band", "region"],
        how="left",
    )

    return respondent_probs.rename(columns={"calibrated_probability": "qualify_probability"})


def simulate_qualification_outcomes(
    respondent_category_probs: pd.DataFrame,
    random_state: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    df = respondent_category_probs.copy()

    df["qualified"] = (
        rng.random(len(df)) < df["qualify_probability"].to_numpy()
    ).astype(int)

    return df


'''
Run functions 
'''
respondents = pd.read_csv('../Data/synthetic_national_sample.csv')
categories = pd.read_csv("../Data/fake_category_data.csv")

category_cell_probs = create_category_cell_probabilities(
    respondents=respondents,
    categories=categories,
)

respondent_category_probs = attach_respondent_category_probabilities(
    respondents=respondents,
    category_cell_probs=category_cell_probs,
)

qualification_outcomes = simulate_qualification_outcomes(
    respondent_category_probs=respondent_category_probs,
    random_state=42,
)

print(category_cell_probs.head())
print(respondent_category_probs.head())
print(qualification_outcomes.head())

def validate_simulated_incidence(
    qualification_outcomes: pd.DataFrame,
    categories: pd.DataFrame,
) -> pd.DataFrame:
    sim = (
        qualification_outcomes.groupby(["category_id", "category_name"], as_index=False)["qualified"]
        .mean()
        .rename(columns={"qualified": "simulated_incidence"})
    )

    out = categories.merge(sim, on=["category_id", "category_name"], how="left")
    out["abs_error"] = (out["simulated_incidence"] - out["incidence_rate"]).abs()
    return out.sort_values("abs_error", ascending=False)

validated_results = validate_simulated_incidence(qualification_outcomes, categories)
print(validated_results[["category_id", "category_name", "incidence_rate", "simulated_incidence", "abs_error"]])    

validated_results.to_csv("../Reports/validated_category_incidence.csv", index=False)