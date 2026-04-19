# Survey Optimisation – Tracksuit Take-Home

## Overview

This project explores how to design survey structures that minimise the number of respondents required to deliver a fixed number of qualified completes per category, while respecting:

- ~200 qualified respondents per category per month  
- a maximum average interview length of 480 seconds  
- nationally representative exposure to category qualifiers  

The core challenge is balancing:
- varying category incidence rates  
- differing survey lengths  
- respondent time constraints  
- probabilistic qualification  

---

## Repository Structure

.
│   README.md
│
├───Code
│       BundleSearch.py
│       BundleSizeAnalysis.py
│       CellLevelIncidence.py
│       CreateSample.py
│       EvaluateBaselines.py
│       RealisticConstraints.py
│
├───Data
│       category_cell_probabilities.csv
│       fake_category_data.csv
│       synthetic_national_sample.csv
│
└───Reports
        bundle_search_candidate_summary.csv
        bundle_search_results.csv
        bundle_search_strategy_summary.csv
        bundle_search_top_candidates.csv
        bundle_size_sweep_results.csv
        bundle_size_sweep_summary.csv
        dev_results.csv
        dev_summary.csv
        holdout_results.csv
        holdout_summary.csv
        realistic_dev_results.csv
        realistic_dev_summary.csv
        realistic_holdout_results.csv
        realistic_holdout_summary.csv
        realistic_test_results.csv
        realistic_test_summary.csv
        test_results.csv
        test_summary.csv
        validated_category_incidence.csv


---

## Approach

The solution is developed through a sequence of structured experiments.

### 1. Synthetic respondent generation (`CreateSample.py`)
A nationally representative dataset is generated across:
- gender  
- age band  
- region  

This forms the base population for simulation.

---

### 2. Cell-level qualification modelling (`CellLevelIncidence.py`)
Category incidence rates are extended into a **demographic-aware model**.

This step:
- introduces demographic skews per category  
- generates qualification probabilities at the **(category × demographic cell)** level  
- recalibrates probabilities to match the original category incidence rates  

This enables realistic variation in qualification behaviour.

---

### 3. Incidence validation
The simulated qualification process is validated against target incidence rates.

Results:
- `Reports/validated_category_incidence.csv`

---

### 4. Baseline policy evaluation (`EvaluateBaselines.py`)
Initial strategies are compared:

- naive single-category surveys  
- one-survey-per-respondent with full screening  
- multi-survey routing (respondents can complete multiple categories)  

---

### 5. Realistic constraints (`RealisticConstraints.py`)
The model is extended to reflect real-world limitations:

- qualifier time cost  
- limited number of qualifiers per respondent  
- time-constrained survey routing  

---

### 6. Bundle size analysis (`BundleSizeAnalysis.py`)
The number of qualifiers shown per respondent is varied.

This evaluates the trade-off between:
- screening efficiency  
- respondent burden  

---

### 7. Bundle composition search (`BundleSearch.py`) *(Exploratory)*
Different ways of grouping categories into bundles are tested:

- random assignment  
- difficulty-balanced grouping  
- similarity-based grouping  

This step explores whether category grouping materially impacts efficiency.

---

## Key Results

### Respondent efficiency

| Policy | Respondents Required |
|------|---------------------|
| Naive (1 category per respondent) | ~40,000 |
| Single survey (full screener) | ~15,000 |
| Multi-survey (ideal) | ~3,800 |
| Multi-survey (realistic) | ~5,000 |

---

### Bundle size impact

Efficiency improves as more qualifiers are shown:

| Avg qualifiers shown | Respondents |
|--------------------|------------|
| ~13 | ~12,000–13,000 |
| ~19 | ~7,500–9,500 |
| ~26 | ~5,900–6,700 |

This reflects improved **respondent reuse**.

---

### Bundle composition

- Different bundle strategies produced similar results  
- Random assignment performed competitively  
- No heuristic clearly dominated  

---
## Final Policy Recommendation

Based on the analysis, the recommended survey design is:

- Respondents are assigned to one of 3–4 predefined survey bundles
- Each bundle contains approximately 20–25 category qualifiers
- Respondents complete all categories they qualify for, subject to:
  - a maximum interview time of 480 seconds
- Categories are prioritised by lowest incidence rate to ensure difficult categories are filled first

This structure:

- minimises total respondents required (~6k–8k)
- ensures all categories reach ~200 completes
- maintains mean interview length well below 480 seconds
- preserves nationally representative exposure to category qualifiers

Bundle composition was explored but found to have a secondary impact under current assumptions, so a simple random or balanced allocation is sufficient.
---

## Key Insights

### 1. Respondent reuse is the dominant driver of efficiency
Allowing respondents to complete multiple category surveys dramatically reduces required sample size.

---

### 2. Qualifier depth is the second most important factor
Showing more qualifiers increases the likelihood of reuse, leading to fewer required respondents.

---

### 3. Bundle composition is a secondary effect
Under the current simulation assumptions, grouping categories has limited impact relative to structural decisions.

---

### 4. Efficiency gains exhibit diminishing returns
Increasing qualifiers yields stepwise improvements rather than continuous gains.

---

## Limitations

- Category qualification is conditionally independent given demographics  
- Real-world co-qualification patterns are not captured  
- Respondent fatigue and drop-off are not explicitly modelled  
- Bundle composition effects may be understated due to synthetic assumptions  

---

## Next Steps

- Incorporate real co-qualification data  
- Model respondent fatigue and drop-off  
- Explore adaptive or dynamic survey routing  
- Apply more advanced optimisation techniques for bundle construction  

---

## How to Run

Run scripts in order:

```bash
python Code/CreateSample.py
python Code/CellLevelIncidence.py
python Code/EvaluateBaselines.py
python Code/RealisticConstraints.py
python Code/BundleSizeAnalysis.py
python Code/BundleSearch.py