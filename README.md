# Approximation of Measures to Lower Granularities

Code and data for the master's thesis *“Approximation of Measures to Lower Granularities”*
(Jiří Landsmann, 2025).

The task: an indicator is published for the 14 Czech regions (**Kraje**) and we want estimates for
the 77 districts (**Okresy**), using auxiliary data that is available at district level. Real
district values exist for both case studies and are used **only for evaluation**.

| Case study | Target | Auxiliary features | Averaging weight |
|---|---|---|---|
| `education/` | % of population with tertiary education (Census 2021) | household-composition shares | households |
| `unemployment/` | % unemployed | vote shares of political blocs (municipal elections) | households |

## Methods

Every estimate is **anchored**: within each Kraj, Okres predictions are rescaled so that their
weighted mean equals the official Kraj value.

* **Baselines** - *Kraj mean* (every Okres gets its region's value) and *direct* models fitted on the
  14 Kraj rows and applied to Okresy, raw and anchored.
* **Method A - noise simulation** (the original thesis method): Okresy are clustered within each Kraj,
  synthetic labels are the Kraj value plus cluster-level and district-level noise, a regressor is
  trained on Okres features → synthetic labels and its predictions are anchored. Repeated over
  50 random seeds.
* **Method B - feature-informed simulation**: synthetic labels come from a ridge regression fitted on
  the Kraj rows and anchored; the rest of the pipeline is the same.

Each is run with Linear Regression, Decision Tree, Random Forest and Gradient Boosting. Metrics (MAE,
RMSE, R², within-Kraj correlation) are computed on 76 districts; Prague is excluded because it is both
a Kraj and its only Okres.

## Layout

```
approx/                 shared code
  data.py               loading, name normalisation, Okres → Kraj map, feature construction
  methods.py            baselines, anchoring, clustering, Method A / B, simulation pipeline
  evaluation.py         metrics and result tables
  plotting.py           thesis figures
education/
  DATA/                 input data + DATA_CARD.md
  baselines.ipynb       baselines in detail
  education.ipynb       Method A and B, comparison with baselines
  results/              CSV outputs (predictions, metrics)
  plots/                PNG figures
unemployment/           same structure (unemployment.ipynb)
```

The unemployment case study also reads `education/DATA/domacnosti.xlsx` for its population weights.

## Running

```bash
pip install -r requirements.txt
jupyter lab
```

Run the notebooks from their own folder (`education/`, `unemployment/`); each notebook is
independent and writes its outputs to `results/` and `plots/`. All randomness is seeded, so
re-running reproduces the same numbers.
