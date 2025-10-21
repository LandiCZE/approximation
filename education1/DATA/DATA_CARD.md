# DATASET_CARD.md

## Title
**Education and Household Structure Data for Okres-Level Approximation**

## Purpose and Context
This dataset is used for the empirical part of the thesis *“Approximation of measures to lower granularities.”*  
Its goal is to enable the approximation of educational attainment indicators from higher (Kraj) to lower (Okres) territorial levels using correlated auxiliary data. The dataset integrates official Czech statistical sources, combining information on the share of higher-educated population with structural household characteristics.

Each row represents one territorial unit (Okres or Kraj) and contains identifiers, target indicators, and predictive features used in the modeling and disaggregation pipeline.

---

## Data Structure
| Level | Rows | Columns | Description |
|--------|-------|----------|-------------|
| Kraj | 14 | 2 | Regional aggregates used as anchors (share of higher education in %) |
| Okres | 77 | 2 | District-level data with target indicator (share of higher education in %) |
| Household statistics | 92 | 6 | Household composition per territorial unit used as predictive input |

All datasets are merged based on the common territorial name (`region` / `Unnamed: 0`), with harmonization of naming differences across sources.

---

## Variable Overview

| Variable ID | Type | Role | Description | Source |
|--------------|------|------|-------------|---------|
| `region` | string | key | Territorial unit name (Okres or Kraj) | all tables |
| `% vyšší vzdělání` | float | **Target (y_target)** | Share of population with tertiary education or higher (%) | *vzdelani_with_percentage_only.xlsx* |
| `anchor_kraj_y` | float | **Anchor** | Regional (Kraj-level) mean share of higher education used as an aggregation constraint | *krajske_vzdelani_percentage_only.xlsx* |
| `celkem` | int | control | Total number of households | *domacnosti.xlsx* |
| `1 rodinou` | int | predictor | Households with one family | *domacnosti.xlsx* |
| `2 a více rodinami` | int | predictor | Households with two or more families | *domacnosti.xlsx* |
| `domácnosti jednotlivců` | int | predictor | One-person households | *domacnosti.xlsx* |
| `vícečlenné domácnosti` | int | predictor | Multi-person households | *domacnosti.xlsx* |

---

## Example of Records (anonymized sample)

| region | % vyšší vzdělání | anchor_kraj_y | celkem | 1 rodinou | 2 a více rodinami | domácnosti jednotlivců | vícečlenné domácnosti |
|--------|-----------------:|---------------:|---------:|------------:|------------------:|-----------------------:|-----------------------:|
| Praha | 33.70 | 33.70 | 658 001 | 323 764 | 4 840 | 308 279 | 21 118 |
| Benešov | 14.34 | 16.94 | 43 227 | 26 652 | 886 | 15 109 | 580 |
| Beroun | 16.65 | 16.94 | 42 130 | 26 235 | 811 | 14 428 | 656 |
| Kladno | 14.63 | 16.94 | 64 111 | 37 865 | 1 135 | 24 252 | 859 |
| Příbram | 12.58 | 16.94 | 58 812 | 34 477 | 1 045 | 22 516 | 774 |

---

## Anchoring Logic
Kraj-level values of the educational indicator (`anchor_kraj_y`) act as *aggregation anchors* for disaggregation toward Okres-level estimates.  
The anchoring ensures that the sum of approximated Okres-level estimates within each Kraj respects the known aggregate mean.  
Household structure variables are used as correlated predictors to distribute the regional totals proportionally to local demographic characteristics, improving spatial realism and internal consistency of the resulting estimates.

---

## Notes
- All data originate from the Czech Statistical Office (Census 2021).  
- Sensitive or identifying information has been removed; only aggregate indicators are retained.  
- Percentages are expressed in absolute percentage points (0–100).  
- Household counts are integer values representing the number of households per territorial unit.  
- No individual-level or microdata are included.

---

## License and Citation
This dataset is based on publicly available statistical data provided by the **Czech Statistical Office (ČSÚ)** and can be reused for academic and non-commercial research purposes with proper citation:

> Czech Statistical Office (2021). *Census 2021 – Population and Housing.* Retrieved from https://www.czso.cz/

