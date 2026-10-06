# DATASET_CARD.md

## Unemployment and Voting Data for Okres-Level Approximation

## Purpose and Context
This dataset supports the empirical analysis of unemployment disaggregation conducted in the thesis *“Approximation of measures to lower granularities.”*  
Its objective is to approximate the **district-level (Okres)** unemployment rate using **regional (Kraj)** unemployment aggregates and **correlated auxiliary indicators**, primarily derived from electoral data.

The data combines official labor market statistics with political voting patterns, which have been shown to correlate spatially with socioeconomic indicators.  
The dataset enables supervised or semi-supervised learning of Okres-level unemployment distributions consistent with regional totals.

---

## Data Structure
| Level | Rows | Columns | Description |
|--------|-------|----------|-------------|
| Kraj | 14 | 3 | Aggregated unemployment indicators per region |
| Okres | 77 | 3 | District-level unemployment indicators (used for evaluation) |
| Combined (Kraj–Okres + Parties) | ~10 000 | 7 | Correlated auxiliary variables from voting data per district and party |

All datasets are merged via standardized territorial identifiers (`Kraj`, `Okres`) after normalization of text and diacritics.

---

## Variable Overview

| Variable ID | Type | Role | Description | Source |
|--------------|------|------|-------------|---------|
| `Kraj` | string | key | Regional name (NUTS3 level) | *nezamestnanost.csv* |
| `Okres` | string | key | District name (LAU1 level) | *Formatted_Okres_Data.csv* |
| `Podíl nezaměstnaných osob [%]` | float | **Target (y_target)** | Share of unemployed persons as percentage of total workforce | *Formatted_Okres_Data.csv*, *nezamestnanost.csv* |
| `Počet uchazečů na 1 VPM` | float | not used (at Okres level it is part of the ground truth) | Number of job applicants per one available job position | *Formatted_Okres_Data.csv*, *nezamestnanost.csv* |
| `Volební strana` | string | auxiliary | Political party name | *combined_with_kraj_okres.xlsx* |
| `Hlasy abs.` | string (int) | auxiliary | Absolute number of votes for a given party in the district | *combined_with_kraj_okres.xlsx* |
| `Hlasy v %` | float | auxiliary | Share of total votes received by the party in the district | *combined_with_kraj_okres.xlsx* |
| `Zastupitelé abs.` | int | auxiliary | Number of representatives elected for the party | *combined_with_kraj_okres.xlsx* |
| `Zastupitelé v %` | float | auxiliary | Share of elected representatives | *combined_with_kraj_okres.xlsx* |

---

## Example of Records (anonymized sample)

| Kraj | Okres | Podíl nezaměstnaných osob [%] | Počet uchazečů na 1 VPM | Volební strana | Hlasy v % |
|------|--------|------------------------------:|--------------------------:|----------------|-----------:|
| Jihočeský kraj | české budějovice | 2.4 | 0.6 | KDU-ČSL | 1.54 |
| Jihočeský kraj | český krumlov | 3.5 | 0.3 | Sociální demokracie | 0.11 |
| Hlavní město Praha | Praha | 3.1 | 0.4 | TOP 09 | 4.20 |

Each row represents one district–party combination used for generating correlated features and training fine-grained unemployment models.

---

## Anchoring Logic
The Kraj unemployment rate acts as an *aggregation anchor*. After each method predicts Okres values,
the predictions within each Kraj are rescaled (ratio adjustment) so that their **weighted mean** equals
the official Kraj value. The weight is the number of households per Okres (Census 2021, taken from
`education/DATA/domacnosti.xlsx`) as a population proxy.

Vote counts are **not** usable as weights or raw features: the voting file contains municipal
elections, where each voter casts as many votes as there are council seats (Prague alone has ~36M
votes). The ~390 local lists are therefore grouped into political blocs by keyword (ANO, SPD & allies,
left, SPOLU parties, Pirates/STAN, independents, other) and used as **vote shares** per Okres.

Known data issue handled in code: *Praha-východ* and *Praha-západ* appear in the voting file as
`Hlavní město Prahavychod` / `Prahazapad` under Prague; they are mapped to Středočeský kraj.
The implementation is in `approx/data.py` and `approx/methods.py`.

---

## Notes
- Data sources are derived from the **Czech Statistical Office (ČSÚ)** and official **State Election Commission (Volby.cz)** results.  
- All identifiers were standardized by removing diacritics and converting to lowercase.  
- Sensitive or personal data were excluded; only aggregate indicators were used.  
- Percentages are expressed as absolute percentage points (0–100).  
- The dataset covers the most recent comparable year available across all sources (2021).

---

## License and Citation
The dataset is based on publicly available data released under open statistical use conditions by:

> Czech Statistical Office (2021). *Unemployment Statistics by Region and District.*  
> Czech Statistical Office. *Municipal Elections – Official Results* (volby.cz).  
> Data accessed through https://www.czso.cz and https://www.volby.cz.
