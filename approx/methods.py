"""Disaggregation methods: Kraj-level values -> Okres-level estimates.

Baselines
    * ``kraj_mean``          - every Okres gets its Kraj value.
    * ``direct``             - a model fitted on the 14 Kraj rows, applied to Okresy.

Simulation pipeline (both variants share it)
    1. generate synthetic Okres labels that respect the Kraj value,
    2. fit a supervised model on Okres features -> synthetic labels,
    3. predict every Okres and re-anchor within each Kraj.

    * Method A, ``noise_labels``     - the original thesis method (fixed): labels are
      the Kraj value plus random cluster-level and district-level noise.
    * Method B, ``informed_labels``  - labels come from a ridge model fitted on Kraj
      rows, so they carry the feature signal instead of pure noise.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from .data import Dataset

RANDOM_STATE = 42


def make_models() -> "OrderedDict[str, object]":
    """The four regressors compared throughout, each with feature scaling."""
    return OrderedDict([
        ("Linear Regression", make_pipeline(StandardScaler(), LinearRegression())),
        ("Decision Tree", make_pipeline(StandardScaler(), DecisionTreeRegressor(random_state=RANDOM_STATE))),
        ("Random Forest", make_pipeline(StandardScaler(), RandomForestRegressor(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1))),
        ("Gradient Boosting", make_pipeline(StandardScaler(), GradientBoostingRegressor(random_state=RANDOM_STATE))),
    ])


# --------------------------------------------------------------------------- #
# Anchoring
# --------------------------------------------------------------------------- #
def anchor(pred: pd.Series, ds: Dataset) -> pd.Series:
    """Rescale predictions so that, within each Kraj, the weighted mean equals
    the official Kraj value (ratio adjustment, which keeps rates non-negative)."""
    pred = pred.clip(lower=0)
    okres = ds.okres
    w = okres["weight"]
    wmean = (pred * w).groupby(okres["kraj"]).sum() / w.groupby(okres["kraj"]).sum()
    target = ds.kraj["y"]
    ratio = (target / wmean).reindex(okres["kraj"]).to_numpy()
    out = pred * ratio
    # Degenerate case: all predictions in a Kraj are zero -> fall back to the Kraj value.
    flat = ~np.isfinite(ratio)
    out[flat] = target.reindex(okres["kraj"]).to_numpy()[flat]
    return out


def anchoring_error(pred: pd.Series, ds: Dataset) -> float:
    """Largest |weighted Okres mean - Kraj value| over all Kraje (0 when anchored)."""
    okres = ds.okres
    w = okres["weight"]
    wmean = (pred * w).groupby(okres["kraj"]).sum() / w.groupby(okres["kraj"]).sum()
    return float((wmean - ds.kraj["y"]).abs().max())


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #
def kraj_mean(ds: Dataset) -> pd.Series:
    return ds.okres["kraj"].map(ds.kraj["y"]).rename("Kraj mean")


def direct(ds: Dataset, model) -> pd.Series:
    """Fit on the 14 Kraj rows (features -> official value), predict the Okresy."""
    model.fit(ds.kraj[ds.features], ds.kraj["y"])
    return pd.Series(model.predict(ds.okres[ds.features]), index=ds.okres.index)


def run_baselines(ds: Dataset) -> "OrderedDict[str, pd.Series]":
    preds = OrderedDict([("Kraj mean", kraj_mean(ds))])
    for name, model in make_models().items():
        raw = direct(ds, model)
        preds[f"Direct {name}"] = raw
        preds[f"Direct {name} + anchor"] = anchor(raw, ds)
    return preds


# --------------------------------------------------------------------------- #
# Synthetic labels
# --------------------------------------------------------------------------- #
def cluster_okresy(ds: Dataset, max_clusters: int = 5) -> pd.Series:
    """K-means within each Kraj on standardised features.

    The number of clusters is chosen per Kraj by silhouette score (2 .. n-1),
    i.e. from how heterogeneous the *features* are. (The original notebook
    keyed it on the variance of the Okres target, which is unknown at
    prediction time - and was NaN in practice, so every Kraj got 7 clusters.)
    """
    scaled = pd.DataFrame(StandardScaler().fit_transform(ds.okres[ds.features]),
                          index=ds.okres.index)
    labels = pd.Series(0, index=ds.okres.index, name="cluster")
    for kraj, idx in ds.okres.groupby("kraj").groups.items():
        X = scaled.loc[idx]
        if len(X) < 3:
            continue
        best_k, best_score, best_labels = 1, -1.0, np.zeros(len(X), dtype=int)
        for k in range(2, min(max_clusters, len(X) - 1) + 1):
            km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X)
            score = silhouette_score(X, km.labels_)
            if score > best_score:
                best_k, best_score, best_labels = k, score, km.labels_
        labels.loc[idx] = best_labels
    return labels


def noise_labels(ds: Dataset, rng: np.random.Generator, clusters: pd.Series,
                 cv: float = 0.10, cluster_share: float = 0.5) -> pd.Series:
    """Method A: Kraj value x (1 + cluster offset + district noise), then anchored.

    ``cv`` is the total coefficient of variation; ``cluster_share`` is the part of
    the variance that is shared within a cluster.
    """
    base = kraj_mean(ds)
    sd_cluster = cv * np.sqrt(cluster_share)
    sd_okres = cv * np.sqrt(1 - cluster_share)
    keys = ds.okres["kraj"] + "/" + clusters.astype(str)
    offsets = {key: rng.normal(0, sd_cluster) for key in sorted(keys.unique())}
    shock = keys.map(offsets) + rng.normal(0, sd_okres, size=len(base))
    return anchor(base * (1 + shock), ds)


def informed_prior(ds: Dataset) -> tuple[pd.Series, object]:
    """Ridge regression fitted on the 14 Kraj rows (alpha by leave-one-out CV),
    applied to the Okresy. Returns the raw Okres prediction and the model."""
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 25)))
    model.fit(ds.kraj[ds.features], ds.kraj["y"])
    return pd.Series(model.predict(ds.okres[ds.features]), index=ds.okres.index), model


def informed_labels(ds: Dataset, rng: np.random.Generator, prior: pd.Series,
                    cv: float = 0.0) -> pd.Series:
    """Method B: anchored ridge prior, optionally with multiplicative noise."""
    noisy = prior * (1 + rng.normal(0, cv, size=len(prior))) if cv > 0 else prior
    return anchor(noisy, ds)


# --------------------------------------------------------------------------- #
# Simulation pipeline
# --------------------------------------------------------------------------- #
def run_simulation(ds: Dataset, label_fn: Callable[[np.random.Generator], pd.Series],
                   seeds) -> "OrderedDict[str, pd.DataFrame]":
    """For every seed: draw synthetic labels, fit each model on Okres features ->
    labels, predict in-sample, anchor. Returns {name: DataFrame(okres x seed)};
    the entry 'Simulated labels' holds the synthetic labels themselves."""
    seeds = list(seeds)
    out = OrderedDict((name, {}) for name in ["Simulated labels", *make_models()])
    X = ds.okres[ds.features]
    for seed in seeds:
        y_sim = label_fn(np.random.default_rng(seed))
        out["Simulated labels"][seed] = y_sim
        for name, model in make_models().items():
            model.fit(X, y_sim)
            out[name][seed] = anchor(pd.Series(model.predict(X), index=X.index), ds)
    return OrderedDict((name, pd.DataFrame(cols)) for name, cols in out.items())


def method_a_sensitivity(ds: Dataset, clusters: pd.Series, cvs, cluster_shares=(0.5,),
                         seeds=range(20)) -> pd.DataFrame:
    """Method A MAE over a grid of noise levels.

    ``cv`` = 0 means the labels are just the Kraj value (no noise at all), so the
    grid shows what the noise itself contributes. Returns one row per
    (cv, cluster_share, model) with the MAE averaged over seeds and its sd."""
    from .evaluation import metrics  # local import: evaluation imports this module

    rows = []
    for share in cluster_shares:
        for cv in cvs:
            runs = run_simulation(
                ds, lambda rng: noise_labels(ds, rng, clusters, cv=cv, cluster_share=share), seeds)
            for name, frame in runs.items():
                if name == "Simulated labels":
                    continue
                maes = [metrics(frame[s], ds)["MAE"] for s in frame.columns]
                rows.append({"cv": cv, "cluster_share": share, "model": name,
                             "MAE": np.mean(maes), "MAE sd": np.std(maes)})
    return pd.DataFrame(rows)
