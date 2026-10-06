"""Metrics against the real Okres values (never used for fitting)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import Dataset
from .methods import anchoring_error, kraj_mean

METRICS = ["MAE", "RMSE", "R2", "Within-Kraj r"]


def metrics(pred: pd.Series, ds: Dataset) -> dict:
    """MAE / RMSE / R2 on the evaluation districts (Prague excluded), plus

    * Within-Kraj r - correlation between true and predicted deviations from the
      Kraj value. This isolates what disaggregation is about: ranking districts
      *inside* a region. The Kraj-mean baseline scores NaN (no variation).
    * Anchor gap - largest |weighted Okres mean - Kraj value| (0 when anchored).
    """
    mask = ds.eval_mask
    y, p = ds.okres["y_true"][mask], pred[mask]
    err = p - y
    r2 = 1 - (err ** 2).sum() / ((y - y.mean()) ** 2).sum()
    base = kraj_mean(ds)[mask]
    dev_p = p - base
    within = np.corrcoef(y - base, dev_p)[0, 1] if dev_p.std() > 1e-9 else np.nan
    return {"MAE": err.abs().mean(), "RMSE": np.sqrt((err ** 2).mean()), "R2": r2,
            "Within-Kraj r": within, "Anchor gap": anchoring_error(pred, ds)}


def metrics_table(preds: dict, ds: Dataset, family: str | None = None) -> pd.DataFrame:
    """One row per method for single-run predictions {name: Series}."""
    rows = {name: metrics(p, ds) for name, p in preds.items()}
    table = pd.DataFrame(rows).T
    table.index.name = "Method"
    if family:
        table.insert(0, "Family", family)
    return table


def simulation_table(runs: dict, ds: Dataset, family: str) -> pd.DataFrame:
    """Metrics for simulation runs {name: DataFrame(okres x seed)}.

    Reports the mean over seeds and the standard deviation of MAE across seeds
    (how much the result depends on the random draw)."""
    rows = {}
    for name, frame in runs.items():
        per_seed = pd.DataFrame({s: metrics(frame[s], ds) for s in frame.columns}).T
        row = per_seed.mean()
        row["MAE sd"] = per_seed["MAE"].std(ddof=0) if len(per_seed) > 1 else 0.0
        rows[name] = row
    table = pd.DataFrame(rows).T
    table.index = [f"{family}: {name}" for name in table.index]
    table.index.name = "Method"
    table.insert(0, "Family", family)
    return table


def predictions_frame(ds: Dataset, preds: dict) -> pd.DataFrame:
    """Okres, Kraj, real value and one column per method (seed-averaged if needed)."""
    out = ds.okres[["kraj", "y_true"]].copy()
    for name, p in preds.items():
        out[name] = p.mean(axis=1) if isinstance(p, pd.DataFrame) else p
    out.index.name = "okres"
    return out


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #
def abs_errors(preds: dict, ds: Dataset) -> pd.DataFrame:
    """|prediction - truth| per evaluation district (rows) and method (columns).

    For simulation runs (DataFrame okres x seed) the error is averaged over seeds
    first; the mean of this column is then exactly the seed-averaged MAE."""
    mask = ds.eval_mask
    y = ds.okres["y_true"][mask]
    cols = {}
    for name, p in preds.items():
        if isinstance(p, pd.DataFrame):
            cols[name] = p[mask].sub(y, axis=0).abs().mean(axis=1)
        else:
            cols[name] = (p[mask] - y).abs()
    return pd.DataFrame(cols)


def _resample_index(ds: Dataset, index: pd.Index, rng, by: str) -> np.ndarray:
    """Positions of one bootstrap resample, by district or by whole Kraj."""
    if by == "okres":
        return rng.integers(0, len(index), len(index))
    kraj = ds.okres.loc[index, "kraj"].to_numpy()
    groups = [np.flatnonzero(kraj == k) for k in np.unique(kraj)]
    picked = rng.integers(0, len(groups), len(groups))
    return np.concatenate([groups[i] for i in picked])


def bootstrap(errors: pd.DataFrame, ds: Dataset, reference: str = "Kraj mean",
              n_boot: int = 2000, by: str = "okres", seed: int = 0) -> pd.DataFrame:
    """Paired bootstrap of MAE and of the MAE difference to ``reference``.

    Every resample draws the same districts for all methods, so differences are
    paired. ``by="kraj"`` resamples whole regions instead of districts - more
    conservative, because districts in the same region have correlated errors.

    Columns: MAE with 95% CI, difference to the reference (negative = better)
    with 95% CI, and the share of resamples in which the method beats the reference.
    """
    rng = np.random.default_rng(seed)
    E = errors.to_numpy()
    maes = np.empty((n_boot, E.shape[1]))
    for b in range(n_boot):
        maes[b] = E[_resample_index(ds, errors.index, rng, by)].mean(axis=0)
    ref = maes[:, errors.columns.get_loc(reference)][:, None]
    diff = maes - ref
    q = lambda a, p: np.percentile(a, p, axis=0)
    table = pd.DataFrame({
        "MAE": E.mean(axis=0),
        "MAE 2.5%": q(maes, 2.5), "MAE 97.5%": q(maes, 97.5),
        f"Δ vs {reference}": E.mean(axis=0) - E[:, errors.columns.get_loc(reference)].mean(),
        "Δ 2.5%": q(diff, 2.5), "Δ 97.5%": q(diff, 97.5),
        "P(better)": (diff < 0).mean(axis=0),
    }, index=errors.columns)
    table.index.name = "Method"
    return table.sort_values("MAE")


def pairwise(errors: pd.DataFrame, ds: Dataset, methods: list[str],
             n_boot: int = 2000, by: str = "okres", seed: int = 0) -> pd.DataFrame:
    """Paired bootstrap for every pair in ``methods``: MAE(a) - MAE(b) with 95% CI
    and the share of resamples in which a beats b."""
    rows = []
    for i, a in enumerate(methods):
        for b in methods[i + 1:]:
            t = bootstrap(errors[[a, b]], ds, reference=b, n_boot=n_boot, by=by, seed=seed).loc[a]
            rows.append({"A": a, "B": b, "MAE(A) - MAE(B)": t.iloc[3],
                         "2.5%": t["Δ 2.5%"], "97.5%": t["Δ 97.5%"], "P(A better)": t["P(better)"]})
    return pd.DataFrame(rows)
