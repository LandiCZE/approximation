"""Static figures for the thesis (matplotlib, saved as PNG)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import Dataset

INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e4e3df"
FAMILY_COLORS = {          # categorical slots 1-3 of the reference palette + neutral
    "Baseline": "#8a8985",
    "Method A": "#eb6834",
    "Method B": "#2a78d6",
}
POINT = "#2a78d6"
DIAGONAL = "#8a8985"

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 300, "font.size": 11,
    "axes.edgecolor": INK_MUTED, "axes.labelcolor": INK, "axes.titlesize": 12,
    "xtick.color": INK_MUTED, "ytick.color": INK_MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
})


def _save(fig, path: Path | None):
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
    return fig


def _grid(n: int):
    cols = 2
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(11, 4.6 * rows), squeeze=False)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    return fig, axes.flat


def prediction_grid(ds: Dataset, preds: dict, title: str, path: Path | None = None):
    """Actual vs predicted per method, with MAE / R2 annotation."""
    mask = ds.eval_mask
    y = ds.okres["y_true"][mask]
    lo, hi = y.min(), y.max()
    for p in preds.values():
        lo, hi = min(lo, p[mask].min()), max(hi, p[mask].max())
    pad = 0.05 * (hi - lo)
    fig, axes = _grid(len(preds))
    for ax, (name, p) in zip(axes, preds.items()):
        p = p[mask]
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=DIAGONAL, lw=1.2, ls="--")
        ax.scatter(y, p, s=36, color=POINT, alpha=0.8, edgecolor="white", linewidth=0.6)
        mae = (p - y).abs().mean()
        r2 = 1 - ((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        ax.text(0.04, 0.95, f"MAE {mae:.2f}\nR² {r2:.2f}", transform=ax.transAxes,
                va="top", color=INK, fontsize=11)
        ax.set(title=name, xlim=(lo - pad, hi + pad), ylim=(lo - pad, hi + pad),
               xlabel=f"Actual {ds.target_label}", ylabel=f"Predicted {ds.target_label}")
        ax.set_aspect("equal")
    fig.suptitle(title, fontsize=14, color=INK)
    fig.tight_layout()
    return _save(fig, path)


def error_grid(ds: Dataset, preds: dict, title: str, path: Path | None = None):
    """Histogram of signed errors (predicted - actual) per method, shared bins."""
    mask = ds.eval_mask
    y = ds.okres["y_true"][mask]
    errors = {name: (p[mask] - y) for name, p in preds.items()}
    span = max(e.abs().max() for e in errors.values())
    bins = np.linspace(-span, span, 25)
    fig, axes = _grid(len(preds))
    for ax, (name, e) in zip(axes, errors.items()):
        ax.hist(e, bins=bins, color=POINT, edgecolor="white", linewidth=1)
        ax.axvline(0, color=DIAGONAL, lw=1.2, ls="--")
        ax.text(0.04, 0.95, f"mean |e| {e.abs().mean():.2f}\nmedian |e| {e.abs().median():.2f}",
                transform=ax.transAxes, va="top", color=INK, fontsize=11)
        ax.set(title=name, xlabel="Prediction error (p.p.)", ylabel="Districts")
    fig.suptitle(title, fontsize=14, color=INK)
    fig.tight_layout()
    return _save(fig, path)


def comparison_bars(table: pd.DataFrame, title: str, path: Path | None = None):
    """Horizontal MAE bars, coloured by family, with seed sd as error bars and the
    Kraj-mean baseline as a reference line."""
    t = table.sort_values("MAE", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(t) + 1.4))
    colors = [FAMILY_COLORS[f] for f in t["Family"]]
    xerr = t["MAE sd"].where(t["MAE sd"] > 0) if "MAE sd" in t else None  # NaN = no bar
    ax.barh(t.index, t["MAE"], xerr=xerr, color=colors, height=0.65,
            error_kw={"ecolor": INK_MUTED, "elinewidth": 1, "capsize": 2})
    ref = table.loc["Kraj mean", "MAE"]
    ax.axvline(ref, color=INK, lw=1, ls=":")
    ax.text(ref, len(t) - 0.4, " Kraj mean", color=INK, fontsize=9, va="bottom")
    ends = t["MAE"] + (xerr.fillna(0) if xerr is not None else 0)
    for i, (v, end) in enumerate(zip(t["MAE"], ends)):
        ax.text(end, i, f"  {v:.3f}", va="center", color=INK_MUTED, fontsize=9)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("MAE (p.p.), lower is better")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in FAMILY_COLORS.values()]
    ax.legend(handles, FAMILY_COLORS.keys(), loc="upper left", bbox_to_anchor=(1.01, 1),
              frameon=False)
    ax.set_title(title, color=INK)
    fig.tight_layout()
    return _save(fig, path)


MODEL_COLORS = {  # categorical slots 1-4 of the reference palette, fixed order
    "Linear Regression": "#2a78d6",
    "Decision Tree": "#eb6834",
    "Random Forest": "#1baf7a",
    "Gradient Boosting": "#eda100",
}


def sensitivity_lines(sens: pd.DataFrame, reference_mae: float, chosen_cv: float,
                      title: str, path: Path | None = None):
    """Method A MAE vs noise level, one line per model, with a ±1 sd band."""
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for model, color in MODEL_COLORS.items():
        d = sens[sens["model"] == model].sort_values("cv")
        ax.fill_between(d["cv"], d["MAE"] - d["MAE sd"], d["MAE"] + d["MAE sd"],
                        color=color, alpha=0.15, linewidth=0)
        ax.plot(d["cv"], d["MAE"], color=color, lw=2, marker="o", ms=5, label=model)
    ax.axhline(reference_mae, color=INK, lw=1, ls=":")
    ax.text(sens["cv"].max(), reference_mae, "Kraj mean ", color=INK, fontsize=9,
            ha="right", va="bottom")
    ax.axvline(chosen_cv, color=INK_MUTED, lw=1, ls="--")
    ax.text(chosen_cv, ax.get_ylim()[1], " CV used", color=INK_MUTED, fontsize=9, va="top")
    ax.set(xlabel="Noise level CV (coefficient of variation of simulated labels)",
           ylabel="MAE (p.p.)", title=title)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    return _save(fig, path)
