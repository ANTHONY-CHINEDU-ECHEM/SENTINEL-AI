"""Report charts. Every figure is drawn from files written by the pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch  # noqa: E402

from ..features.catalog import FEATURE_LABELS  # noqa: E402
from ..features.extractor import FEATURE_GROUPS  # noqa: E402
from ..schema import SCENARIO_TITLES, SCENE_TITLES, Scenario, Scene  # noqa: E402
from ..simulation.behaviours import REGISTRY  # noqa: E402

INK, PANEL, GRID = "#0a0e16", "#111826", "#263248"
TEXT, MUTED = "#e2e8f0", "#8c98ac"
MINT, AMBER, RED, GREEN, VIOLET, SKY = "#5eead4", "#fbbf24", "#f87171", "#4ade80", "#a78bfa", "#7dd3fc"
GROUP_COLOURS = {"pose": SKY, "item_flow": AMBER, "pos": MINT, "context": VIOLET}
GROUP_TITLES = {"pose": "Pose", "item_flow": "Item flow", "pos": "Point of sale", "context": "Context"}
CMAP = LinearSegmentedColormap.from_list("sentinel", [PANEL, "#1f5f66", MINT])
SHORT = {
    "Normal shopping": "Normal",
    "Concealment in a bag": "Bag",
    "Concealment in clothing": "Clothing",
    "Shelf sweep": "Sweep",
    "Self checkout skip scan": "Skip\nscan",
    "Ticket switch": "Ticket\nswitch",
    "Sweethearting at the till": "Sweet\nhearting",
    "Trolley push out": "Push\nout",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": INK,
            "axes.facecolor": PANEL,
            "savefig.facecolor": INK,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT,
            "axes.titlecolor": TEXT,
            "text.color": TEXT,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "grid.color": GRID,
            "axes.grid": True,
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 14,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )


def _save(fig, path: Path, note: str | None = None) -> None:
    if note:
        fig.text(0.012, -0.035, note, color=MUTED, fontsize=8.5, ha="left", va="top")
    fig.savefig(path, dpi=160, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


NOTE = "Shelf Sentinel AI. Synthetic corpus, held out stores, metrics weighted to deployment prevalence."


def chart_dataset(tables: Path, out: Path, metrics: dict) -> None:
    df = pd.read_csv(tables / "dataset_composition.csv")
    df["theft"] = df["scenario"] != 0
    df = df.sort_values(["scene", "theft", "clips"], ascending=[True, True, True]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(11, 10))
    y = np.arange(len(df)) + df["scene"].to_numpy() * 0.9
    ax.barh(y, df["clips"], color=np.where(df["theft"], AMBER, SKY), height=0.72)
    ax.set_yticks(y, [REGISTRY[s].description for s in df["subtype"]], fontsize=9.5)
    for yi, n in zip(y, df["clips"], strict=True):
        ax.text(n + df["clips"].max() * 0.008, yi, f"{n:,}", va="center", fontsize=8.5, color=MUTED)
    for scene, grp in df.groupby("scene"):
        ax.text(df["clips"].max() * 1.0, y[grp.index].max() + 0.1, SCENE_TITLES[Scene(int(scene))].upper(), color=TEXT, fontsize=10, fontweight="bold", ha="right", va="center")
    ax.set_xlabel("Track clips")
    ax.grid(axis="y", visible=False)
    d = metrics["dataset"]
    ax.set_title(f"{d['clips']:,} track clips across {d['stores']} stores: {d['benign_subtypes']} benign behaviours and 7 theft scenarios")
    ax.legend(handles=[Patch(color=SKY, label="Benign behaviour"), Patch(color=AMBER, label="Theft scenario")], loc="lower right")
    _save(fig, out / "dataset_composition.png", "Shelf Sentinel AI. Theft is enriched in the corpus for learning and reweighted to a realistic base rate for evaluation.")


def chart_confusion(out: Path, metrics: dict) -> None:
    cm = np.array(metrics["model"]["classification"]["confusion_matrix"], dtype=float)
    norm = cm / cm.sum(axis=1, keepdims=True)
    names = [SHORT[SCENARIO_TITLES[Scenario(k)]] for k in range(len(cm))]
    fig, ax = plt.subplots(figsize=(9.5, 8))
    ax.imshow(norm, cmap=CMAP, vmin=0, vmax=1)
    for i in range(len(cm)):
        for j in range(len(cm)):
            if norm[i, j] >= 0.005:
                ax.text(j, i, f"{norm[i, j] * 100:.1f}", ha="center", va="center", fontsize=10, color=INK if norm[i, j] > 0.55 else TEXT, fontweight="bold" if i == j else "normal")
    ax.set_xticks(range(len(cm)), names, fontsize=9)
    ax.set_yticks(range(len(cm)), names, fontsize=9)
    ax.set_xlabel("Predicted scenario")
    ax.set_ylabel("True scenario")
    ax.grid(False)
    c = metrics["model"]["classification"]
    ax.set_title(f"Where the detector is right and wrong (row percent)\nMacro F1 {c['macro_f1']:.3f} on {int(cm.sum()):,} clips from stores never seen in training")
    _save(fig, out / "confusion_matrix.png", "Shelf Sentinel AI. Synthetic corpus, held out stores, most likely class per clip.")


def chart_model_comparison(out: Path, metrics: dict) -> None:
    rules = metrics["rules"]["alerts"]
    logit = metrics["matched_alert_budget"]["logistic"]
    model = metrics["model"]["review_tier"]
    names = ["Hand written\nrules", "Logistic\nregression", "Shelf\nSentinel"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8), gridspec_kw={"wspace": 0.32})
    for ax, key, title, fmt in (
        (axes[0], "precision", "Precision: share of alerts that are real", "{:.0%}"),
        (axes[1], "recall", "Recall: share of theft clips flagged", "{:.0%}"),
        (axes[2], "false_alerts_per_10k_normal", "False alerts per 10,000 honest clips", "{:.1f}"),
    ):
        vals = [v or 0.0 for v in (rules[key], logit[key], model[key])]  # None when a detector raised no alerts
        bars = ax.bar(names, vals, color=[MUTED, SKY, MINT], width=0.62)
        for b, v in zip(bars, vals, strict=True):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt.format(v), ha="center", va="bottom", fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=11.5)
        ax.tick_params(axis="x", labelsize=10)
        ax.margins(y=0.12)
        ax.grid(axis="x", visible=False)
        if key != "false_alerts_per_10k_normal":
            ax.set_ylim(0, 1.12)
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    fig.suptitle("The learned detector against two baselines on stores it has never seen", x=0.012, y=1.06, ha="left", fontsize=15, fontweight="bold")
    _save(fig, out / "model_comparison.png", NOTE + " Logistic regression is given the same alert budget as Shelf Sentinel.")


def chart_per_scenario(out: Path, metrics: dict) -> None:
    model = metrics["model"]["review_tier"]["per_class"]
    rules = metrics["rules"]["alerts"]["per_class"]
    names = list(model)
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5), gridspec_kw={"wspace": 0.18})
    for ax, key, title in ((axes[0], "recall", "Recall by scenario"), (axes[1], "precision", "Precision by scenario")):
        r = [rules[n][key] or 0 for n in names]
        m = [model[n][key] or 0 for n in names]
        ax.bar(x - 0.2, r, width=0.38, color=MUTED)
        ax.bar(x + 0.2, m, width=0.38, color=MINT)
        for xi, v in zip(x + 0.2, m, strict=True):
            ax.text(xi, v + 0.015, f"{v:.0%}", ha="center", fontsize=9, fontweight="bold")
        ax.set_xticks(x, [SHORT[n] for n in names], fontsize=9)
        ax.set_ylim(0, 1.14)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
    fig.legend(
        handles=[Patch(color=MUTED, label="Hand written rules"), Patch(color=MINT, label="Shelf Sentinel (review tier)")],
        loc="upper right", bbox_to_anchor=(0.9, 1.04), ncol=2,
    )
    _save(fig, out / "per_scenario_performance.png", NOTE)


def chart_ablation(tables: Path, out: Path) -> None:
    df = pd.read_csv(tables / "sensor_ablation.csv")
    cols = [c for c in df.columns if c.startswith("recall_")]
    mat = df[cols].to_numpy()
    fig, ax = plt.subplots(figsize=(12.5, 4.2))
    ax.imshow(mat, cmap=CMAP, vmin=0, vmax=1, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.0%}", ha="center", va="center", fontsize=12, fontweight="bold", color=INK if mat[i, j] > 0.55 else TEXT)
    ax.set_xticks(range(len(cols)), [SHORT[c.replace("recall_", "")] for c in cols], fontsize=9.5)
    ax.set_yticks(range(len(df)), [f"{r.features}\nmacro F1 {r.macro_f1:.2f}" for r in df.itertuples()], fontsize=10)
    ax.grid(False)
    ax.set_title("What each sensor family buys: recall by scenario as data sources are added")
    _save(fig, out / "sensor_ablation.png", "Shelf Sentinel AI. Each row is a separate model trained on a growing feature set and tested on held out stores.")


def chart_importance(tables: Path, out: Path) -> None:
    df = pd.read_csv(tables / "feature_importance.csv").head(16).iloc[::-1]
    group_of = {n: g for g, names in FEATURE_GROUPS.items() for n in names}
    fig, ax = plt.subplots(figsize=(10.5, 6.6))
    ax.barh([FEATURE_LABELS[f][0] for f in df["feature"]], df["importance"], color=[GROUP_COLOURS[group_of[f]] for f in df["feature"]], height=0.7)
    ax.set_xlabel("Increase in log loss when the feature is shuffled")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[Patch(color=c, label=GROUP_TITLES[g]) for g, c in GROUP_COLOURS.items()], loc="lower right", title="Sensor family")
    ax.set_title("The signals the detector leans on most")
    _save(fig, out / "feature_importance.png", "Shelf Sentinel AI. Permutation importance on held out stores.")


def chart_false_alerts(tables: Path, out: Path) -> None:
    df = pd.read_csv(tables / "false_alert_sources.csv").head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    bars = ax.barh([REGISTRY[s].description for s in df["subtype"]], df["rate_per_1000"], color=AMBER, height=0.66)
    for b, r in zip(bars, df.itertuples(), strict=True):
        ax.text(b.get_width() + df["rate_per_1000"].max() * 0.012, b.get_y() + b.get_height() / 2, f"{r.rate_per_1000:.1f}   ({int(r.false_alerts)} of {int(r.clips):,})", va="center", fontsize=9.5, color=MUTED)
    ax.set_xlim(0, df["rate_per_1000"].max() * 1.28)
    ax.set_xlabel("False alerts per 1,000 clips of that behaviour")
    ax.grid(axis="y", visible=False)
    ax.set_title("Which honest behaviours still trigger an alert")
    _save(fig, out / "false_alert_sources.png", "Shelf Sentinel AI. Review tier alerts on benign clips from held out stores.")


def chart_slices(tables: Path, out: Path) -> None:
    df = pd.read_csv(tables / "robustness_slices.csv")
    order = {"camera_quality": ["hd", "sd", "low_light"], "crowding": ["quiet", "busy", "crowded"], "store_format": ["convenience", "supermarket", "superstore"]}
    titles = {"camera_quality": "Camera quality", "crowding": "Crowding", "store_format": "Store format"}
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), gridspec_kw={"wspace": 0.22})
    for ax, (dim, names) in zip(axes, order.items(), strict=True):
        sub = df[df["dimension"] == dim].set_index("slice").reindex(names).dropna(subset=["clips"])
        labels = [n.replace("_", " ") for n in sub.index]
        bars = ax.bar(labels, sub["theft_recall"], color=MINT, width=0.58)
        for b, r in zip(bars, sub.itertuples(), strict=True):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.012, f"{r.theft_recall:.0%}", ha="center", fontsize=12, fontweight="bold")
            ax.text(b.get_x() + b.get_width() / 2, 0.04, f"{r.false_alerts_per_1000_normal:.1f} false\nper 1,000", ha="center", fontsize=8.5, color=INK)
        ax.set_ylim(0, 1.12)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.set_title(titles[dim], fontsize=12)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Theft recall at the review tier")
    fig.suptitle("Robustness: does performance hold in the hard conditions?", x=0.012, y=1.06, ha="left", fontsize=15, fontweight="bold")
    _save(fig, out / "robustness_slices.png", "Shelf Sentinel AI. Held out stores.")


def chart_economics(tables: Path, out: Path, metrics: dict) -> None:
    df = pd.read_csv(tables / "economics_curve.csv")
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    ax.plot(df["alerts_per_store_day"], df["net_value_per_store_year_gbp"] / 1000, color=MINT, lw=2.5, label="Shelf Sentinel, one global threshold")
    pts = (
        ("Review tier", metrics["model"]["economics_review"], AMBER),
        ("Priority tier", metrics["model"]["economics_priority"], RED),
        ("Hand written rules", metrics["rules"]["economics"], MUTED),
    )
    offsets = {"Review tier": (26, 6), "Priority tier": (-8, -44), "Hand written rules": (-150, 14)}
    for name, e, col in pts:
        xy = (e["alerts_per_store_day"], e["net_value_per_store_year_gbp"] / 1000)
        ax.scatter(*xy, s=130, color=col, zorder=5, edgecolor=INK, linewidth=1.5)
        ax.annotate(f"{name}: {e['alerts_per_store_day']:.1f} alerts a day", xy, xytext=offsets[name], textcoords="offset points", fontsize=10, color=col, fontweight="bold")
    ax.set_xlabel("Alerts sent for review per store per day")
    ax.set_ylabel("Net value per store per year (thousand pounds)")
    ax.set_xlim(0, max(df["alerts_per_store_day"].quantile(0.97), metrics["rules"]["economics"]["alerts_per_store_day"] * 1.25))
    lo = min(metrics["rules"]["economics"]["net_value_per_store_year_gbp"], df["net_value_per_store_year_gbp"].quantile(0.05)) / 1000
    ax.set_ylim(lo - 6, df["net_value_per_store_year_gbp"].max() / 1000 + 5)
    ax.set_title("Reviewer workload against value: more alerts stop paying for themselves")
    ax.legend(loc="lower left")
    _save(fig, out / "economics_curve.png", "Shelf Sentinel AI. Illustrative cost and recovery assumptions from configs/default.yaml; replace with figures from your own estate.")


def chart_prevalence(tables: Path, out: Path, metrics: dict) -> None:
    df = pd.read_csv(tables / "precision_vs_prevalence.csv")
    dep = metrics["config"]["policy"]["deployment_theft_prevalence"]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.plot(df["prevalence"] * 100, df["precision"], color=MINT, lw=2.5, marker="o", ms=7)
    for r in df.itertuples():
        ax.annotate(f"{r.precision:.0%}", (r.prevalence * 100, r.precision), xytext=(0, 10), textcoords="offset points", ha="center", fontsize=9.5)
    ax.axvline(dep * 100, color=AMBER, ls="--", lw=1.5)
    ax.text(dep * 100 * 1.08, 0.08, f"deployment assumption\n{dep:.1%} of clips", color=AMBER, fontsize=9.5)
    ax.set_xscale("log")
    ax.set_xticks(df["prevalence"] * 100, [f"{p * 100:g}%" for p in df["prevalence"]])
    ax.set_ylim(0, 1.1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("Share of track clips that contain theft (log scale)")
    ax.set_ylabel("Precision of review alerts")
    ax.set_title("The base rate problem: the same detector, the same thresholds, different stores")
    _save(fig, out / "precision_vs_prevalence.png", "Shelf Sentinel AI. Thresholds fixed at the deployment assumption; held out stores.")


def chart_calibration(tables: Path, out: Path, metrics: dict) -> None:
    df = pd.read_csv(tables / "calibration.csv")
    cal = metrics["model"]["calibration"]
    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    ax.plot([0, 1], [0, 1], color=MUTED, ls="--", lw=1.2, label="Perfect calibration")
    final = df[df["stage"] == "final"]
    ax.plot(final["predicted"], final["observed"], color=MINT, lw=2.5, marker="o", ms=7, label=f"Shelf Sentinel (ECE {cal['test_ece_final']:.4f})")
    ax.set_xlabel("Predicted probability of theft")
    ax.set_ylabel("Observed share of theft")
    ax.set_title("Can the risk score be read as a probability?")
    ax.legend(loc="upper left")
    _save(fig, out / "calibration.png", "Shelf Sentinel AI. Held out stores at deployment prevalence.")


def chart_latency(tables: Path, out: Path) -> None:
    df = pd.read_csv(tables / "detection_latency.csv")
    fig, ax = plt.subplots(figsize=(11, 4.8))
    x = np.arange(len(df))
    ax.bar(x, df["median_latency_s"], color=np.where(df["median_latency_s"] <= 0, GREEN, MINT), width=0.6)
    ax.scatter(x, df["p90_latency_s"], color=AMBER, zorder=5, s=60, label="90th percentile")
    ax.axhline(0, color=TEXT, lw=1)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo - 1.3, hi + 0.5)
    for xi, r in zip(x, df.itertuples(), strict=True):
        ax.text(xi + 0.34, r.median_latency_s, f"{r.median_latency_s:+.1f} s", ha="left", va="center", fontsize=10.5, fontweight="bold")
        ax.text(xi, lo - 1.2, f"{r.detected_in_clip:.0%} caught\nin clip", ha="center", va="bottom", fontsize=8.5, color=MUTED)
    ax.set_xlim(-0.6, len(df) - 0.1)
    ax.set_xticks(x, [SHORT[s].replace("\n", " ") for s in df["scenario"]], fontsize=9.5)
    ax.set_ylabel("Seconds from the theft event to the first alert")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right")
    ax.set_title("How fast the alert fires in streaming replay (negative means before the act completes)")
    _save(fig, out / "detection_latency.png", "Shelf Sentinel AI. Fresh clips from held out stores, replayed frame by frame.")


def chart_streaming(out: Path, metrics: dict) -> None:
    naive, aware = metrics["streaming"]["whole_clip_only_model"], metrics["streaming"]["streaming_aware_model"]
    labels = [
        "Trained on whole clips,\nscored live",
        "Trained on whole clips,\nlive + confirmation",
        "Streaming aware,\nscored live",
        "Streaming aware,\nlive + confirmation",
        "Scored once the\nclip is complete",
    ]
    values = [
        naive["streaming_false_alerts_unconfirmed_per_1000"],
        naive["streaming_false_alerts_per_1000"],
        aware["streaming_false_alerts_unconfirmed_per_1000"],
        aware["streaming_false_alerts_per_1000"],
        aware["whole_clip_false_alerts_per_1000"],
    ]
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    bars = ax.bar(labels, values, color=[RED, AMBER, SKY, MINT, GREEN], width=0.62)
    for b, v in zip(bars, values, strict=True):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + max(values) * 0.015, f"{v:.1f}", ha="center", fontsize=13, fontweight="bold")
    ax.set_ylabel("False alerts per 1,000 honest clips")
    ax.set_ylim(0, max(values) * 1.14)
    ax.tick_params(axis="x", labelsize=9.5)
    ax.grid(axis="x", visible=False)
    ax.set_title("Scoring a clip while it unfolds is a different problem from scoring it afterwards")
    _save(fig, out / "streaming_false_alerts.png", f"Shelf Sentinel AI. {aware['clips']:,} fresh clips from held out stores, replayed frame by frame.")


def make_all_charts(root: str | Path = ".") -> list[Path]:
    """Draw every report chart. Returns the files written."""
    root = Path(root)
    tables, out = root / "reports" / "tables", root / "docs" / "images"
    out.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((root / "reports" / "metrics.json").read_text())
    _style()
    chart_dataset(tables, out, metrics)
    chart_confusion(out, metrics)
    chart_model_comparison(out, metrics)
    chart_per_scenario(out, metrics)
    chart_false_alerts(tables, out)
    chart_slices(tables, out)
    chart_economics(tables, out, metrics)
    chart_prevalence(tables, out, metrics)
    chart_calibration(tables, out, metrics)
    if "whole_clip_only_model" in metrics.get("streaming", {}):
        chart_streaming(out, metrics)
    for name, fn in (("sensor_ablation.csv", chart_ablation), ("feature_importance.csv", chart_importance), ("detection_latency.csv", chart_latency)):
        if (tables / name).exists():
            fn(tables, out)
    return sorted(out.glob("*.png"))
