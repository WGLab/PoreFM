"""Plot completed runs from the PoreFM HeLa RNA walkthrough.

Run from the analysis directory containing outputs/supervised,
outputs/supervised-pseu, outputs/prototype, outputs/zero-shot, and/or
outputs/pairwise. The supervised plot requires both supervised runs. Select one
method with --method, or use all after completing all methods. Writes summary
Parquets and PNGs under
summaries/ and plots/; rerunning replaces those derived files. No GPU is needed.
Coordinates in stored summaries are zero-based; figures display one-based sites.
"""

import argparse
from porefm.outputs import open_output

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

START, END = 134843446, 134843646
region = [("chrom", "==", "chr9"), ("position", ">=", START),
          ("position", "<", END), ("strand", "==", 0)]
truth = [(134843484, "pseU", "#009988"),
         (134843495, "m6A", "#CC3311"),
         (134843606, "m6A", "#CC3311")]

def finish_plot(fig, axes, title, filename):
    """Save a genomic track, converting zero-based sites to one-based display.

    ``axes`` may be one Matplotlib axis or an array. ``filename`` is a stem
    under ``plots/``; that directory must exist. The figure is also displayed.
    """
    axes = np.atleast_1d(axes)
    for ax in axes:
        for position, label, color in truth:
            ax.axvline(position + 1, color=color, linestyle=":", linewidth=1)
        ax.set_xlim(START + 1, END)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.2)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
    for i, (position, label, color) in enumerate(truth):
        axes[0].text(position + 1, 1.04 + 0.10 * (i == 1), label,
                     transform=axes[0].get_xaxis_transform(),
                     ha="center", color=color, fontsize=10)
    axes[-1].set_xlabel("GRCh38 chr9 coordinate (1-based)")
    axes[0].set_title(title, loc="left", pad=48, fontsize=13)
    fig.tight_layout()
    fig.savefig(f"plots/{filename}.png", dpi=180, bbox_inches="tight")
    plt.show()

def plot_supervised():
    """Plot both RNA classifiers, retaining each model's site denominator.

    Read ``outputs/supervised`` (A/m6A/inosine) and
    ``outputs/supervised-pseu`` (U/pseU). Write a summary for each model and
    one combined figure, showing sites with at least 20 calls per model.
    """
    fig, ax = plt.subplots(figsize=(11, 3.8))
    runs = [
        ("supervised", [("m6A", "m6A", "#CC3311"),
                        ("Ino", "Inosine", "#4477AA")]),
        ("supervised-pseu", [("pseU", "pseU", "#009988")]),
    ]
    for directory, classes in runs:
        calls = open_output(f"outputs/{directory}")
        sites = calls.summarize_loci(
            filters=region, include_class_counts=True,
        ).sort_values("position")
        sites.to_parquet(f"summaries/{directory}.parquet", index=False)
        plot_sites = sites[sites.called_count >= 20]
        columns = ["position", "called_count"] + [f"freq_{label}" for label, _, _ in classes]
        print(directory, plot_sites[columns].head())
        for label, name, color in classes:
            ax.scatter(plot_sites.position + 1, 100 * plot_sites[f"freq_{label}"],
                       s=18, label=name, color=color)
    ax.set(ylabel="Modification calls (%)", ylim=(-3, 103))
    ax.legend(loc="upper right", frameon=False)
    finish_plot(fig, ax, "Supervised calls · HeLa WT / COL5A1", "supervised")

def plot_prototype():
    """Write the prototype site summary and example figure."""
    prototype = open_output("outputs/prototype")
    print(prototype.class_labels)
    prototype_sites = prototype.summarize_loci(
        filters=region, include_class_counts=True,
    ).sort_values("position")
    prototype_sites.to_parquet("summaries/prototype.parquet", index=False)
    modifications = [label for label in prototype.class_labels
                     if label not in {"A", "C", "G", "T", "U"}]
    colors = {"m6A": "#CC3311", "Inosine": "#4477AA", "2OmeA": "#EEAA33",
              "m5C": "#228833", "2OmeC": "#AA3377", "2OmeG": "#66CCEE",
              "pseU": "#009988", "2OmeU": "#8844AA"}

    fig, ax = plt.subplots(figsize=(11, 4.3))
    for label in modifications:
        ax.scatter(prototype_sites.position + 1, 100 * prototype_sites[f"freq_{label}"],
                   s=19, alpha=0.8, color=colors[label], label=label)
    ax.set(ylabel="Modification calls (%)", ylim=(-3, 103))
    ax.legend(loc="upper right", ncol=4, frameon=False, fontsize=8)
    finish_plot(fig, ax, "Prototype calls · HeLa WT / COL5A1", "prototype")

def plot_zero_shot():
    """Write the zero-shot site summary and example figure."""
    denovo = open_output("outputs/zero-shot")
    denovo_sites = denovo.summarize_loci(
        filters=region, modified_pvalue_cutoff=0.01, combine_pvalues=False,
    ).sort_values("position")
    denovo_sites.to_parquet("summaries/zero-shot.parquet", index=False)
    read_calls = pd.concat([
        table.to_pandas() for table in denovo.iter_tables(
            columns=["readname", "position", "pvalue"], filters=region,
        )
    ], ignore_index=True)
    coverage = read_calls.groupby("readname").position.nunique().rename("positions").reset_index()
    readname = coverage.sort_values(["positions", "readname"], ascending=[False, True]).iloc[0].readname
    one_read = read_calls[read_calls.readname == readname].sort_values("position")
    print("Example read:", readname, "scored positions:", len(one_read))
    one_read.to_parquet("summaries/zero-shot-read.parquet", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    axes[0].plot(denovo_sites.position + 1, 100 * denovo_sites.fraction_modified, color="#284B63")
    axes[0].set(ylabel="Anomaly calls (%)", ylim=(-3, 103))
    axes[1].plot(one_read.position + 1, one_read.pvalue.clip(lower=1e-8),
                 color="#4477AA", marker=".", markersize=3)
    axes[1].axhline(0.01, color="#777777", linestyle="--", label="p = 0.01")
    lower = min(0.001, max(1e-8, one_read.pvalue.min()))
    axes[1].set(yscale="log", ylim=(1, lower), ylabel="One-read anomaly p-value")
    axes[1].legend(frameon=False)
    finish_plot(fig, axes, "De novo detection · HeLa WT / COL5A1", "zero-shot")

def plot_pairwise():
    """Write the pairwise site summary and example figure."""
    pairwise = open_output("outputs/pairwise")
    pairwise_sites = pairwise.summarize_loci(
        filters=region, modified_pvalue_cutoff=0.01,
    ).sort_values("position")
    shifts = pd.concat([
        table.to_pandas() for table in pairwise.iter_site_tables(
            kind="locus_metrics", filters=region,
        )
    ], ignore_index=True).sort_values("position")
    pairwise_sites.to_parquet("summaries/pairwise.parquet", index=False)
    shifts.to_parquet("summaries/embedding-shifts.parquet", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    axes[0].plot(shifts.position + 1, shifts.cosine_distance, color="#284B63")
    axes[0].set(ylabel="Mean-embedding\ncosine distance", ylim=(0, None))
    axes[1].plot(pairwise_sites.position + 1, 100 * pairwise_sites.fraction_modified, color="#284B63")
    axes[1].set(ylabel="WT anomaly calls (%)", ylim=(-3, 103))
    finish_plot(fig, axes, "Pairwise WT vs IVT · COL5A1", "pairwise")

def main():
    """Select completed inference outputs and create derived plots/summaries."""
    methods = {
        "supervised": plot_supervised,
        "prototype": plot_prototype,
        "zero-shot": plot_zero_shot,
        "pairwise": plot_pairwise,
    }
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["all", *methods], default="all")
    args = parser.parse_args()
    selected = list(methods) if args.method == "all" else [args.method]
    required = [directory for name in selected
                for directory in (["supervised", "supervised-pseu"]
                                  if name == "supervised" else [name])]
    missing = [name for name in required if not Path("outputs", name).is_dir()]
    if missing:
        parser.error("Complete the walkthrough inference first; missing output directories: "
                     + ", ".join(missing))
    Path("plots").mkdir(exist_ok=True)
    Path("summaries").mkdir(exist_ok=True)
    for name in selected:
        methods[name]()


if __name__ == "__main__":
    main()
