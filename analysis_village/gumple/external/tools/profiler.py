#!/usr/bin/env python3
"""Generic plotting script for PROfit profile ROOT outputs (*_PROfile.root).

Usage:
    python profiler.py -f /path/to/files/*.root
    python profiler.py -f /path/to/dir1 /path/to/dir2 -o ./plots -p my_run_01
"""

import argparse
import math
from pathlib import Path
import sys
import warnings

try:
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import uproot
except ModuleNotFoundError as error:
    raise SystemExit(
        f"Missing Python package {error.name!r}. Install required packages via:\n"
        "python -m pip install numpy pandas matplotlib uproot"
    ) from error

NONZERO_PULL_TOL = 1.0e-12
TOP_N = 10
PARAMS_PER_ROW = 5
OSCILLATION_PARAMETERS = {"dmsq", "sinsq2thmm"}


def read_pull_file(profile_root: Path) -> pd.DataFrame:
    """Extract nuisance pulls and post-fit 1-sigma half-widths from a ROOT file."""
    with uproot.open(profile_root) as root_file:
        if "global_fit_result" not in root_file or "one_sigma_errs" not in root_file:
            raise KeyError(
                f"Missing required trees/graphs in {profile_root}. "
                "Expected 'global_fit_result' and 'one_sigma_errs'."
            )

        global_fit = root_file["global_fit_result"]
        one_sigma = root_file["one_sigma_errs"]

        labels = np.asarray(global_fit.axes[0].labels(flow=False), dtype=str)
        fit_values = np.asarray(global_fit.values(flow=False), dtype=float)

        _, pull_values = one_sigma.values()
        pull_values = np.asarray(pull_values, dtype=float)

        postfit_sigma_low = np.abs(
            np.asarray(one_sigma.errors("low")[1], dtype=float)
        )
        postfit_sigma_high = np.abs(
            np.asarray(one_sigma.errors("high")[1], dtype=float)
        )

    # Trim leading metadata/oscillation bins if one_sigma has fewer points than global_fit_result labels
    offset = len(labels) - len(pull_values)
    if offset > 0:
        labels = labels[offset:]
        fit_values = fit_values[offset:]

    lengths = {
        len(labels),
        len(fit_values),
        len(pull_values),
        len(postfit_sigma_low),
        len(postfit_sigma_high),
    }
    if len(lengths) != 1:
        raise ValueError(
            f"Cannot align pull arrays in {profile_root}: "
            f"labels={len(labels)}, fit={len(fit_values)}, "
            f"pulls={len(pull_values)}, low={len(postfit_sigma_low)}, "
            f"high={len(postfit_sigma_high)}"
        )

    result = pd.DataFrame(
        {
            "parameter": labels,
            "fit_value": fit_values,
            "postfit_pull": pull_values,
            "postfit_sigma_low": postfit_sigma_low,
            "postfit_sigma_high": postfit_sigma_high,
        }
    )

    # Filter out oscillation parameters or non-systematic metadata labels
    result = result[~result["parameter"].isin(OSCILLATION_PARAMETERS)].copy()
    
    # Drop entries matching file paths or run tags stored in global_fit_result
    result = result[
        ~result["parameter"].str.contains(r"\.root$|^run_|^tag_", case=False, regex=True)
    ].copy()

    finite_columns = ["postfit_pull", "postfit_sigma_low", "postfit_sigma_high"]
    result = result[np.isfinite(result[finite_columns]).all(axis=1)].copy()

    result["prefit_pull"] = 0.0
    result["prefit_sigma_low"] = 1.0
    result["prefit_sigma_high"] = 1.0
    result["postfit_sigma_mean"] = (
        result["postfit_sigma_low"] + result["postfit_sigma_high"]
    ) / 2.0
    result["postfit_to_prefit_sigma_ratio"] = result["postfit_sigma_mean"]
    result["max_side_abs_change_from_prefit"] = np.maximum(
        np.abs(result["postfit_sigma_low"] - 1.0),
        np.abs(result["postfit_sigma_high"] - 1.0),
    )

    return result.reset_index(drop=True)


def plot_all_pulls_grid(
    pulls: pd.DataFrame,
    run_labels: list,
    parameters: list,
    *,
    title: str,
    output_path: Path,
):
    """Draw all systematic pulls stacked vertically in rows of 5 parameters each."""
    parameters = list(parameters)
    if not parameters:
        return

    num_rows = math.ceil(len(parameters) / PARAMS_PER_ROW)
    fig, axes = plt.subplots(
        num_rows, 1, figsize=(12, 3.2 * num_rows), sharey=True, squeeze=False
    )
    axes = axes.flatten()

    offsets = np.linspace(-0.35, 0.35, len(run_labels))

    # Calculate global y-limits across all parameters to ensure consistent vertical scaling
    plotted_extents = [-1.0, 1.0]
    for label in run_labels:
        run = pulls[pulls["run_label"] == label].set_index("parameter").reindex(parameters)
        centers = run["postfit_pull"].to_numpy(dtype=float)
        low = run["postfit_sigma_low"].to_numpy(dtype=float)
        high = run["postfit_sigma_high"].to_numpy(dtype=float)
        valid = np.isfinite(centers) & np.isfinite(low) & np.isfinite(high)
        plotted_extents.extend(centers[valid] - low[valid])
        plotted_extents.extend(centers[valid] + high[valid])

    finite_extents = [abs(float(val)) for val in plotted_extents if np.isfinite(val)]
    y_limit = max(1.5, np.ceil((max(finite_extents, default=1.0) + 0.1) * 2.0) / 2.0)

    for row_idx in range(num_rows):
        ax = axes[row_idx]
        row_params = parameters[
            row_idx * PARAMS_PER_ROW : (row_idx + 1) * PARAMS_PER_ROW
        ]
        x = np.arange(len(row_params), dtype=float)

        # Background shaded band representing Pre-fit [-1.0, 1.0] region
        ax.axhspan(-1.0, 1.0, color="0.90", alpha=0.6, zorder=0)
        ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.0, zorder=1)

        for offset, label in zip(offsets, run_labels):
            selected = pulls[pulls["run_label"] == label].copy()
            run = (
                selected.drop_duplicates("parameter")
                .set_index("parameter")
                .reindex(row_params)
            )

            centers = run["postfit_pull"].to_numpy(dtype=float)
            low = run["postfit_sigma_low"].to_numpy(dtype=float)
            high = run["postfit_sigma_high"].to_numpy(dtype=float)
            valid = np.isfinite(centers) & np.isfinite(low) & np.isfinite(high)

            if not valid.any():
                continue

            ax.errorbar(
                x[valid] + offset,
                centers[valid],
                yerr=np.vstack([low[valid], high[valid]]),
                fmt="o",
                markersize=5,
                capsize=4.0,
                capthick=2.0,
                linewidth=2.0,
                label=label if len(run_labels) > 1 else "Post-fit pull",
                zorder=2,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(row_params, rotation=25, ha="right", fontsize=9)
        ax.set_xlim(-0.6, PARAMS_PER_ROW - 0.4)
        ax.set_ylim(-y_limit, y_limit)
        ax.set_ylabel("Nuisance pull [\u03c3]", fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(title, fontsize=13, y=1.01)
    if len(run_labels) > 1:
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.98, 1.02), ncol=min(4, len(run_labels)))

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_top_pulls(
    pulls: pd.DataFrame,
    run_labels: list,
    parameters: list,
    *,
    title: str,
    output_path: Path,
):
    """Draw single row pull plot for the top 10 pulls."""
    parameters = list(parameters)
    if not parameters:
        return

    x = np.arange(len(parameters), dtype=float)
    offsets = np.linspace(-0.35, 0.35, len(run_labels))

    fig, ax = plt.subplots(figsize=(11, 5.5))

    ax.axhspan(-1.0, 1.0, color="0.90", alpha=0.6, zorder=0, label="Pre-fit 1-sigma region")
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.0, zorder=1)

    plotted_extents = [-1.0, 1.0]

    for offset, label in zip(offsets, run_labels):
        selected = pulls[pulls["run_label"] == label].copy()
        run = (
            selected.drop_duplicates("parameter")
            .set_index("parameter")
            .reindex(parameters)
        )

        centers = run["postfit_pull"].to_numpy(dtype=float)
        low = run["postfit_sigma_low"].to_numpy(dtype=float)
        high = run["postfit_sigma_high"].to_numpy(dtype=float)
        valid = np.isfinite(centers) & np.isfinite(low) & np.isfinite(high)

        if not valid.any():
            continue

        ax.errorbar(
            x[valid] + offset,
            centers[valid],
            yerr=np.vstack([low[valid], high[valid]]),
            fmt="o",
            markersize=5,
            capsize=4.0,
            capthick=2.0,
            linewidth=2.0,
            label=label if len(run_labels) > 1 else "Post-fit pull",
            zorder=2,
        )

        plotted_extents.extend(centers[valid] - low[valid])
        plotted_extents.extend(centers[valid] + high[valid])

    ax.set_xticks(x)
    ax.set_xticklabels(parameters, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Nuisance pull [pre-fit sigma units]")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)

    finite_extents = [abs(float(val)) for val in plotted_extents if np.isfinite(val)]
    y_limit = max(1.5, np.ceil((max(finite_extents, default=1.0) + 0.1) * 2.0) / 2.0)
    ax.set_ylim(-y_limit, y_limit)

    if len(run_labels) > 1:
        ax.legend(ncol=min(4, len(run_labels) + 1), fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="profiler.py",
        description="Analyze and plot PROfit profile nuisance parameters from ROOT files.",
    )
    parser.add_argument(
        "-f",
        "--files",
        nargs="+",
        type=Path,
        required=True,
        help="Input ROOT files (*_PROfile.root) or directories containing them.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("./pull_plots_output"),
        help="Directory to save output CSVs and plot figures (default: ./pull_plots_output).",
    )
    parser.add_argument(
        "-p",
        "--output-prefix",
        type=str,
        default="",
        help="Custom prefix/name appended to output files (e.g., -p my_analysis).",
    )
    return parser.parse_args()


def resolve_root_files(inputs: list) -> dict:
    """Map sample labels to resolved ROOT file paths."""
    resolved_files = {}

    for item in inputs:
        if item.is_file() and item.suffix == ".root":
            label = item.stem.replace("_PROfile", "")
            resolved_files[label] = item
        elif item.is_dir():
            matches = sorted(item.glob("*_PROfile.root"))
            if not matches:
                matches = sorted(item.glob("*.root"))

            for match in matches:
                label = f"{item.name}/{match.stem.replace('_PROfile', '')}"
                resolved_files[label] = match

    return resolved_files


def main():
    args = parse_args()
    output_dir = args.output
    prefix = f"{args.output_prefix}_" if args.output_prefix else ""
    output_dir.mkdir(parents=True, exist_ok=True)

    input_map = resolve_root_files(args.files)
    if not input_map:
        sys.exit("Error: No valid ROOT files found from input arguments.")

    print(f"Found {len(input_map)} ROOT file(s) to analyze.")

    frames = []
    run_labels = []

    for label, path in input_map.items():
        try:
            df = read_pull_file(path)
            df.insert(0, "run_label", label)
            frames.append(df)
            run_labels.append(label)
            print(f"Loaded: {label} ({path.name})")
        except Exception as err:
            warnings.warn(f"Failed to read file {path}: {err}")

    if not frames:
        sys.exit("Error: No valid data could be loaded from the ROOT files.")

    pulls = pd.concat(frames, ignore_index=True)
    pulls["abs_postfit_pull"] = pulls["postfit_pull"].abs()
    pulls.to_csv(output_dir / f"{prefix}all_nuisance_pulls.csv", index=False)

    ordered_parameters = []
    for label in run_labels:
        run_params = pulls.loc[
            (pulls["run_label"] == label) & (pulls["abs_postfit_pull"] > NONZERO_PULL_TOL),
            "parameter",
        ]
        for name in run_params:
            if name not in ordered_parameters:
                ordered_parameters.append(name)

    # Output 1: Single master grid figure with all pulls (5 parameters per row)
    if ordered_parameters:
        plot_all_pulls_grid(
            pulls,
            run_labels,
            ordered_parameters,
            title="All Nuisance Parameter Pulls",
            output_path=output_dir / f"{prefix}all_systematic_pulls.png",
        )

    # Output 2: Top 10 largest pulls summary plot
    top_pull_parameters = (
        pulls[pulls["abs_postfit_pull"] > NONZERO_PULL_TOL]
        .groupby("parameter", as_index=False)
        .agg(max_abs_postfit_pull=("abs_postfit_pull", "max"))
        .nlargest(TOP_N, "max_abs_postfit_pull")["parameter"]
        .tolist()
    )

    if top_pull_parameters:
        plot_top_pulls(
            pulls,
            run_labels,
            top_pull_parameters,
            title=f"Top {len(top_pull_parameters)} Largest Post-fit Nuisance Pulls",
            output_path=output_dir / f"{prefix}top_10_largest_pulls.png",
        )

    print(f"\nProcessing complete! Outputs saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
