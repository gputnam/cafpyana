#!/usr/bin/python3

import argparse
import matplotlib.pyplot as plt
import numpy as np
import uproot


def plot_and_calculate_p_value(
    root_file_path, obs_chi2_null, obs_chi2_osc, output_plot="pvalue_plot.png"
):
    """Calculates p-value and generates a distribution plot for Delta Chi2."""
    obs_delta_chi2 = abs(obs_chi2_null - obs_chi2_osc)

    # 1. Load data from ROOT file
    with uproot.open(root_file_path) as file:
        tree = file["tree"]
        chi2_null_arr = tree["chi2_syst"].array(library="np")
        chi2_osc_arr = tree["chi2_osc"].array(library="np")

    delta_chi2_dist = chi2_null_arr - chi2_osc_arr

    # 2. Calculate empirical p-value
    n_total = len(delta_chi2_dist)
    n_exceed = np.sum(delta_chi2_dist >= obs_delta_chi2)
    p_value = n_exceed / n_total

    # 3. Create the plot
    plt.figure(figsize=(8, 5))

    # Plot pseudo-experiment distribution
    counts, bins, _ = plt.hist(
        delta_chi2_dist,
        bins=50,
        density=True,
        histtype="stepfilled",
        alpha=0.4,
        color="steelblue",
        edgecolor="navy",
        label=r"Null Distribution ($\Delta\chi^2_{MC}$)",
    )

    # Highlight the p-value tail area (>= observed value)
    tail_bins = bins[bins >= obs_delta_chi2]
    if len(tail_bins) > 1:
        # Interpolate counts for tail shading
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        mask = bin_centers >= obs_delta_chi2
        plt.fill_between(
            bin_centers[mask],
            0,
            counts[mask],
            color="crimson",
            alpha=0.5,
            step="mid",
            label=f"Tail Area (p = {p_value:.4f})",
        )

    # Vertical line for observed value
    plt.axvline(
        obs_delta_chi2,
        color="crimson",
        linestyle="--",
        linewidth=2,
        label=rf"Observed $\Delta\chi^2$ = {obs_delta_chi2:.2f}",
    )

    # Formatting
    plt.xlabel(r"$\Delta\chi^2 = \chi^2_{\mathrm{null}} - \chi^2_{\mathrm{osc}}$", fontsize=12)
    plt.ylabel("Probability Density", fontsize=12)
    plt.title(
        rf"Feldman-Cousins $\Delta\chi^2$ Distribution (p-value = {p_value:.4f})",
        fontsize=13,
    )
    plt.legend(frameon=True, loc="upper right")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    # Save and display
    plt.savefig(output_plot, dpi=300)
    print(f"Plot saved as: {output_plot}")
    print(f"Observed Delta Chi2: {obs_delta_chi2:.4f}")
    print(f"Calculated p-value: {p_value:.5f}")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot Delta Chi2 distribution and observed data value."
    )

    parser.add_argument(
        "--file",
        "-f",
        type=str,
        default="Gump_FCDist_FC.root",
        help="Path to ROOT file",
    )
    parser.add_argument(
        "--null",
        "-n",
        type=float,
        required=True,
        help="Observed chi2 value under null hypothesis",
    )
    parser.add_argument(
        "--osc",
        "-o",
        type=float,
        required=True,
        help="Observed chi2 value under oscillation hypothesis",
    )
    parser.add_argument(
        "--output",
        "-out",
        type=str,
        default="pvalue_plot.png",
        help="Filename for saved plot",
    )

    args = parser.parse_args()

    plot_and_calculate_p_value(args.file, args.null, args.osc, args.output)
