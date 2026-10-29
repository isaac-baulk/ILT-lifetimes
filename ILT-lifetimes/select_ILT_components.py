"""Run either Alex ILT implementation and summarize the first fit with MAWR < 0.9.

Place this file alongside:
    L1_ILT_Alex.py
    tikhonov_ILT_Alex.py

Run from the terminal:
    python select_ILT_components.py

No changes are required in the two existing ILT scripts.
"""

from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np


# =========================== EDIT HERE ================================
METHOD = "lasso"         # "lasso" or "tikhonov"
INPUT_FILE = Path(__file__).resolve().parent / "data/81801-bwd-counts.dat"
MAX_COMPONENTS = 8           # Maximum number to try
MAWR_THRESHOLD = 0.9         # Select the FIRST fit with MAWR < 0.9

# Use the same rate grid when comparing the two methods.
# Set the limits to suit your experiment; rates are in inverse time units.
LAMBDAS = np.linspace(0.0, 1.7, 80)

# All integrated contributions must have a common interval.
# By default integrate from 0 to the largest measured time, including lambda=0.
INTEGRAL_START = 0.0
INTEGRAL_END = None         # None: largest measured time; np.inf: infinity

# Plot all iterations and the final normalized contributions.
PLOT_ITERATIONS = True      # For Tikhonov; L1 always plots in current file
PLOT_COMPONENTS = True

# The third column of the Tikhonov data file can optionally contain errors.
# Use "poisson" for the same weighting as L1.
TIKHONOV_WEIGHT_MODE = "poisson"  # "poisson", "file", or "uniform"

# t in microseconds -> tau in microseconds; change for other time units.
TIME_UNIT = "µs"
OUTPUT_CSV = Path(__file__).resolve().parent / "selected_ILT_components.csv"
# ======================================================================


def fit_until_threshold(method, input_file, lambdas, max_components, threshold):
    """Use the existing iterative fits and their native target-MAWR stopping."""
    # The original functions stop when MAWR <= target.  Using nextafter
    # implements the requested strict criterion MAWR < threshold.
    strict_target = np.nextafter(float(threshold), -np.inf)

    if method.lower() in ("lasso", "l1"):
        from L1_ILT_Alex import do_iterative_lasso_ILT

        parameters, history = do_iterative_lasso_ILT(
            str(input_file),
            num_elements=max_components,
            lambdas=lambdas,
            target_mawr=strict_target,
        )
    elif method.lower() in ("tikhonov", "l2"):
        from tikhonov_ILT_Alex import do_iterative_ILT

        parameters, history = do_iterative_ILT(
            str(input_file),
            num_elements=max_components,
            lambdas=lambdas,
            target_mawr=strict_target,
            weight_mode=TIKHONOV_WEIGHT_MODE,
            plot=PLOT_ITERATIONS,
        )
    else:
        raise ValueError("METHOD must be 'lasso' or 'tikhonov'.")

    parameters = np.asarray(parameters, dtype=float)
    history = np.asarray(history, dtype=float)
    if history.ndim != 1 or len(history) == 0 or not np.all(np.isfinite(history)):
        raise ValueError("Invalid MAWR history returned by the ILT module.")

    matches = np.flatnonzero(history < threshold)
    if len(matches) == 0:
        return None, parameters, history

    first_n = int(matches[0])
    if first_n != len(parameters) // 2:
        # This would happen if the called ILT function were modified to
        # continue fitting beyond the FIRST threshold crossing.  In that
        # case its final parameters would not represent the selected n.
        raise RuntimeError(
            "The returned parameters are not from the first MAWR crossing. "
            "Use target_mawr=threshold in the called fit function."
        )
    return first_n, parameters, history


def integral_per_exponential(amplitudes, rates, start_time, end_time):
    """Compute each A*exp(-lambda*t) integral on the SAME time interval.

    Finite interval [a,b]:
       A * exp(-lambda*a) * (1 - exp(-lambda*(b-a))) / lambda
       A * (b-a)                                 if lambda == 0

    Infinite interval [a,+inf):
       (A/lambda)*exp(-lambda*a), lambda > 0 only.

    -expm1(-x) avoids losing precision near lambda=0.
    """
    amplitudes = np.asarray(amplitudes, dtype=float)
    rates = np.asarray(rates, dtype=float)
    if np.any(amplitudes < 0) or np.any(rates < 0):
        raise ValueError("This summary assumes nonnegative fitted A and lambda.")
    if start_time < 0:
        raise ValueError("INTEGRAL_START must be nonnegative.")
    if not end_time > start_time:
        raise ValueError("INTEGRAL_END must exceed INTEGRAL_START.")

    if np.isinf(end_time):
        if np.any(rates == 0):
            raise ValueError(
                "A lambda=0 component has infinite area over [start,+inf). "
                "Set INTEGRAL_END=None to integrate over the measured window."
            )
        return (amplitudes / rates) * np.exp(-rates * start_time)

    duration = end_time - start_time
    # For lambda -> 0, (1-exp(-lambda*duration))/lambda -> duration.
    decay_integral = np.full_like(rates, duration, dtype=float)
    np.divide(
        -np.expm1(-rates * duration),
        rates,
        out=decay_integral,
        where=rates > 0,
    )
    return amplitudes * np.exp(-rates * start_time) * decay_integral


def summarize_components(parameters, start_time, end_time):
    """Return one row per fitted exponential, preserving its fit index."""
    amplitudes = parameters[0::2]
    rates = parameters[1::2]
    areas = integral_per_exponential(amplitudes, rates, start_time, end_time)
    total_area = float(np.sum(areas))
    if not np.isfinite(total_area) or total_area <= 0:
        raise ValueError("Cannot normalize: total integrated contribution is zero or nonfinite.")

    normalized_areas = areas / total_area
    lifetimes = np.divide(
        1.0,
        rates,
        out=np.full_like(rates, np.inf, dtype=float),
        where=rates > 0,
    )
    return [
        {
            "component": i + 1,
            "amplitude": float(amplitudes[i]),
            "rate": float(rates[i]),
            "lifetime": float(lifetimes[i]),
            "integral": float(areas[i]),
            "fraction": float(normalized_areas[i]),
        }
        for i in range(len(rates))
    ]


def plot_integral_fractions(rows, n_selected, mawr):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    xs = np.arange(1, len(rows) + 1)
    fractions = [row["fraction"] for row in rows]
    bars = ax.bar(xs, fractions)
    labels = [
        f"{row['component']}\nτ={row['lifetime']:.4g} {TIME_UNIT}"
        if np.isfinite(row["lifetime"])
        else f"{row['component']}\nτ=∞"
        for row in rows
    ]
    ax.set_xticks(xs, labels)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Normalized exponential integral")
    ax.set_xlabel("Component and lifetime")
    ax.set_title(f"{METHOD.upper()} — {n_selected} components, MAWR={mawr:.4f}")
    for bar, fraction in zip(bars, fractions):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            fraction + 0.015,
            f"{fraction:.1%}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    plt.show()


def main():
    if not INPUT_FILE.is_file():
        raise FileNotFoundError(f"Data file not found: {INPUT_FILE}")
    if not np.isfinite(MAWR_THRESHOLD) or MAWR_THRESHOLD <= 0:
        raise ValueError("MAWR_THRESHOLD must be positive and finite.")

    # Read time only: the fitting modules both use t > 0.5.
    times = np.loadtxt(INPUT_FILE, comments="#", usecols=0, ndmin=1)
    times = times[np.isfinite(times) & (times > 0.5)]
    if len(times) == 0:
        raise ValueError("No measurements with t > 0.5 found.")
    end_time = float(times.max()) if INTEGRAL_END is None else float(INTEGRAL_END)

    n_selected, parameters, metrics = fit_until_threshold(
        METHOD, INPUT_FILE, LAMBDAS, MAX_COMPONENTS, MAWR_THRESHOLD
    )

    print("\nNumber of exponentials vs MAWR:")
    for n, metric in enumerate(metrics):
        label = "  <-- first below threshold" if n_selected == n else ""
        print(f"  {n:2d} : {metric:.6f}{label}")

    if n_selected is None:
        print(f"\nNo fit reached MAWR < {MAWR_THRESHOLD:.3f} "
              f"within {MAX_COMPONENTS} components (or fitting stopped earlier).")
        print("No model has been selected; no normalized integral is reported.")
        return None

    print(f"\nSelected FIRST model: n = {n_selected}, MAWR = {metrics[n_selected]:.6f}")
    if n_selected == 0:
        print("Zero exponentials: there are no component integrals or lifetimes.")
        return []

    rows = summarize_components(parameters, INTEGRAL_START, end_time)

    right_edge = "+∞" if np.isinf(end_time) else f"{end_time:g}"
    print(f"\nIntegrals over t = [{INTEGRAL_START:g}, {right_edge}] {TIME_UNIT}")
    print(f"{'Component':>9} {'Amplitude':>12} {'Rate (1/'+TIME_UNIT+')':>14} "
          f"{'Lifetime ('+TIME_UNIT+')':>17} {'Normalized area':>17}")
    for row in rows:
        tau = f"{row['lifetime']:.6g}" if np.isfinite(row["lifetime"]) else "infinity"
        print(f"{row['component']:9d} {row['amplitude']:12.6g} "
              f"{row['rate']:14.6g} {tau:>17} {row['fraction']:17.6f}")
    print(f"{'TOTAL':>9} {'':>12} {'':>14} {'':>17} "
          f"{sum(row['fraction'] for row in rows):17.6f}")

    # Easy to import into Excel or further scripts.
    with OUTPUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved component table to: {OUTPUT_CSV}")

    if PLOT_COMPONENTS:
        plot_integral_fractions(rows, n_selected, float(metrics[n_selected]))
    return rows


if __name__ == "__main__":
    main()
