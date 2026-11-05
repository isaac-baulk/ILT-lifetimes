"""Event-count sweep for L1, Tikhonov, or 'dumb' multiexponential fitting.

Place beside generate_synthetic_musr.py, L1_ILT_Alex.py,
 tikhonov_ILT_Alex.py, and dumb_ILT_Alex.py.

For each event count and random trial:
  - create a synthetic 100 ns--32 us, 3-column decay histogram;
  - run the requested iterative global fits with NO MAWR early stopping;
  - plot all MAWR-versus-component curves on one plot per method;
  - check whether the KNOWN lifetimes and integrated fractions are recovered.

Outputs: mawr_curves_<method>.png, recovery_probability.png,
         metric_by_trial.csv, recovery_by_trial.csv, fitted_components.csv.
"""

from contextlib import nullcontext, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import csv
import io

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

from generate_synthetic_musr import generate_synthetic_data
from select_ILT_components import integral_per_exponential


# ========================== EDIT HERE ===============================

# One or several: 'lasso', 'tikhonov', 'dumb'
METHODS = ("dumb",)  # e.g. ("lasso", "tikhonov", "dumb")

# Integrated signal fractions (NOT starting amplitudes) and lifetimes (us).
STOICHIOMETRY = [0.7, 0.2, 0.1]
LIFETIMES_US = [1.0, 1.2, 1.5]

EVENT_COUNTS = [int(c) for c in np.logspace(5, 15, 21)] # 100_000, 300_000, 1_000_000, 3_000_000, 10_000_000, 30_000_000, 100_000_000, 300_000_000, 1_000_000_000, 3_000_000_000, 10_000_000_000, 30_000_000_000, 100_000_000_000]
N_TRIALS = 1           # Use >= 10 for a meaningful detection probability.
RANDOM_SEED = 42

START_TIME_NS = 100.0
END_TIME_US = 15.0
BIN_WIDTH_NS = 16.0

# Flat background as a FRACTION of the total events, so it scales with
# measurement duration. A nonzero fraction counts as an additional
# constant (lambda=0) component when evaluating recoverability.
BACKGROUND_FRACTION = 0.0

# None => number of true components (fastest). Set e.g. 4 to examine
# overfitting; doing that adds an extra true-n refit for recovery checks.
MAX_COMPONENTS = 5
LAMBDAS = np.linspace(0.0, 1.7, 60)
MAWR_Y_SCALE = "log"  # "log" or "symlog"
MAWR_THRESHOLD = 0.9

# A fit has 'resolved' the components only if the FIRST model below the
# MAWR threshold contains exactly the true number of components, AND its
# normalized integral fractions and lifetimes match within these limits.
MAX_REL_LIFETIME_ERROR = 0.20    # 20% error in each nonzero lifetime
MAX_ABS_FRACTION_ERROR = 0.03    # 3 percentage points per component
MAX_ZERO_RATE = 0.01            # 1/us, for a constant-background component
REQUIRED_SUCCESS_RATE = 0.80    # at least 80% of trials (if repeated)

OUTPUT_DIR = Path(__file__).resolve().parent / "synthetic_ilt_sweep"
SAVE_SYNTHETIC_DATA = False
SHOW_PLOTS = True
VERBOSE_FITS = False

# ====================================================================


def save_csv(path, rows, columns):
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def run_fit(method, path, n_components, lambdas):
    """Call the UNMODIFIED Alex fitters; force the complete MAWR curve.

    L1 always creates figures: temporarily close/suppress them so an
    event-count sweep doesn't open hundreds of windows. Other fitters
    already accept plot=False.
    """
    method = method.lower()
    if method == "lasso":
        from L1_ILT_Alex import do_iterative_lasso_ILT as fit
        args = dict(input_file=str(path), num_elements=n_components,
                    lambdas=lambdas, target_mawr=None)
        # Suppress L1's plotting calls without editing L1_ILT_Alex.py.
        plot_context = patch.multiple(
            plt,
            show=lambda *a, **kw: plt.close("all"),
            pause=lambda *a, **kw: None,
        )
    elif method in ("tikhonov", "dumb"):
        if method == "tikhonov":
            from tikhonov_ILT_Alex import do_iterative_ILT as fit
        else:
            from dumb_ILT_Alex import do_iterative_ILT as fit
        args = dict(input_file=str(path), num_elements=n_components,
                    lambdas=lambdas, target_mawr=None,
                    weight_mode="poisson", plot=False)
        plot_context = nullcontext()
    else:
        raise ValueError("Method must be 'lasso', 'tikhonov', or 'dumb'.")

    output_context = nullcontext() if VERBOSE_FITS else redirect_stdout(io.StringIO())
    with plot_context, output_context:
        params, history = fit(**args)
    return np.asarray(params, float), np.asarray(history, float)


def check_recovery(params, true_rates, true_fractions, t_start, t_end):
    """Match fitted components to truth, independent of fit ordering."""
    n = len(true_rates)
    if len(params) != 2 * n:
        return False, [], np.inf, np.inf

    fitted_amplitudes = params[0::2]
    fitted_rates = params[1::2]
    integrals = integral_per_exponential(
        fitted_amplitudes, fitted_rates, t_start, t_end
    )
    if not np.isfinite(integrals).all() or integrals.sum() <= 0:
        return False, [], np.inf, np.inf
    fitted_fractions = integrals / integrals.sum()

    # Assignment cost combines differences in fractions and lifetimes.
    rate_errors = np.empty((n, n))
    fraction_errors = np.empty((n, n))
    for i, (r_true, f_true) in enumerate(zip(true_rates, true_fractions)):
        for j, (r_fit, f_fit) in enumerate(zip(fitted_rates, fitted_fractions)):
            if r_true == 0:
                rate_errors[i, j] = r_fit / MAX_ZERO_RATE
            elif r_fit <= 0:
                rate_errors[i, j] = np.inf
            else:
                rate_errors[i, j] = abs(r_true / r_fit - 1.0)
            fraction_errors[i, j] = abs(f_fit - f_true)

    cost = (np.minimum(rate_errors / MAX_REL_LIFETIME_ERROR, 1e6)
            + fraction_errors / MAX_ABS_FRACTION_ERROR)
    true_indices, fit_indices = linear_sum_assignment(cost)

    matched = []
    all_good = True
    max_rate_err, max_fraction_err = 0.0, 0.0
    for i, j in zip(true_indices, fit_indices):
        r_true, r_fit = true_rates[i], fitted_rates[j]
        tau_true = np.inf if r_true == 0 else 1 / r_true
        tau_fit = np.inf if r_fit == 0 else 1 / r_fit
        r_error = rate_errors[i, j]
        f_error = fraction_errors[i, j]
        rate_ok = (r_fit <= MAX_ZERO_RATE if r_true == 0
                   else r_error <= MAX_REL_LIFETIME_ERROR)
        good = bool(rate_ok and f_error <= MAX_ABS_FRACTION_ERROR)
        all_good &= good
        max_rate_err = max(max_rate_err, float(r_error))
        max_fraction_err = max(max_fraction_err, float(f_error))
        matched.append({
            "true_component": int(i + 1),
            "fitted_component": int(j + 1),
            "true_fraction": float(true_fractions[i]),
            "fitted_fraction": float(fitted_fractions[j]),
            "true_tau_us": float(tau_true),
            "fitted_tau_us": float(tau_fit),
            "fraction_abs_error": float(f_error),
            "lifetime_rel_error": float(r_error) if r_true != 0 else np.nan,
            "recovered": good,
        })
    return bool(all_good), matched, max_rate_err, max_fraction_err


def plot_mawr(method, results, event_counts, n_max, n_true):
    fig, ax = plt.subplots(figsize=(10, 6))
    for count in event_counts:
        histories = [r["history"] for r in results
                     if r["method"] == method and r["events"] == count]
        values = np.full((len(histories), n_max + 1), np.nan)
        for i, hist in enumerate(histories):
            values[i, :min(len(hist), n_max + 1)] = hist[:n_max + 1]
        median = np.full(n_max + 1, np.nan)
        low, high = median.copy(), median.copy()
        for k in range(n_max + 1):
            v = values[:, k]
            v = v[np.isfinite(v)]
            if len(v):
                median[k] = np.median(v)
                low[k], high[k] = np.percentile(v, [16, 84])
        xs = np.arange(n_max + 1)
        line, = ax.plot(xs, median, "o-", label=f"{count:,} events")
        if len(histories) > 1:
            ax.fill_between(xs, low, high, color=line.get_color(), alpha=0.15)

    ax.axhline(MAWR_THRESHOLD, color="black", linestyle="--",
               label=f"MAWR = {MAWR_THRESHOLD:g}")
    ax.axvline(n_true, color="gray", linestyle=":",
               label=f"True number of components = {n_true}")
    ax.set_xlabel("Number of fitted exponentials")
    ax.set_ylabel("Mean absolute weighted residual (MAWR)")
    ax.set_title(f"{method.upper()} — event-count dependence"
                 + (" (median and 16–84% range)" if N_TRIALS > 1 else ""))
    if MAWR_Y_SCALE == "symlog":
        ax.set_yscale("symlog", linthresh=0.01)
        ax.set_ylim(bottom=0)
    else:
        ax.set_yscale("log")
    ax.set_xticks(np.arange(n_max + 1))
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out = OUTPUT_DIR / f"mawr_curves_{method}.png"
    fig.savefig(out, dpi=180)
    print(f"Saved {out}")


def main():
    methods = tuple(m.lower() for m in METHODS)
    if not methods or not set(methods) <= {"lasso", "tikhonov", "dumb"}:
        raise ValueError("METHODS must contain lasso, tikhonov, and/or dumb.")
    weights = np.asarray(STOICHIOMETRY, dtype=float)
    taus = np.asarray(LIFETIMES_US, dtype=float)
    if weights.ndim != 1 or len(weights) == 0 or len(weights) != len(taus):
        raise ValueError("STOICHIOMETRY and LIFETIMES_US must be equal-length lists.")
    if (np.any(weights <= 0) or not np.isfinite(weights).all()
            or np.any(taus <= 0) or not np.isfinite(taus).all()):
        raise ValueError("Stoichiometries and lifetimes must be finite, positive.")
    if not 0 <= BACKGROUND_FRACTION < 1:
        raise ValueError("BACKGROUND_FRACTION must be between 0 and 1 (exclusive).")
    if N_TRIALS < 1 or not np.isfinite(MAWR_THRESHOLD) or MAWR_THRESHOLD <= 0:
        raise ValueError("N_TRIALS must be >= 1 and MAWR_THRESHOLD positive.")
    if not 0 < REQUIRED_SUCCESS_RATE <= 1:
        raise ValueError("REQUIRED_SUCCESS_RATE must be in (0, 1].")
    if any(int(n) != n or n <= 0 for n in EVENT_COUNTS):
        raise ValueError("EVENT_COUNTS must contain positive integers.")

    signal_fractions = weights / weights.sum()
    true_rates = 1.0 / taus
    true_fractions = (1 - BACKGROUND_FRACTION) * signal_fractions
    if BACKGROUND_FRACTION > 0:
        true_rates = np.r_[true_rates, 0.0]
        true_fractions = np.r_[true_fractions, BACKGROUND_FRACTION]
    n_true = len(true_rates)
    n_max = n_true if MAX_COMPONENTS is None else int(MAX_COMPONENTS)
    if n_max < 1:
        raise ValueError("MAX_COMPONENTS must be >= 1.")

    n_bins = max(2, int(round((END_TIME_US - START_TIME_NS / 1000.0)
                               / (BIN_WIDTH_NS / 1000.0))) + 1)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metric_rows, recovery_rows, fitted_rows, all_runs = [], [], [], []
    event_counts = sorted(set(map(int, EVENT_COUNTS)))

    print(f"True components: {n_true}; trials/count: {N_TRIALS}; max fit: {n_max}")
    print("True lifetimes (us):", ["infinity" if r == 0 else round(1 / r, 5)
                                  for r in true_rates])
    print("True integrated fractions:", np.round(true_fractions, 6))

    with TemporaryDirectory() as tempdir:
        for event_count in event_counts:
            background_per_bin = BACKGROUND_FRACTION * event_count / n_bins
            for trial in range(N_TRIALS):
                seed = RANDOM_SEED + 10000 * event_counts.index(event_count) + trial
                t, counts, sigma, _expected, _parts = generate_synthetic_data(
                    stoichiometry=STOICHIOMETRY,
                    lifetimes_us=LIFETIMES_US,
                    total_events=event_count,
                    start_time_ns=START_TIME_NS,
                    end_time_us=END_TIME_US,
                    bin_width_ns=BIN_WIDTH_NS,
                    background_counts_per_bin=background_per_bin,
                    seed=seed,
                )
                input_path = (OUTPUT_DIR if SAVE_SYNTHETIC_DATA
                              else Path(tempdir)) / f"synthetic_{event_count}_trial{trial+1}.dat"
                np.savetxt(input_path, np.column_stack((t, counts, sigma)),
                           fmt=["%.9f", "%d", "%.9f"],
                           header="time_us counts sigma_counts")

                for method in methods:
                    print(f"{method:9s} | N={event_count:>10,} | trial={trial+1}/{N_TRIALS}",
                          end="", flush=True)
                    try:
                        params, history = run_fit(method, input_path, n_max, LAMBDAS)
                        if not np.all(np.isfinite(history)):
                            raise ValueError("Fit returned nonfinite MAWR")
                    except Exception as exc:
                        print(f" | FAILED: {type(exc).__name__}: {exc}")
                        params = np.array([])
                        history = np.array([])

                    crossing = np.flatnonzero(history < MAWR_THRESHOLD)
                    selected_n = int(crossing[0]) if len(crossing) else None
                    all_runs.append({"method": method, "events": event_count,
                                     "trial": trial + 1, "history": history})
                    for k, mawr in enumerate(history):
                        metric_rows.append({"method": method, "events": event_count,
                                            "trial": trial + 1, "components": k,
                                            "mawr": float(mawr)})

                    # If n_max > n_true, the returned parameters describe the
                    # final fit. Repeat only up to n_true to evaluate recovery.
                    at_true_n = len(history) - 1 == n_true
                    p_true = params if at_true_n else None
                    if len(history) - 1 > n_true:
                        try:
                            p_true, _ = run_fit(method, input_path, n_true, LAMBDAS)
                        except Exception as exc:
                            print(f" | true-n refit failed: {exc}")

                    accurate = False
                    matched = []
                    if p_true is not None and len(p_true) == 2 * n_true:
                        accurate, matched, _, _ = check_recovery(
                            p_true, true_rates, true_fractions, t[0], t[-1]
                        )
                    selected_correct = bool(selected_n == n_true and accurate)
                    recovery_rows.append({
                        "method": method, "events": event_count, "trial": trial + 1,
                        "first_below_threshold": selected_n if selected_n is not None else "",
                        "mawr_at_true_n": (float(history[n_true])
                                            if len(history) > n_true else np.nan),
                        "parameters_match_truth": accurate,
                        "correct_selection_and_recovery": selected_correct,
                    })
                    for m in matched:
                        fitted_rows.append({"method": method, "events": event_count,
                                            "trial": trial + 1, **m})
                    print(f" | selected={selected_n} | truth_recovered={accurate}")

    save_csv(OUTPUT_DIR / "metric_by_trial.csv", metric_rows,
             ["method", "events", "trial", "components", "mawr"])
    save_csv(OUTPUT_DIR / "recovery_by_trial.csv", recovery_rows,
             ["method", "events", "trial", "first_below_threshold",
              "mawr_at_true_n", "parameters_match_truth",
              "correct_selection_and_recovery"])
    save_csv(OUTPUT_DIR / "fitted_components.csv", fitted_rows,
             ["method", "events", "trial", "true_component", "fitted_component",
              "true_fraction", "fitted_fraction", "true_tau_us", "fitted_tau_us",
              "fraction_abs_error", "lifetime_rel_error", "recovered"])

    for method in methods:
        plot_mawr(method, all_runs, event_counts, n_max, n_true)

    fig, ax = plt.subplots(figsize=(9, 5))
    for method in methods:
        subset = [[r for r in recovery_rows
                   if r["method"] == method and r["events"] == count]
                  for count in event_counts]
        matched_rates = [np.mean([r["parameters_match_truth"] for r in rows])
                         for rows in subset]
        selected_rates = [np.mean([r["correct_selection_and_recovery"]
                                   for r in rows]) for rows in subset]
        line, = ax.plot(event_counts, matched_rates, "o--",
                        label=f"{method}: parameters recovered")
        ax.plot(event_counts, selected_rates, "o-", color=line.get_color(),
                label=f"{method}: MAWR selects true model")
        reached = [count for count, p in zip(event_counts, matched_rates)
                   if p >= REQUIRED_SUCCESS_RATE]
        selected = [count for count, p in zip(event_counts, selected_rates)
                    if p >= REQUIRED_SUCCESS_RATE]
        print(f"{method}: first tested N with >= {REQUIRED_SUCCESS_RATE:.0%} "
              "parameter recovery: "
              + (f"{reached[0]:,}" if reached else "none"))
        print(f"{method}: first tested N with >= {REQUIRED_SUCCESS_RATE:.0%} "
              "MAWR selection + recovery: "
              + (f"{selected[0]:,}" if selected else "none"))
    ax.axhline(REQUIRED_SUCCESS_RATE, color="gray", linestyle="--")
    ax.set_xscale("log")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Total detected events")
    ax.set_ylabel("Fraction of successful trials")
    ax.set_title("Component recovery and MAWR selection versus event count")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = OUTPUT_DIR / "recovery_probability.png"
    fig.savefig(out, dpi=180)
    print(f"Saved {out}")
    print(f"CSV results saved in: {OUTPUT_DIR.resolve()}")
    if N_TRIALS < 10:
        print("Note: run N_TRIALS >= 10 before interpreting success as a probability.")
    if SHOW_PLOTS and "agg" not in plt.get_backend().lower():
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
