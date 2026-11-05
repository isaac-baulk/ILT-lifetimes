"""Find and scan the event count needed for N exponentials with simple MAWR.

Requires the existing files in the same folder:
    generate_synthetic_musr.py
    sweep_synthetic_ILT.py       (provides run_fit without iteration plots)
    L1_ILT_Alex.py, tikhonov_ILT_Alex.py, dumb_ILT_Alex.py
    select_ILT_components.py     (used by sweep_synthetic_ILT.py)

Definition of 'N components detected':
    MAWR with N components < threshold
    AND the first model below threshold has EXACTLY N components.
In particular, accepting N=2 requires MAWR(1) >= threshold and MAWR(2) < threshold.
This uses the SAME simple MAWR rule as select_ILT_components.py; it does
not test recovery of the known synthetic lifetimes or fractions.

Synthetic data use one reproducible RNG seed. Results can fluctuate with
seed/count, so a 1% bisection bracket is NOT 1% statistical confidence.
"""

from pathlib import Path
from tempfile import TemporaryDirectory
import csv

import matplotlib.pyplot as plt
import numpy as np

from generate_synthetic_musr import generate_synthetic_data
from sweep_synthetic_ILT import run_fit


# =========================== CONFIGURATION ===========================

METHODS = ("dumb",)             # e.g. ("lasso", "tikhonov", "dumb")
STOICHIOMETRY = [0.5, 0.5]  # Integrated signal fractions (or ratios)
LIFETIMES_US = [0.8, 1.2]     # One per signal fraction

# Leave None to use len(STOICHIOMETRY), plus 1 if there is a background.
# Or set explicitly to the number of exponentials you want to detect.
TARGET_COMPONENTS = None

INITIAL_EVENTS = 1_000_000
MAX_EVENTS = 1_000_000_000_000
MIN_EVENTS = 1_000
RELATIVE_RESOLUTION = 0.01      # 1% precision on the event-count bracket
MAWR_THRESHOLD = 0.9
RANDOM_SEED = 42

START_TIME_NS = 100.0
END_TIME_US = 32.0
BIN_WIDTH_NS = 16.0
BACKGROUND_FRACTION = 0.0      # Fraction of TOTAL events (constant background)
LAMBDAS = np.linspace(0.0, 1.7, 60)

# Scan either "stoichiometry" or "lifetime" for one component.
SCAN_PARAMETER = "lifetime"
SCAN_INDEX = 1                 # Zero-based (1 means the second component)
SCAN_VALUES = np.linspace(0.81, 1.2, 100)
# Example lifetime scan: SCAN_PARAMETER="lifetime", SCAN_VALUES=np.linspace(2, 8, 8)

OUTPUT_DIR = Path(__file__).resolve().parent / "ilt_detection_limit"
SHOW_PLOT = True
# =====================================================================


def _validate_components(stoichiometry, lifetimes_us):
    weights = np.asarray(stoichiometry, dtype=float)
    lifetimes = np.asarray(lifetimes_us, dtype=float)
    if (weights.ndim != 1 or lifetimes.ndim != 1
            or len(weights) < 1 or weights.shape != lifetimes.shape):
        raise ValueError("STOICHIOMETRY and LIFETIMES_US must have equal nonzero length.")
    if (not np.isfinite(weights).all() or np.any(weights < 0)
            or weights.sum() <= 0):
        raise ValueError("Stoichiometry must be nonnegative with positive sum.")
    if not np.isfinite(lifetimes).all() or np.any(lifetimes <= 0):
        raise ValueError("Lifetimes must be finite and strictly positive (us).")
    return weights / weights.sum(), lifetimes


def _number_of_bins():
    return max(2, int(round(
        (END_TIME_US - START_TIME_NS / 1000.0) / (BIN_WIDTH_NS / 1000.0)
    )) + 1)


def find_required_events(stoichiometry, lifetimes_us, *, method="dumb",
                         target_components=None, initial_events=1_000_000,
                         threshold=0.9, resolution=0.01, max_events=1_000_000_000,
                         min_events=1_000, seed=42, background_fraction=0.0,
                         lambdas=None):
    """Return event count needed to select exactly N components via MAWR.

    Doubles the count until first success; if initial count already succeeds,
    halves it to establish a failure below. Bisects until (high-low)/high<=resolution.

    Returns a dictionary:
        events : minimum successful count within the bracketing resolution,
                 or None if none was found up to max_events.
        lower, upper : failing/succeeding bracket (upper=None if not found)
        evaluations : list of checked counts and MAWR at N-1 and N

    A fixed seed makes evaluations reproducible, but the success predicate may
    still be non-monotonic as the counts change (counting fluctuations and
    nonlinear optimization). This is a numerical threshold estimate, not a
    guaranteed physical detection limit or statistical confidence interval.
    """
    fractions, taus = _validate_components(stoichiometry, lifetimes_us)
    method = method.lower()
    if method not in ("lasso", "tikhonov", "dumb"):
        raise ValueError("method must be 'lasso', 'tikhonov', or 'dumb'.")
    if not 0 <= background_fraction < 1:
        raise ValueError("background_fraction must be in [0,1).")
    n_target = (len(fractions) + int(background_fraction > 0)
                if target_components is None else int(target_components))
    if n_target < 1:
        raise ValueError("target_components must be >= 1.")
    if not (0 < threshold and 0 < resolution < 1):
        raise ValueError("threshold > 0 and resolution in (0,1) required.")
    if not (1 <= min_events <= initial_events <= max_events):
        raise ValueError("Require 1 <= min_events <= initial_events <= max_events.")
    if lambdas is None:
        lambdas = LAMBDAS
    lambdas = np.asarray(lambdas, dtype=float)
    if (lambdas.ndim != 1 or not np.isfinite(lambdas).all()
            or np.any(np.diff(lambdas) <= 0) or lambdas[0] < 0):
        raise ValueError("lambdas must be a finite, increasing nonnegative grid.")

    n_bins = _number_of_bins()
    checked = {}  # cache repeated N evaluations during search

    with TemporaryDirectory(prefix="musr_detection_") as workdir:
        input_path = Path(workdir) / "synthetic.dat"

        def evaluate(n_events):
            n_events = int(n_events)
            if n_events in checked:
                return checked[n_events]

            background_per_bin = (background_fraction * n_events / n_bins)
            t, counts, sigma, _, _ = generate_synthetic_data(
                stoichiometry=fractions,
                lifetimes_us=taus,
                total_events=n_events,
                start_time_ns=START_TIME_NS,
                end_time_us=END_TIME_US,
                bin_width_ns=BIN_WIDTH_NS,
                background_counts_per_bin=background_per_bin,
                seed=seed,
            )
            np.savetxt(input_path, np.column_stack((t, counts, sigma)),
                       fmt=["%.9f", "%d", "%.9f"],
                       header="time_us counts sigma_counts")

            _params, history = run_fit(method, input_path, n_target, lambdas)
            if not np.isfinite(history).all():
                raise ValueError("The fitter returned non-finite MAWR values.")

            crossed = np.flatnonzero(history < threshold)
            first_below = int(crossed[0]) if len(crossed) else None
            selected_exactly_n = (first_below == n_target)
            row = {
                "events": n_events,
                "mawr_n_minus_1": (
                    float(history[n_target - 1])
                    if len(history) > n_target - 1 else float("nan")
                ),
                "mawr_n": (
                    float(history[n_target])
                    if len(history) > n_target else float("nan")
                ),
                "selected_components": first_below,
                "detected": bool(selected_exactly_n),
            }
            checked[n_events] = row
            print(f"  {method:9s} events={n_events:>12,} | "
                  f"MAWR({n_target-1})={row['mawr_n_minus_1']:.4f} | "
                  f"MAWR({n_target})={row['mawr_n']:.4f} | "
                  f"first below={first_below} | "
                  f"{'DETECTED' if selected_exactly_n else 'not detected'}")
            return row

        start = int(initial_events)
        maximum = int(max_events)
        minimum = int(min_events)
        # Locate a failing lower bound and a successful upper bound.
        if evaluate(start)["detected"]:
            high = start
            low = minimum - 1
            while high > minimum:
                smaller = max(minimum, high // 2)
                if smaller >= high:
                    break
                if evaluate(smaller)["detected"]:
                    high = smaller
                else:
                    low = smaller
                    break
        else:
            low = start
            high = start
            while high < maximum:
                larger = min(maximum, high * 2)
                if evaluate(larger)["detected"]:
                    high = larger
                    break
                low = larger
                high = larger
            else:
                return {
                    "events": None,
                    "lower": low,
                    "upper": None,
                    "evaluations": sorted(checked.values(), key=lambda row: row["events"]),
                }

        # lower is unsuccessful; upper is successful (or lower=min_events-1).
        while high - low > 1 and (high - low) / high > resolution:
            middle = (low + high) // 2
            if evaluate(middle)["detected"]:
                high = middle
            else:
                low = middle

    return {
        "events": high,
        "lower": low,
        "upper": high,
        "evaluations": sorted(checked.values(), key=lambda row: row["events"]),
    }


def change_component_parameter(stoichiometry, lifetimes_us, parameter,
                               index, value):
    """Return modified fractions/lifetimes without changing the originals.

    For a stoichiometry scan, set selected fraction to `value`, and distribute
    1-value among the other components in proportion to their original ratios.
    Example: [0.7, 0.2, 0.1], index=1, value=0.4 -> [0.525, 0.4, 0.075].
    """
    fractions, taus = _validate_components(stoichiometry, lifetimes_us)
    if not 0 <= index < len(fractions):
        raise IndexError("Component index is outside the component list.")

    if parameter.lower() in ("stoichiometry", "fraction"):
        if len(fractions) < 2:
            raise ValueError("A stoichiometry scan requires at least 2 components.")
        if not 0 <= value <= 1:
            raise ValueError("Stoichiometric fraction must be between 0 and 1.")
        others = np.arange(len(fractions)) != index
        other_sum = fractions[others].sum()
        if other_sum <= 0 and value != 1:
            raise ValueError("Other fractions sum to zero; cannot preserve ratios.")
        new_fractions = fractions.copy()
        new_fractions[index] = float(value)
        if other_sum > 0:
            new_fractions[others] *= (1.0 - value) / other_sum
        # remove possible rounding drift without changing the selected fraction
        return new_fractions, taus

    if parameter.lower() in ("lifetime", "life-time", "tau"):
        if not np.isfinite(value) or value <= 0:
            raise ValueError("Lifetime values must be finite and >0.")
        new_taus = taus.copy()
        new_taus[index] = float(value)
        return fractions, new_taus

    raise ValueError("parameter must be 'stoichiometry' or 'lifetime'.")


def scan_component_parameter(stoichiometry, lifetimes_us, *, parameter,
                             component_index, values, methods=("dumb",),
                             output_dir=OUTPUT_DIR, **search_kwargs):
    """Scan one component's stoichiometry or lifetime and plot event threshold.

    Returns rows of results (also saves a CSV + PNG). All methods are drawn
    on the same figure. `**search_kwargs` go to find_required_events().
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("values must be a nonempty list of finite numbers.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    checked_rows = []

    curves = {}
    for method in methods:
        event_limits = []
        for val in values:
            fractions, taus = change_component_parameter(
                stoichiometry, lifetimes_us, parameter, component_index, float(val)
            )
            print(f"\n{method.upper()} | {parameter}[{component_index}] = {val:.5g} "
                  f"| fractions={np.round(fractions, 4)} "
                  f"| taus={np.round(taus, 4)} us")
            outcome = find_required_events(
                fractions, taus, method=method, **search_kwargs
            )
            events = outcome["events"]
            event_limits.append(float(events) if events is not None else np.nan)
            rows.append({
                "method": method,
                "scan_parameter": parameter,
                "component_index_zero_based": component_index,
                "value": float(val),
                "fractions": ";".join(f"{f:.8g}" for f in fractions),
                "lifetimes_us": ";".join(f"{tau:.8g}" for tau in taus),
                "required_events": events if events is not None else "",
                "bracket_lower": outcome["lower"],
                "bracket_upper": (outcome["upper"] if outcome["upper"] is not None else ""),
            })
            for e in outcome["evaluations"]:
                checked_rows.append({"method": method, "scan_value": float(val), **e})

        curves[method] = event_limits

    # Create the scan plot only AFTER the fits: the L1 fitter suppresses its
    # own figures by closing them, which would also close an existing scan plot.
    fig, ax = plt.subplots(figsize=(9, 5))
    for method, event_limits in curves.items():
        ax.plot(values, event_limits, "o-", label=method)

    ax.set_yscale("log")
    ax.set_xlabel(
        f"Stoichiometric fraction of component {component_index + 1}"
        if parameter.lower() in ("stoichiometry", "fraction")
        else f"Lifetime of component {component_index + 1} (µs)"
    )
    ax.set_ylabel("Events required to select N components (MAWR criterion)")
    ax.set_title("Muon-event detection threshold")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    kind = "stoichiometry" if parameter.lower() in ("stoichiometry", "fraction") else "lifetime"
    plot_path = output_dir / f"required_events_vs_{kind}.png"
    fig.savefig(plot_path, dpi=180)

    csv_path = output_dir / f"required_events_vs_{kind}.csv"
    columns = ["method", "scan_parameter", "component_index_zero_based", "value",
               "fractions", "lifetimes_us", "required_events",
               "bracket_lower", "bracket_upper"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    trace_path = output_dir / f"search_evaluations_{kind}.csv"
    with trace_path.open("w", newline="", encoding="utf-8") as f:
        cols = ["method", "scan_value", "events", "mawr_n_minus_1", "mawr_n",
                "selected_components", "detected"]
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(checked_rows)
    print(f"\nSaved plot: {plot_path}")
    print(f"Saved results: {csv_path}")
    print(f"Saved search checks: {trace_path}")
    if SHOW_PLOT:
        plt.show()
    else:
        plt.close(fig)
    return rows


if __name__ == "__main__":
    scan_component_parameter(
        STOICHIOMETRY,
        LIFETIMES_US,
        parameter=SCAN_PARAMETER,
        component_index=SCAN_INDEX,
        values=SCAN_VALUES,
        methods=METHODS,
        output_dir=OUTPUT_DIR,
        target_components=TARGET_COMPONENTS,
        initial_events=INITIAL_EVENTS,
        min_events=MIN_EVENTS,
        max_events=MAX_EVENTS,
        threshold=MAWR_THRESHOLD,
        resolution=RELATIVE_RESOLUTION,
        seed=RANDOM_SEED,
        background_fraction=BACKGROUND_FRACTION,
        lambdas=LAMBDAS,
    )
