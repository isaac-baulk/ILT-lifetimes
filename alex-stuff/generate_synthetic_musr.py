"""Generate synthetic muon-decay time histograms compatible with the ILT scripts.

The ``STOICHIOMETRY`` entries are fractions or ratios of *recorded signal
events* (integrated exponential areas over the simulated time range), NOT
fractions of the initial amplitudes. The entries are normalized to sum to 1. The expected count rate is a mixture of exponential
probability densities and an optional flat background.

Output format, like 81801-bwd-counts.dat:
    time_us    integer_counts    uncertainty_counts

A multinomial draw fixes the TOTAL_EVENTS count exactly; individual bins
have approximately Poisson-like noise when their event probabilities are small.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


# =========================== EDIT HERE ================================
# Each stoichiometric fraction corresponds to the lifetime at the same index.
STOICHIOMETRY = [0.9, 0.1]       # Integrated fractions or ratios (e.g. [2, 1])
LIFETIMES_US = [0.80, 1.50]       # Lifetimes in microseconds

TOTAL_EVENTS = 10_000_000_000         # EXACT total, signal + background
BACKGROUND_COUNTS_PER_BIN = 10  # Flat accidental background, 0 to disable

START_TIME_NS = 100.0             # First time value in output: 100 ns
END_TIME_US = 32.0                # Last time value in output: 32 us
BIN_WIDTH_NS = 16.0               # Approximate time-bin spacing (reference: 16 ns)

RANDOM_SEED = 42                  # Repeatable statistical fluctuations
OUTPUT_FILE = Path(__file__).resolve().parent / 'data/synthetic-bwd-counts.dat'
PLOT_RESULTS = True
# ====================================================================== 


def generate_synthetic_data(
    stoichiometry,
    lifetimes_us,
    total_events,
    start_time_ns=100.0,
    end_time_us=32.0,
    bin_width_ns=16.0,
    background_counts_per_bin=25.0,
    seed=42,
):
    """Return (times_us, counts, sigma_counts, expected_counts, components).

    Each component's expected signal counts sum to
    ``stoichiometry[i] * (total_events - expected_background_events)``.

    The first/last time points are exactly the requested values. To do so,
    the bin spacing is adjusted slightly from the nominal bin_width_ns.
    """
    fractions = np.asarray(stoichiometry, dtype=float)
    taus = np.asarray(lifetimes_us, dtype=float)

    if (fractions.ndim != 1 or taus.ndim != 1 or len(fractions) == 0
            or fractions.shape != taus.shape):
        raise ValueError('STOICHIOMETRY and LIFETIMES_US must be equally long lists.')
    if (not np.all(np.isfinite(fractions)) or np.any(fractions < 0)
            or fractions.sum() <= 0):
        raise ValueError('Stoichiometric fractions/ratios must be nonnegative with positive sum.')
    fractions = fractions / fractions.sum()
    if not np.all(np.isfinite(taus)) or np.any(taus <= 0):
        raise ValueError('Every lifetime must be finite and strictly positive.')
    if not isinstance(total_events, (int, np.integer)) or total_events <= 0:
        raise ValueError('TOTAL_EVENTS must be a positive integer.')

    t_start = start_time_ns / 1000.0  # ns -> us
    t_end = float(end_time_us)
    step_requested = bin_width_ns / 1000.0
    if (not np.isfinite(t_start) or not np.isfinite(t_end)
            or not np.isfinite(step_requested) or t_start < 0
            or t_end <= t_start or step_requested <= 0):
        raise ValueError('Invalid time limits or bin spacing.')
    if not np.isfinite(background_counts_per_bin) or background_counts_per_bin < 0:
        raise ValueError('BACKGROUND_COUNTS_PER_BIN must be finite and >= 0.')

    n_bins = max(2, int(round((t_end - t_start) / step_requested)) + 1)
    t = np.linspace(t_start, t_end, n_bins)
    actual_width = float(t[1] - t[0])

    # Each t is a bin centre. Compute edges to integrate exponentials per bin.
    edges = np.r_[t[0] - actual_width / 2,
                  0.5 * (t[:-1] + t[1:]),
                  t[-1] + actual_width / 2]
    widths = np.diff(edges)

    expected_background = background_counts_per_bin * n_bins
    signal_events = total_events - expected_background
    if signal_events <= 0:
        raise ValueError('Background uses all events: increase TOTAL_EVENTS or lower background.')

    component_counts = []
    for fraction, tau in zip(fractions, taus):
        # Integral of exp(-t/tau)/tau over each bin. Subtract the first edge
        # for numerical stability; normalization cancels that common factor.
        weights = (
            np.exp(-(edges[:-1] - edges[0]) / tau)
            * (-np.expm1(-widths / tau))
        )
        weights /= weights.sum()
        component_counts.append(signal_events * fraction * weights)

    components = np.asarray(component_counts)  # (component, time_bin)
    expected_counts = components.sum(axis=0) + background_counts_per_bin

    # EXACT fixed total event count across all output bins.
    probabilities = expected_counts / expected_counts.sum()
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(total_events, probabilities)

    # Poisson standard deviation for the measured counts; use 1 for empty bins
    # so weight_mode='file' in the Tikhonov fitter stays well defined.
    errors = np.sqrt(np.maximum(counts, 1))

    return t, counts, errors, expected_counts, components


def save_dataset(path, times_us, counts, errors):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        'Synthetic muon decay data\n'
        'Column 1: time (microseconds)\n'
        'Column 2: events per bin\n'
        'Column 3: sqrt(max(counts, 1))\n'
        'Stoichiometry (input weights): '
        + ', '.join(f'{x:.8g}' for x in STOICHIOMETRY) + '\n'
        'Normalized signal fractions: '
        + ', '.join(f'{x / sum(STOICHIOMETRY):.8g}' for x in STOICHIOMETRY) + '\n'
        'Lifetimes (microseconds): '
        + ', '.join(f'{x:.8g}' for x in LIFETIMES_US) + '\n'
        f'Total detected events (including background): {TOTAL_EVENTS}\n'
        f'Expected constant background (counts/bin): {BACKGROUND_COUNTS_PER_BIN:g}\n'
        'time_us counts sigma'
    )
    np.savetxt(
        path,
        np.column_stack((times_us, counts, errors)),
        fmt=['%.6f', '%d', '%.6f'],
        header=header,
        comments='# ',
    )
    return path


def plot_dataset(times_us, counts, expected_counts, components, output_file):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax1.plot(times_us, counts, '.', markersize=2, label='Simulated counts')
    ax1.plot(times_us, expected_counts, '-', linewidth=1.5,
             label='Expected counts')
    fractions = np.asarray(STOICHIOMETRY) / sum(STOICHIOMETRY)
    for idx, (curve, fraction, tau) in enumerate(
        zip(components, fractions, LIFETIMES_US), start=1
    ):
        ax1.plot(times_us, curve, '--', linewidth=1,
                 label=f'Component {idx}: {fraction:.0%}, tau={tau:g} us')
    if BACKGROUND_COUNTS_PER_BIN > 0:
        ax1.axhline(BACKGROUND_COUNTS_PER_BIN, linestyle=':', color='black',
                    label='Flat background')
    ax1.set_yscale('log')
    ax1.set_ylim(bottom=0.8)
    ax1.set_ylabel('Counts per bin')
    ax1.set_title('Synthetic muon decay histogram')
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.25)

    standard_residual = (
        (counts - expected_counts) / np.sqrt(np.maximum(expected_counts, 1))
    )
    ax2.plot(times_us, standard_residual, '.', markersize=2)
    ax2.axhline(0, color='black', linewidth=0.8)
    ax2.set_xlabel('Time (us)')
    ax2.set_ylabel('(counts - expected) / sqrt(expected)')
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    plot_path = Path(output_file).with_suffix('.png')
    fig.savefig(plot_path, dpi=160)
    if plt.get_backend().lower() != 'agg':
        plt.show()
    else:
        plt.close(fig)
    return plot_path


def main():
    t, counts, errors, expected_counts, components = generate_synthetic_data(
        stoichiometry=STOICHIOMETRY,
        lifetimes_us=LIFETIMES_US,
        total_events=TOTAL_EVENTS,
        start_time_ns=START_TIME_NS,
        end_time_us=END_TIME_US,
        bin_width_ns=BIN_WIDTH_NS,
        background_counts_per_bin=BACKGROUND_COUNTS_PER_BIN,
        seed=RANDOM_SEED,
    )
    output = save_dataset(OUTPUT_FILE, t, counts, errors)
    print(f'Saved: {output.resolve()}')
    print(f'Time: {t[0]:.6f} to {t[-1]:.6f} us')
    print(f'Bins: {len(t)}, spacing: {(t[1]-t[0])*1000:.6f} ns')
    print(f'Events: {counts.sum():,} (requested {TOTAL_EVENTS:,})')
    print(f'Background: {BACKGROUND_COUNTS_PER_BIN:g} expected counts/bin')
    fractions = np.asarray(STOICHIOMETRY) / sum(STOICHIOMETRY)
    for i, (fraction, tau, component) in enumerate(
        zip(fractions, LIFETIMES_US, components), start=1
    ):
        print(f'Component {i}: fraction={fraction:.4f}, '
              f'lifetime={tau:g} us, lambda={1/tau:.6g} 1/us, '
              f'expected signal events={component.sum():.1f}')

    if PLOT_RESULTS:
        plot_path = plot_dataset(t, counts, expected_counts, components, output)
        print(f'Plot: {plot_path.resolve()}')


if __name__ == '__main__':
    main()
