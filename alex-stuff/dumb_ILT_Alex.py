"""Iterative multiexponential fit with a fixed ('dumb') initial guess.

CONTROL EXPERIMENT against L1_ILT_Alex.py / tikhonov_ILT_Alex.py:
    At each iteration, initialise ONE NEW exponential with A=0.5, lambda=1.0.
    Then refit ALL amplitudes and relaxation rates against the ORIGINAL data.

There is NO inverse Laplace transform and NO regularisation in this method.
The fixed values are INITIAL GUESSES ONLY, not the fitted results.

To keep the comparison fair, data selection, scaling, Poisson weighting,
nonnegative parameter bounds, global least_squares fit, MAWR definition,
and stopping criteria match tikhonov_ILT_Alex.py.

The model is sum(A_i * exp(-lambda_i*t)), WITHOUT a background parameter.
A component is allowed to converge to lambda=0 (constant contribution).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares


# =========================== CONFIGURATION ===========================
INITIAL_AMPLITUDE = 0.5  # Initial guess, in units of max(raw_data)
INITIAL_RATE = 1.0       # Initial guess, inverse units of the time column
# =====================================================================


def load_data(path):
    """Load time, counts, and optional third-column uncertainties."""
    data = np.loadtxt(path, comments="#", ndmin=2)
    if data.shape[1] < 2:
        raise ValueError("Data file must contain at least time and counts.")
    t = data[:, 0]
    counts = data[:, 1]
    errors = data[:, 2] if data.shape[1] >= 3 else None
    return t, counts, errors


def exponential_model(t, parameters):
    """params = [A1, lambda1, A2, lambda2, ...]; no separate background."""
    amplitudes = parameters[0::2]
    rates = parameters[1::2]
    if len(amplitudes) == 0:
        return np.zeros_like(t)
    return np.exp(-t[:, None] * rates[None, :]) @ amplitudes


def metric_func(residual, sigma):
    """Mean absolute weighted residual, identical to the other versions."""
    return float(np.mean(np.abs(residual) / sigma))


def guess_next_component():
    """Return the SAME initial parameters for EVERY new component."""
    return INITIAL_AMPLITUDE, INITIAL_RATE


def do_iterative_ILT(input_file, num_elements, lambdas=None,
                     target_mawr=None, weight_mode="poisson", plot=True):
    """Fit exponentials iteratively without any ILT-based initialisation.

    The signature and outputs match tikhonov_ILT_Alex.do_iterative_ILT().

    Parameters
    ----------
    input_file : str or pathlib.Path
        Data with columns time, counts, [optional sigma].
    num_elements : int
        Maximum number of exponential components.
    lambdas : array_like or None
        Used ONLY to specify the same maximum rate / fit bounds as the ILT
        methods; this method does not use a relaxation-rate grid otherwise.
        If omitted, uses np.linspace(0.0, 1.7, 300).
    target_mawr : float or None
        Stop when MAWR <= target; None to stop only at the max component
        count or at a numerically zero residual.
    weight_mode : {'poisson', 'file', 'uniform'}
        'poisson' is the default and matches both ILT scripts.
        'file' uses column 3, 'uniform' uses sigma=1 (scaled units).
    plot : bool
        Show the 3-panel plot at each iteration and final MAWR plot.

    Returns
    -------
    parameters : numpy.ndarray
        [A1, lambda1, A2, lambda2, ...], fitted globally; NOT fixed guesses.
    metric_history : numpy.ndarray
        MAWR for 0, 1, ..., n fitted components.
    """
    # STEP 0: Read exactly the same time range as both ILT implementations.
    t, raw_data, file_errors = load_data(input_file)
    t_mask = t > 0.5
    t = t[t_mask]
    raw_data = raw_data[t_mask]
    if file_errors is not None:
        file_errors = file_errors[t_mask]

    if len(t) < 4:
        raise ValueError("Not enough data points above t = 0.5.")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(raw_data)):
        raise ValueError("Time and counts must be finite.")

    if lambdas is None:
        lambdas = np.linspace(0.0, 1.7, 300)
    lambdas = np.asarray(lambdas, dtype=float)
    if (lambdas.ndim != 1 or len(lambdas) < 3
            or not np.all(np.isfinite(lambdas))
            or np.any(lambdas < 0)
            or np.any(np.diff(lambdas) <= 0)
            or lambdas[-1] <= 0):
        raise ValueError(
            "lambdas must be strictly increasing, nonnegative, "
            "and include a positive rate."
        )
    if lambdas[-1] < INITIAL_RATE:
        raise ValueError(
            f"The comparison rate bound max(lambdas)={lambdas[-1]:g} "
            f"is below the fixed initial rate {INITIAL_RATE:g}. "
            "Increase the maximum fitted rate to at least INITIAL_RATE."
        )
    if num_elements < 1:
        raise ValueError("num_elements must be at least 1.")

    scale_factor = float(np.max(raw_data))
    if scale_factor <= 0:
        raise ValueError("The data must have positive counts for scaling.")
    raw_data_scaled = raw_data / scale_factor

    # Identical uncertainties to tikhonov_ILT_Alex.py.
    if weight_mode == "poisson":
        if np.any(raw_data < 0):
            raise ValueError("Poisson weights require nonnegative counts.")
        sigma = np.sqrt(np.maximum(raw_data, 1.0)) / scale_factor
    elif weight_mode == "file":
        if file_errors is None:
            raise ValueError(
                "weight_mode='file' requires a third (uncertainty) column."
            )
        if np.any(~np.isfinite(file_errors)) or np.any(file_errors <= 0):
            raise ValueError(
                "All supplied uncertainties must be positive and finite."
            )
        sigma = file_errors / scale_factor
    elif weight_mode == "uniform":
        sigma = np.ones_like(raw_data_scaled)
    else:
        raise ValueError(
            "weight_mode must be 'poisson', 'file', or 'uniform'."
        )

    parameters = np.array([], dtype=float)
    current_data = raw_data_scaled.copy()
    metric_history = [metric_func(current_data, sigma)]  # 0 exponentials

    for iteration in range(num_elements):
        n_components = iteration + 1
        if 2 * n_components >= len(t):
            print("Not enough data points for another exponential.")
            break
        print(
            f"\n--- Iteration {n_components}/{num_elements}: "
            "DUMB initialization ---"
        )

        # STEP 1: NO ILT. Start EVERY NEW component from the same guess.
        new_amplitude, new_rate = guess_next_component()
        print(
            f"New component INITIAL GUESS: A={new_amplitude:.6f}, "
            f"lambda={new_rate:.6f}"
        )

        # STEP 2: Retain all previously fitted parameters as starting guesses.
        initial_guess = np.r_[parameters, new_amplitude, new_rate]
        lower_bounds = np.tile([0.0, 0.0], n_components)
        upper_bounds = np.tile([np.inf, lambdas[-1]], n_components)

        # STEP 3: Refit ALL amplitudes and rates, against ORIGINAL data.
        result = least_squares(
            lambda p: (
                exponential_model(t, p) - raw_data_scaled
            ) / sigma,
            x0=initial_guess,
            bounds=(lower_bounds, upper_bounds),
            x_scale='jac',
            max_nfev=10000,
        )
        if not result.success:
            print(
                f"Warning: nonlinear least-squares solver: "
                f"{result.message}"
            )
        parameters = result.x
        fit = exponential_model(t, parameters)

        # STEP 4: Signed residual and same MAWR as both ILT methods.
        current_data = raw_data_scaled - fit
        mawr = metric_func(current_data, sigma)
        metric_history.append(mawr)
        print(f"Mean absolute weighted residual = {mawr:.6f}")
        for k in range(n_components):
            print(
                f"  Component {k + 1}: A = {parameters[2*k]:.6f}, "
                f"lambda = {parameters[2*k+1]:.6f}"
            )

        # STEP 5: Plots. Bottom panel substitutes initial vs fitted values
        # for the ILT spectrum, since no ILT is calculated.
        if plot:
            fig, (ax1, ax2, ax3) = plt.subplots(
                3, 1, figsize=(10, 9)
            )
            ax1.plot(
                t, raw_data_scaled, '.', markersize=3, label='Data'
            )
            ax1.plot(
                t, fit, '-', linewidth=2, label='Global fit'
            )
            for k in range(n_components):
                A = parameters[2*k]
                rate = parameters[2*k+1]
                ax1.plot(
                    t,
                    A * np.exp(-rate * t),
                    '--',
                    alpha=0.7,
                    label=f'Exponential {k + 1}'
                )
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Scaled counts')
            ax1.set_title(
                f'Dumb initialization + global fit: '
                f'{n_components} exponentials'
            )
            ax1.legend()

            ax2.axhline(
                0, color='black', linewidth=0.8
            )
            ax2.plot(
                t, current_data, '.', markersize=3
            )
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Data - Fit')
            ax2.set_title(
                f'Signed residual: MAWR = {mawr:.4f}'
            )

            ax3.plot(
                INITIAL_RATE,
                INITIAL_AMPLITUDE,
                'rx',
                markersize=9,
                label='Fixed initial guess for new component'
            )
            ax3.scatter(
                parameters[1::2],
                parameters[0::2],
                facecolors='none',
                edgecolors='C0',
                s=65,
                label='Globally fitted components'
            )
            ax3.set_xscale(
                'symlog', linthresh=0.01
            )
            ax3.set_xlim(left=0)
            ax3.set_xlabel('Relaxation rate')
            ax3.set_ylabel('Scaled amplitude')
            ax3.set_title(
                'Initial guess vs fitted parameters (no ILT)'
            )
            ax3.legend()
            fig.tight_layout()
            plt.show(block=False)
            plt.pause(0.05)

        # STEP 6: Same stopping criteria as the Tikhonov/L1 scripts.
        if np.max(np.abs(current_data)) < 1e-10:
            print("Residual is numerically zero.")
            break
        if target_mawr is not None and mawr <= target_mawr:
            print("Weighted residual target reached.")
            break

    if plot:
        plt.figure(figsize=(7, 4))
        plt.plot(
            np.arange(len(metric_history)),
            metric_history,
            'o-'
        )
        if target_mawr is not None:
            plt.axhline(
                target_mawr,
                color='gray',
                linestyle='--',
                label='Noise target'
            )
            plt.legend()
        plt.xlabel('Number of exponentials')
        plt.ylabel('Mean absolute weighted residual')
        plt.title('Dumb initialization: convergence')
        plt.yscale('symlog', linthresh=0.01)
        plt.xticks(np.arange(len(metric_history)))
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    return parameters, np.asarray(metric_history)


# Optional explicit name; do_iterative_ILT is kept for selector compatibility.
do_iterative_dumb_ILT = do_iterative_ILT


if __name__ == "__main__":
    input_file = 'data/81801-bwd-counts.dat'
    lambdas = np.linspace(0.0, 1.7, 300)
    parameters, metric_history = do_iterative_ILT(
        input_file,
        num_elements=6,
        lambdas=lambdas,
        target_mawr=None,      # Or 0.9 to stop at the first noise-level fit.
        weight_mode="poisson",
        plot=True,
    )