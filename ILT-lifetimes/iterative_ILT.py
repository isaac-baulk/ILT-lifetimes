import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import lsq_linear, least_squares # Alex: add this import
from scipy.signal import find_peaks

from tikhonov_ILT_Alex import load_data
from L1_ILT_Alex import perform_ILT_lasso, alpha_method_2, optimise_alpha_lasso, plot_optimal, exponential_model

#quanitfy how many MEv needed for a run to separate components - given set of expected lifetimes and ratio of elements in sample
# check against exisitng files
#minimum noise is 0.8 for metric
# higher statistics should shift metric curve up
# make sure run starts early enough to capture silver lifetime


# look into period per point for bluesky steering

def do_iterative_lasso_ILT(data_path, num_elements=3, lambdas=(np.linspace(0.1, 1.7, 200)), target_mawr=None, gen_intermediate_plots=True, plot_metric_history=False):
    # use a forward modelling approach to iteratively add exponentials to the fit, using L1-regularized ILT to identify the next exponential to add

    t, raw_data, errors = load_data(data_path)
    found_components = {}

    if len(t) < 4:
        raise ValueError("Not enough data points above t=0.5")

    if np.any(raw_data < 0):
        raise ValueError("Poisson weights require nonnegative counts")

    # Normalize data
    scale_factor = np.max(raw_data)

    if scale_factor <= 0:
        raise ValueError("Data have no positive counts to fit")

    raw_data_scaled = raw_data / scale_factor

    # 1. Create a mask identifying where the loaded errors are zero
    zero_error_mask = (errors == 0)

    # 2. Calculate Poisson fallback errors (enforcing a minimum of 1 count to prevent sqrt(0))
    poisson_fallback = np.sqrt(np.maximum(raw_data, 1.0))

    # 3. Merge the arrays: use Poisson where mask is True, otherwise use raw_errors
    safe_raw_errors = np.where(zero_error_mask, poisson_fallback, errors)

    # 4. Scale the final safe errors to match the scaled raw data
    errors = safe_raw_errors / scale_factor

    # Initially there are no exponential components
    parameters = np.array([], dtype=float)
    #errors = np.sqrt(raw_data) / scale_factor

    # Initial residual equals original data
    current_data = raw_data_scaled.copy()

    # Metric for zero exponentials
    metric_history = [
        np.mean(np.abs(current_data) / errors)
    ]

    for iteration in range(num_elements):

        if 2 * (iteration + 1) >= len(t):
            print("Not enough data points for another exponential.")
            break

        print(f"\n--- Iteration {iteration + 1}/{num_elements} ---")

        # STEP 1: Perform ILT on the current residual
        _, optimal_alpha, _, ilt_dist, _ = optimise_alpha_lasso(current_data, t, lambdas)

        if np.max(ilt_dist) <= 1e-12:
            print("No positive ILT peak left. Stopping.")
            break

        # STEP 2: Identify the tallest peak in the ILT distribution
        peak_idx = int(np.argmax(ilt_dist))

        new_rate = float(lambdas[peak_idx])
        new_amplitude = max(float(ilt_dist[peak_idx]), 1e-9)

        print(f"Tallest ILT peak: lambda = {new_rate:.5f}")

        # STEP 3: Add a new exponential
        initial_guess = np.r_[parameters, new_amplitude, new_rate]

        # Amplitudes >= 0
        # Relaxation rates between 0 and max(lambdas)
        lower_bounds = np.tile([0.0, 0.0], iteration + 1)

        upper_bounds = np.tile(
            [np.inf, float(np.max(lambdas))],
            iteration + 1
        )

        # STEP 4: Globally refit ALL exponentials
        result = least_squares(lambda p: (exponential_model(t, p) - raw_data_scaled) / errors,

            x0=initial_guess,

            bounds=(
                lower_bounds,
                upper_bounds
            ),

            x_scale='jac',
            max_nfev=10000
        )

        # Updated parameters for ALL exponentials
        parameters = result.x

        # Global reconstruction
        fit = exponential_model(t, parameters)

        # -----------------------------------------------
        # STEP 5: Calculate the signed residual
        # -----------------------------------------------

        current_data = raw_data_scaled - fit

        # Mean absolute weighted residual
        mawr = np.mean(
            np.abs(current_data) / errors
        )

        metric_history.append(mawr)

        print(f"Weighted residual: {mawr:.6f}")

        # Calculate Covariance Matrix from the Jacobian
        # Use pseudo-inverse (pinv) to prevent crashes if the matrix is nearly singular
        cov = np.linalg.pinv(result.jac.T @ result.jac)
        param_errors = np.sqrt(np.diag(cov))

        # integrate under exponential for true amplitude
        for k in range(iteration + 1):
            A = parameters[2*k]
            rate = parameters[2*k + 1]
            
            A_err = param_errors[2*k]
            rate_err = param_errors[2*k + 1]
            
            tau = 1.0 / rate if rate > 0 else np.inf
            tau_err = (1.0 / rate**2) * rate_err if rate > 0 else 0.0

            print(
                f"Component {k+1}: \n"
                f"  A      = {A:.6f} +/- {A_err:.6f}\n"
                f"  lambda = {rate:.6f} +/- {rate_err:.6f} us^-1\n"
                f"  tau    = {tau*1e3:.3f} +/- {tau_err*1e3:.3f} ns"
            )

        found_components[iteration + 1] = {
            "amplitude": A,
            "rate": rate
        }

        # -----------------------------------------------
        # STEP 6: Display the results
        # -----------------------------------------------
        if gen_intermediate_plots:
            fig, (ax1, ax2, ax3) = plt.subplots(
                3, 1, figsize=(10, 9)
            )

            # Experimental data and global fit
            ax1.plot(
                t, raw_data_scaled,
                '.', markersize=3,
                label='Data'
            )

            ax1.plot(
                t, fit,
                '-', linewidth=2,
                label='Global fit'
            )

            # Show individual exponentials
            for k in range(iteration + 1):

                A = parameters[2*k]
                rate = parameters[2*k + 1]

                component = A * np.exp(-rate * t)

                ax1.plot(
                    t, component,
                    '--', alpha=0.7,
                    label=f'Exponential {k+1}'
                )

            ax1.set_xlabel('Time')
            ax1.set_ylabel('Scaled counts')
            ax1.set_title(
                f'Fit with {iteration + 1} exponentials'
            )
            ax1.legend()

            # Signed residual
            ax2.axhline(0, color='black', linewidth=0.8)
            ax2.plot(t, current_data, '.', markersize=3)

            ax2.set_xlabel('Time')
            ax2.set_ylabel('Data - Fit')
            ax2.set_title(f'Residual: MAWR = {mawr:.4f}')

            # ILT distribution used to initialize this iteration
            ax3.plot(
                lambdas, ilt_dist,
                label='ILT of previous residual'
            )

            ax3.plot(
                new_rate,
                ilt_dist[peak_idx],
                'ro',
                label='Selected peak'
            )

            ax3.set_xlabel('Relaxation rate')
            ax3.set_xscale('symlog', linthresh=0.01)
            ax3.set_ylabel('ILT amplitude')
            ax3.legend()

            plt.tight_layout()
            plt.show(block=False)
            plt.pause(0.05)

        # -----------------------------------------------
        # STEP 7: Stopping condition
        # -----------------------------------------------

        if np.max(np.abs(current_data)) < 1e-10:
            print("Residual is numerically zero.")
            break

        if target_mawr is not None and mawr <= target_mawr:
            print("Weighted residual target reached.")
            break

    # -----------------------------------------------
    # FINAL: Metric versus number of exponentials
    # -----------------------------------------------
    if plot_metric_history:
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

        plt.xlabel('Number of exponentials')
        plt.ylabel('Mean absolute weighted residual')
        plt.yscale('symlog', linthresh=0.01)
        plt.xticks(np.arange(len(metric_history)))
        plt.grid(alpha=0.3)

        if target_mawr is not None:
            plt.legend()

        plt.tight_layout()
        plt.show()

    return found_components, np.asarray(metric_history)

if __name__ == "__main__":

    input_file = 'data/mantid/81801-bwd-counts.dat'

    # Include lambda=0 to permit a constant exponential component
    lambdas = np.linspace(0, 2, 100)

    do_iterative_lasso_ILT(
        input_file,
        num_elements=4,
        lambdas=lambdas,
        target_mawr=None
    )