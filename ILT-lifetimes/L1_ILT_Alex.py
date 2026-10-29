"""Iterate ILT using Lasso regularization to extract multiple elemental peaks from noisy data."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import lsq_linear, least_squares # Alex: add this import
from scipy.signal import find_peaks
from sklearn.linear_model import Lasso
import warnings
from sklearn.exceptions import ConvergenceWarning
# Suppress warnings if the solver hits the iteration limit on heavily penalized steps
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def load_data(path):
    data = np.loadtxt(path, comments="#", usecols=(0, 1), ndmin=2) 
    t = data[:, 0] 
    raw_data = data[:, 1] 
    return t, raw_data


def perform_ILT_lasso(alpha, K, raw_data):

    lasso = Lasso(
        alpha=alpha,
        positive=True,
        fit_intercept=False,
        max_iter=100000
    )

    lasso.fit(K, raw_data)

    return lasso.coef_

# def perform_ILT_lasso(alpha, K, raw_data):
#     # positive=True enforces physical reality
#     # fit_intercept=True automatically fits the unpenalized flat background
#     lasso = Lasso(alpha=alpha, positive=True, fit_intercept=True, max_iter=100000)
#     lasso.fit(K, raw_data)
    
#     # Pack the coefficients and the intercept into a single array 
#     # to maintain compatibility with the rest of your tracking script
#     dist = np.append(lasso.coef_, lasso.intercept_)
#     return dist

def alpha_method_2(chi_squared, alphas, norms, plot=False):
    chi = np.asarray(chi_squared) 
    norms = np.asarray(norms) 
    alphas = np.asarray(alphas) 
     
    # chi_norm = (chi - np.min(chi)) / (np.max(chi) - np.min(chi)) 
    # norm_norm = (norms - np.min(norms)) / (np.max(norms) - np.min(norms)) 
    # Alex: Make the L-curve robust when the residual becomes small
    chi_norm = (chi - np.min(chi)) / max(np.ptp(chi), 1e-30)
    norm_norm = (norms - np.min(norms)) / max(np.ptp(norms), 1e-30)

    dx = np.gradient(chi_norm) 
    dy = np.gradient(norm_norm) 
    ddx = np.gradient(dx) 
    ddy = np.gradient(dy) 
    
    # curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2)**(1.5) 
    # optimal_idx = np.argmax(curvature)
    denominator = (dx**2 + dy**2)**1.5

    curvature = np.divide(
        np.abs(dx * ddy - dy * ddx),
        denominator,
        out=np.zeros_like(dx),
        where=denominator > 1e-12
    )
    
    optimal_idx = int(np.argmax(curvature[1:-1])) + 1
    optimal_alpha = alphas[optimal_idx] 
    
    if plot:
        optimal_chi2 = chi[optimal_idx]
        optimal_norm = norms[optimal_idx]
        plt.figure(figsize=(8, 6)) 
        plt.plot(chi, norms, 'bo-', linewidth=1.5, markersize=5, label='L-curve') 
        plt.plot(optimal_chi2, optimal_norm, 'ro', markersize=9, 
                 label=f'Optimal $\\alpha$ = {optimal_alpha:.2e}') 
        plt.axvline(optimal_chi2, color='r', linestyle='--', alpha=0.5) 
        plt.axhline(optimal_norm, color='r', linestyle='--', alpha=0.5)
        plt.xlabel(r'$\chi^2$', fontsize=12) 
        plt.ylabel(r'$\|\rho\|$', fontsize=12)
        plt.title('Regularization Trade-off Curve (L-Curve)', fontsize=14) 
        plt.legend(fontsize=10) 
        plt.grid(True, alpha=0.3) 
        plt.tight_layout() 
        plt.show() 

    return optimal_alpha, optimal_idx

def optimise_alpha_lasso(raw_data, t, lambdas, plot_lcurve=False): 
    T, L = np.meshgrid(t, lambdas, indexing='ij') 
    
    # Pure exponential kernel. No column of 1s needed; Lasso handles the intercept.
    K = np.exp(-L * T) 

    # Sklearn's Lasso scales the penalty by the number of samples, 
    # so the optimal alpha values will be mathematically smaller here.
    #alphas = np.logspace(-7, -3, 50)  
    alpha_max = max(
        0.0,
        float(np.max(K.T @ raw_data) / len(raw_data))
    ) # Alex: Automatically adapts the Lasso regularization to the residual being analyzed

    if alpha_max == 0:
        return lambdas, 0.0, np.zeros_like(raw_data), np.zeros(len(lambdas)), K
    
    alphas = alpha_max * np.logspace(-5, -0.15, 35)
    chi_squared = [] 
    reconstructions = [] 
    distributions = [] 
    norms = [] 

    for alpha in alphas: 
        dist = perform_ILT_lasso(alpha, K, raw_data) 
        distributions.append(dist) 
        
        # Strip the intercept (background) off before calculating the L1 norm
        dist_rates = dist # Alex [:-1]
        
        # For L1 regularization, the norm is the sum of absolute values, not squares
        norm = np.sum(np.abs(dist_rates))
        norms.append(norm) 
        
        # Reconstruct the fit: (Kernel * rates) + background
        reconstructed = (K @ dist_rates) # Alex: + dist[-1]
        reconstructions.append(reconstructed) 
        
        residual = reconstructed - raw_data 
        chi2 = np.sum(residual ** 2) 
        chi_squared.append(chi2) 

    optimal_alpha, optimal_idx = alpha_method_2(chi_squared, alphas, norms, plot=plot_lcurve) 

    optimal_reconstruction = reconstructions[optimal_idx] 
    optimal_dist = distributions[optimal_idx] 
    
    #fitted_bg = optimal_dist[-1]
    #optimal_dist = optimal_dist[:-1]

    return lambdas, optimal_alpha, optimal_reconstruction, optimal_dist, K

def alpha_sweep_lasso(raw_data, lambdas, K, optimal_alpha, sweep_factor=3.0, steps=20):
    sweep_alphas = np.logspace(np.log10(optimal_alpha / sweep_factor), 
                               np.log10(optimal_alpha * sweep_factor), steps)
    
    sweep_distributions = []
    print(f"Running Alpha Sweep over {steps} values (from {sweep_alphas[0]:.2e} to {sweep_alphas[-1]:.2e})...")
    
    for alpha in sweep_alphas:
        # Call the new lasso function (Gamma and padded d are gone)
        dist = perform_ILT_lasso(alpha, K, raw_data)
        dist_rates = dist[:-1] 
        
        area = np.trapezoid(dist_rates, lambdas)
        if area > 0:
            dist_rates = dist_rates / area
            
        sweep_distributions.append(dist_rates)
        
    return np.array(sweep_distributions)

def track_elemental_peaks_sweep(optimal_dist, lambdas, sweep_distributions, num_elements):
    peaks, _ = find_peaks(optimal_dist, prominence=0.01, distance=12)
    
    if len(peaks) == 0:
        print("No significant peaks found in the distribution.")
        return None
        
    peak_heights = optimal_dist[peaks]
    sorted_indices = np.argsort(peak_heights)[::-1]
    top_peaks = np.sort(peaks[sorted_indices][:num_elements])
    
    results = []
    
    for i, peak_idx in enumerate(top_peaks):
        optimal_rate = lambdas[peak_idx]
        
        window_mask = (lambdas > optimal_rate / 1.5) & (lambdas < optimal_rate * 1.5)
        tracked_rates = []
        
        for dist in sweep_distributions:
            window_dist = dist[window_mask]
            window_lambdas = lambdas[window_mask]
            
            if len(window_dist) > 0:
                local_max_idx = np.argmax(window_dist)
                tracked_rates.append(window_lambdas[local_max_idx])
            
        rates = np.array(tracked_rates)
        
        # Because this is deterministic, we use absolute min and max shifts, not percentiles
        rate_min = np.min(rates)
        rate_max = np.max(rates)
        
        tau_opt = 1.0 / optimal_rate
        tau_min = 1.0 / rate_max # highest rate = shortest lifetime
        tau_max = 1.0 / rate_min # lowest rate = longest lifetime
        
        results.append({
            'rate_opt': optimal_rate,
            'rate_err_plus': rate_max - optimal_rate,
            'rate_err_minus': optimal_rate - rate_min,
            'tau_opt': tau_opt,
            'tau_err_plus': tau_max - tau_opt,
            'tau_err_minus': tau_opt - tau_min
        })
        
        print(f"--- Element Peak (~{optimal_rate:.2f} us^-1) ---")
        print(f"Rate:     {optimal_rate:.3f} +{rate_max - optimal_rate:.3f} -{optimal_rate - rate_min:.3f} us^-1")
        print(f"Lifetime: {tau_opt*1e3:.3f} +{(tau_max - tau_opt)*1e3:.3f} -{(tau_opt - tau_min)*1e3:.3f} ns\n")
        
    return results

def plot_optimal(t, raw_data, lambdas, optimal_reconstruction, optimal_dist, element_data=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4)) 

    ax1.plot(t, raw_data, label='Noisy Data', alpha=0.5) 
    ax1.plot(t, optimal_reconstruction, 'r-', linewidth=2, label='ILT Fit (Optimal α)') 
    ax1.set_xlabel('Time (t)') 
    ax1.set_ylabel('Raw Counts')
    ax1.legend() 

    ax2.plot(lambdas, optimal_dist, 'b-', linewidth=2, label='P(λ)') 
    
    if element_data is not None:
        for i, elem in enumerate(element_data):
            rate = elem['rate_opt']
            err_minus = abs(elem['rate_err_minus'])
            err_plus = abs(elem['rate_err_plus'])
            
            idx = (np.abs(lambdas - rate)).argmin()
            peak_height = optimal_dist[idx]
            
            ax2.errorbar(rate, peak_height, xerr=[[err_minus], [err_plus]], 
                         fmt='o', color='g', capsize=4, markersize=7, 
                         label='Extracted Peaks' if i == 0 else "")

    ax2.set_xscale('log') 
    ax2.set_xlabel('Relaxation Rate (lambda)') 
    ax2.set_ylabel('Probability Density') 
    
    handles, labels = ax2.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax2.legend(by_label.values(), by_label.keys())

    plt.tight_layout() 
    plt.show() 

def do_alpha_sweep_ILT(input_file, num_elements, lambdas=(np.logspace(-1, np.log10(20), 100))):
    t, raw_data = load_data(input_file)

    t_mask = t > 0.5
    t = t[t_mask]
    raw_data = raw_data[t_mask]
    
    # 1. Initial optimal fit

    scale_factor = np.max(raw_data)
    raw_data = raw_data / scale_factor
    
    lambdas, optimal_alpha, optimal_reconstruction, optimal_dist, K, Gamma = optimise_alpha_lasso(raw_data, t, lambdas)
    print(f"\nInitial Optimal alpha value: {optimal_alpha:.6e}") 
    
    # 2. Normalize the initial distribution
    area = np.trapezoid(optimal_dist, lambdas)
    optimal_dist_norm = optimal_dist / area
    
    # 3. Run the deterministic Alpha Sweep
    # sweep_factor=3.0 tests alphas from optimal/3 to optimal*3. 
    sweep_distributions = alpha_sweep_lasso(raw_data, lambdas, K, Gamma, optimal_alpha, sweep_factor=3.0, steps=20)

    # 4. Extract peaks and track them across the sweep distributions
    element_data = track_elemental_peaks_sweep(optimal_dist_norm, lambdas, sweep_distributions, num_elements=num_elements)

    # 5. Plot the result
    plot_optimal(t, raw_data, lambdas, optimal_reconstruction, optimal_dist_norm, element_data)


def exponential_model(t, parameters):

    # parameters = [A1, lambda1, A2, lambda2, ...]

    amplitudes = parameters[0::2]
    rates = parameters[1::2]

    if not len(amplitudes):
        return np.zeros_like(t)

    return np.exp(-t[:, None] * rates[None, :]) @ amplitudes


def do_iterative_lasso_ILT(input_file, num_elements, lambdas,
                           target_mawr=None):

    # Load experimental data
    t, raw_data = load_data(input_file)

    t_mask = t > 0.5
    t = t[t_mask]
    raw_data = raw_data[t_mask]

    if len(t) < 4:
        raise ValueError("Not enough data points above t=0.5")

    if np.any(raw_data < 0):
        raise ValueError("Poisson weights require nonnegative counts")

    # Normalize data
    scale_factor = np.max(raw_data)

    if scale_factor <= 0:
        raise ValueError("Data have no positive counts to fit")

    raw_data_scaled = raw_data / scale_factor

    # Poisson uncertainty (assuming raw_data contains counts)
    sigma = np.sqrt(np.maximum(raw_data, 1.0)) / scale_factor

    # Initially there are no exponential components
    parameters = np.array([], dtype=float)

    # Initial residual equals original data
    current_data = raw_data_scaled.copy()

    # Metric for zero exponentials
    metric_history = [
        np.mean(np.abs(current_data) / sigma)
    ]

    for iteration in range(num_elements):

        if 2 * (iteration + 1) >= len(t):
            print("Not enough data points for another exponential.")
            break

        print(f"\n--- Iteration {iteration + 1}/{num_elements} ---")

        # -----------------------------------------------
        # STEP 1: Perform ILT on the current residual
        # -----------------------------------------------

        _, optimal_alpha, _, ilt_dist, _ = optimise_alpha_lasso(
            current_data, t, lambdas
        )

        if np.max(ilt_dist) <= 1e-12:
            print("No positive ILT peak left. Stopping.")
            break

        # -----------------------------------------------
        # STEP 2: Select the tallest ILT peak
        # -----------------------------------------------

        peak_idx = int(np.argmax(ilt_dist))

        new_rate = float(lambdas[peak_idx])
        new_amplitude = max(float(ilt_dist[peak_idx]), 1e-9)

        print(f"Tallest ILT peak: lambda = {new_rate:.5f}")

        # -----------------------------------------------
        # STEP 3: Add a new exponential
        # -----------------------------------------------

        initial_guess = np.r_[
            parameters,
            new_amplitude,
            new_rate
        ]

        # Amplitudes >= 0
        # Relaxation rates between 0 and max(lambdas)
        lower_bounds = np.tile(
            [0.0, 0.0], iteration + 1
        )

        upper_bounds = np.tile(
            [np.inf, float(np.max(lambdas))],
            iteration + 1
        )

        # -----------------------------------------------
        # STEP 4: Globally refit ALL exponentials
        # -----------------------------------------------

        result = least_squares(
            lambda p: (
                exponential_model(t, p) - raw_data_scaled
            ) / sigma,

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
            np.abs(current_data) / sigma
        )

        metric_history.append(mawr)

        print(f"Weighted residual: {mawr:.6f}")

        for k in range(iteration + 1):

            A = parameters[2*k]
            rate = parameters[2*k + 1]

            print(
                f"Component {k+1}: "
                f"A = {A:.6f}, "
                f"lambda = {rate:.6f}"
            )

        # -----------------------------------------------
        # STEP 6: Display the results
        # -----------------------------------------------

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

    return parameters, np.asarray(metric_history)

# def do_iterative_lasso_ILT(input_file, num_elements, lambdas):
#     t, raw_data = load_data(input_file)

#     t_mask = t > 0.5
#     t = t[t_mask]
#     raw_data = raw_data[t_mask]
    
#     # Scale data for Lasso
#     scale_factor = np.max(raw_data)
#     raw_data_scaled = raw_data / scale_factor
#     current_data = raw_data_scaled.copy()
    
#     all_element_data = []
#     cumulative_reconstruction = np.zeros_like(raw_data_scaled)
#     composite_dist = np.zeros(len(lambdas))
#     final_bg = 0

#     for iteration in range(num_elements):
#         print(f"\n--- Iteration {iteration + 1}/{num_elements} ---")
        
#         lambdas, optimal_alpha, _, optimal_dist, K = optimise_alpha_lasso(
#             current_data, t, lambdas
#         )
#         print(f"Optimal alpha value: {optimal_alpha:.6e}") 
        
#         # Separate rates from background intercept
#         dist_rates = optimal_dist[:-1]
#         bg = optimal_dist[-1]
        
#         # Find dominant peak in this iteration
#         peaks, _ = find_peaks(dist_rates, prominence=np.max(dist_rates) * 0.05, distance=3)
#         if len(peaks) == 0:
#             print("No significant peaks left in residual data. Stopping.")
#             break
            
#         peak_heights = dist_rates[peaks]
#         dominant_peak_idx = peaks[np.argmax(peak_heights)]
#         optimal_rate = lambdas[dominant_peak_idx]
        
#         sweep_distributions = alpha_sweep_lasso(
#             current_data, lambdas, K, optimal_alpha, sweep_factor=3.0, steps=20
#         )

#         window_mask = (lambdas > optimal_rate / 1.5) & (lambdas < optimal_rate * 1.5)
#         tracked_rates = []
        
#         for dist in sweep_distributions:
#             window_dist = dist[window_mask]
#             window_lambdas = lambdas[window_mask]
            
#             if len(window_dist) > 0:
#                 local_max_idx = np.argmax(window_dist)
#                 tracked_rates.append(window_lambdas[local_max_idx])
                
#         rates = np.array(tracked_rates)
#         rate_min, rate_max = np.min(rates), np.max(rates)
        
#         tau_opt = 1.0 / optimal_rate
#         tau_min, tau_max = 1.0 / rate_max, 1.0 / rate_min 
        
#         all_element_data.append({
#             'rate_opt': optimal_rate,
#             'rate_err_plus': rate_max - optimal_rate,
#             'rate_err_minus': optimal_rate - rate_min,
#             'tau_opt': tau_opt,
#             'tau_err_plus': tau_max - tau_opt,
#             'tau_err_minus': tau_opt - tau_min
#         })
        
#         print(f"Extracted Peak: {optimal_rate:.3f} us^-1 ({tau_opt*1e3:.3f} ns)")

#         # Isolate the dominant peak's probability distribution
#         isolated_dist = np.zeros(len(lambdas))
#         isolated_dist[window_mask] = dist_rates[window_mask]
        
#         # Reconstruct ONLY this peak's time decay (Do NOT add the background intercept here)
#         isolated_reconstruction = K @ isolated_dist
        
#         # Prevent negative residuals which destabilize Lasso
#         current_data = np.maximum(current_data - isolated_reconstruction, 0)
        
#         cumulative_reconstruction += isolated_reconstruction
#         composite_dist += isolated_dist
        
#         if iteration == num_elements - 1:
#             final_bg = bg

#     # Add the final background back into the total cumulative reconstruction for plotting
#     cumulative_reconstruction += final_bg

#     area = np.trapezoid(composite_dist, lambdas)
#     if area > 0:
#         composite_dist_norm = composite_dist / area
#     else:
#         composite_dist_norm = composite_dist
        
#     plot_optimal(t, raw_data_scaled, lambdas, cumulative_reconstruction, composite_dist_norm, all_element_data)
if __name__ == "__main__":

    input_file = 'data/81801-bwd-counts.dat'

    # Include lambda=0 to permit a constant exponential component
    lambdas = np.logspace(0.4, 20, 50)

    do_iterative_lasso_ILT(
        input_file,
        num_elements=6,
        lambdas=lambdas,
        target_mawr=None
    )
    
# if __name__ == "__main__":
#     input_file = 'data/81801-bwd-counts.dat'  
#     # Dense linear grid targeting the light elements
#     lambdas = np.linspace(0.3, 1.2, 200)
#     do_iterative_lasso_ILT(input_file, num_elements=3, lambdas=lambdas)