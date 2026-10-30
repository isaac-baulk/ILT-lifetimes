import chemparse
import matplotlib.pyplot as plt
import pandas as pd
import re
import numpy as np
from scipy.optimize import lsq_linear, least_squares # Alex: add this import
from scipy.signal import find_peaks

from tikhonov_ILT_Alex import load_data
from L1_ILT_Alex import optimise_alpha_lasso, plot_optimal, exponential_jacobian, exponential_model
from utils.isotope_wrapper import parse_formula_isotopes, ABUNDANCE, lookup_name
from iterative_ILT import do_iterative_lasso_ILT

# ======================USER CONFIG==========================
PATH = "data/muon_capture_tables_III_IV.xlsx"
FORMULA = "CaCO3"  # Example formula; replace with the desired chemical formula
MU_RATE = 0.45516
BG_FRAC = 0.0001
DB = {"H", "O", "O-18", "Li-6", "Li-7", "F", "C", "C-13"}

# ===========================================================


def read_and_clean_data(): 
    """Read the Excel file, clean the data, and compute averages for elements with multiple entries."""
    # --- load, lower-case the column names ---
    df = pd.read_excel(PATH)
    df.columns = df.columns.str.lower()
    
    # 1. drop rows with no mean lifetime
    df = df.dropna(subset=["mean_lifetime_ns"]).reset_index(drop=True)

    #patterndd = re.compile(r"-\d\d", re.IGNORECASE)
    #patternd = re.compile(r"-\d", re.IGNORECASE)
    #patternnat = re.compile(r"-nat", re.IGNORECASE)
    #patterns = [patterndd, patternd, patternnat]

    #pattern = re.compile(r"[A-Za-z]{1,2}-(?:\d{1,2}|nat)")

    #drop specific isotopes
    df['element'] = df['element'].str.replace(r'-(?:\d{1,2}|nat)', '', regex=True)

    print(df.head())

    def calc_avg(g):
        t = g["mean_lifetime_ns"].to_numpy()
        err = g["mean_lifetime_err_ns"].to_numpy()

        #weight mean
        w = 1.0 / err**2
        mean = np.sum(w * t) / np.sum(w)
        mean_err = np.sqrt(1.0 / np.sum(w))
        return pd.Series(
            {
                "average_mean_lifetime": mean,
                "average_mean_lifetime_error": mean_err,
                "n_entries": len(t),
            }
        )
 
    avg_df = (
        df.groupby("element", sort=False)
        .apply(calc_avg, include_groups=False)
        .reset_index()
    )
    avg_df["n_entries"] = avg_df["n_entries"].astype(int)
    
    return df, avg_df


def get_element_amplitudes(elements_df, avg_df, df):
    """Calculate the amplitude for each element based on its count and atomic number."""
    # 1. Validate that all elements exist in avg_df
    missing_elements = set(elements_df['element']) - set(avg_df['element'])
    if missing_elements:
        raise ValueError(f"Elements not found in the data: {missing_elements}")

    # 2. Vectorize the Z-law calculation (no loops required)
    # Create a mapping dictionary of {element: Z} from df
    z_map = df.drop_duplicates(subset=["element"]).set_index("element")["z"]
    
    # Map Z values to the elements and calculate weights (count * Z)
    weights = elements_df["count"] * elements_df["element"].map(z_map)
    
    # 3. Calculate the final fractional amplitude
    elements_df = elements_df.copy() # Prevents SettingWithCopyWarning
    elements_df["amplitude"] = weights / weights.sum()
    
    return elements_df


def get_element_rates_old(elements_df, avg_df, df):
    avg_lifetimes = (avg_df.set_index('element').loc[elements_df['element'], 'average_mean_lifetime'].values) / 10**3
    avg_lifetime_errors = (avg_df.set_index('element').loc[elements_df['element'], 'average_mean_lifetime_error'].values) / 10**3

    elements_df['rate'] = 1.0 / avg_lifetimes  # Convert ns to μs for rate calculation
    elements_df['rate_error'] = avg_lifetime_errors / (avg_lifetimes**2)  # Propagate error for rate calculation

    return elements_df

def get_element_rates(elements_df, avg_df, df=None):
    """Rate (1/us) and rate error for each element, from its average lifetime.
 
    Resolution order for each name in elements_df['element']:
      1. exact entry in avg_df (measured element or isotope, e.g. 'Ni', 'O-18')
      2. an isotope missing from avg_df falls back to the plain element ('O-17' -> 'O')
      3. a plain element with no entry of its own (e.g. 'Li') is built from its
         isotopes, weighting the RATES by natural abundance (renormalised over
         the isotopes present in avg_df) and adding the errors in quadrature.
    `df` is unused; kept so the signature is unchanged.
    """
    avg = avg_df.set_index("element")
    tau = avg["average_mean_lifetime"] / 1e3              # ns -> us
    tau_err = avg["average_mean_lifetime_error"] / 1e3
    rate = 1.0 / tau                                      # 1/us
    rate_err = tau_err / tau**2                           # propagated error
    available = set(avg.index)
 
    def resolve(name):
        key = lookup_name(name, available)                # steps 1 and 2
        if key is not None:
            return rate[key], rate_err[key]
        iso = [f"{name}-{a}" for a in ABUNDANCE.get(name, {}) if f"{name}-{a}" in available]
        if not iso:
            raise KeyError(f"'{name}' has no entry (element or isotope) in avg_df")
        w = np.array([ABUNDANCE[name][int(k.split("-")[1])] for k in iso])
        w /= w.sum()                                      # step 3
        r = rate[iso].to_numpy()
        e = rate_err[iso].to_numpy()
        return float(w @ r), float(np.sqrt(np.sum((w * e) ** 2)))
 
    out = elements_df.copy()
    resolved = [resolve(n) for n in out["element"]]
    out["rate"] = [r for r, _ in resolved]
    out["rate_error"] = [e for _, e in resolved]
    return out

def generate_synethtic_data(elements_df, avg_df):
    """Generate synthetic data based on the elements' amplitudes and rates."""
    # 1. Calculate the total amplitude and normalize
    total_amplitude = elements_df['amplitude'].sum()
    if total_amplitude <= 0:
        raise ValueError("Total amplitude must be positive to generate synthetic data.")
    
    normalized_amplitudes = elements_df['amplitude'] / total_amplitude

    # 2. Generate synthetic data for each element
    synthetic_data = []
    for _, row in elements_df.iterrows():
        element = row['element']
        amplitude = row['amplitude']
        rate = row['rate']
        
        # Generate synthetic data points (e.g., using an exponential decay model)
        time_points = np.linspace(0, 15, 2000)  # Example time points from 0 to 15 μs
        decay_curve = amplitude * np.exp(-rate * time_points)
        
        synthetic_data.append({
            'element': element,
            'decay_curve': decay_curve
        })

    pure_signal_counts = BG_FRAC + np.sum([d['decay_curve'] for d in synthetic_data], axis=0)
    return pure_signal_counts, synthetic_data, time_points

def plot_synthetic_data(synthetic_data, pure_signal_counts, time_points):
    """Plot the synthetic decay curves and the pure signal counts."""
    plt.figure(figsize=(10, 6))
    for data in synthetic_data:
        plt.plot(time_points, data['decay_curve'], label=f"{data['element']} Decay Curve")
    plt.plot(time_points, pure_signal_counts, label="Pure Signal Counts", color='black', linewidth=2)
    plt.title("Synthetic Decay Curves and Pure Signal Counts")
    plt.xlabel("Time (μs)")
    plt.ylabel("Counts")
    plt.legend()
    plt.grid()
    plt.show()

def generate_dynamic_grids(elements_df):
    """
    Creates time and lambda grids scaled exactly to the physical rates of the compound.
    Slashes array sizes to massively speed up the ILT lasso solver.
    """
    rates = elements_df['rate'].values
    min_rate = np.min(rates)
    max_rate = np.max(rates)
    
    # --- 1. Dynamic Time Grid ---
    # Must be fast enough to resolve the shortest lifetime (min 15 points per decay)
    tau_min = 1.0 / max_rate
    dt = tau_min / 15.0
    
    # Must be long enough to see the longest lifetime die out (approx 5 lifetimes)
    # Capped at 15-20us since muons completely decay away by then anyway
    tau_max = 1.0 / min_rate
    t_max = min(5.0 * tau_max, 16.0) 
    
    n_points = int(t_max / dt)
    # Keep array size within sensible bounds for matrix math
    n_points = np.clip(n_points, 100, 1500)
    time_points = np.linspace(0, t_max, n_points)
    
    # --- 2. Dynamic Lambda Grid ---
    # Bound the search space tightly around the known theoretical extremes (+/- 50%)
    l_min = max(0.01, min_rate * 0.5)
    l_max = max_rate * 1.5
    
    # Because the bounds are now tight, 150 points provides massive resolution 
    # while keeping the Lasso inversion lightning fast.
    lambdas = np.linspace(l_min, l_max, 150)
    
    print(f"Dynamic Grids -> Time: {n_points} points | Lambdas: {l_min:.2f} to {l_max:.2f}")
    
    return time_points, lambdas

def scale_and_noise(pure_signal_counts, time_points, target_mev):
    """Scale the pure signal counts to a target MeV and add Gaussian noise."""
    target_mev *= 1e6
    dx = time_points[1] - time_points[0]
    integral = np.trapezoid(pure_signal_counts, dx=dx)
    sf = target_mev / integral
    scaled_counts = pure_signal_counts * sf

    # 2. Add Gaussian noise based on the square root of the counts (Poisson noise approximation)
    noise = np.random.normal(0, np.sqrt(scaled_counts), size=scaled_counts.shape)
    noisy_counts = scaled_counts + noise
    noisy_counts_errors = np.sqrt(scaled_counts)  # Error is the square root of the counts

    return noisy_counts, noisy_counts_errors

def do_iterative_lasso_ILT(time_points, raw_data, errors, mev, lambdas, target_mawr=1.0, print_int=False):
    # use a forward modelling approach to iteratively add exponentials to the fit, using L1-regularized ILT to identify the next exponential to add

    # Normalize data
    scale_factor = np.max(raw_data)
    if scale_factor <= 0:
        raise ValueError("Data have no positive counts to fit")
    raw_data_scaled = raw_data / scale_factor

    # 1. Create a mask identifying where the loaded errors are zero
    zero_error_mask = (errors == 0)
    poisson_fallback = np.sqrt(np.maximum(raw_data, 1.0))
    safe_raw_errors = np.where(zero_error_mask, poisson_fallback, errors)
    errors = safe_raw_errors / scale_factor


    parameters = np.array([np.min(raw_data_scaled)], dtype=float)
    current_data = raw_data_scaled.copy()
    metric_history = [
        np.mean(np.abs(current_data) / errors)
    ]

    found_components = {}
    iteration = 0
    print(f'NEW RUN: MAWR = {metric_history[-1]:.6f}')
    while metric_history[-1] > target_mawr:

        print(f"\n--- Iteration {iteration + 1} ---")

        # STEP 1: Perform ILT on the current residual
        _, optimal_alpha, _, ilt_dist, _ = optimise_alpha_lasso(current_data, time_points, lambdas)

        if np.max(ilt_dist) <= 1e-12:
            print("No positive ILT peak left. Stopping.")
            break

        # STEP 2: Identify the tallest peak in the ILT distribution
        peak_idx = int(np.argmax(ilt_dist))
        new_rate = float(lambdas[peak_idx])
        new_amplitude = max(float(ilt_dist[peak_idx]), 1e-9)

        # STEP 3: Add a new exponential
        initial_guess = np.r_[parameters, new_amplitude, new_rate]
        num_params = len(initial_guess)
        # Lower bounds: 0.0 for background, all amplitudes, and all rates
        lower_bounds = np.zeros(num_params)
        
        # Upper bounds: start with infinity for background and amplitudes
        upper_bounds = np.full(num_params, np.inf)
        # Apply the max lambda limit ONLY to the rate parameters (indices 2, 4, 6...)
        upper_bounds[2::2] = float(np.max(lambdas))

        # STEP 4: Globally refit ALL exponentials
        result = least_squares(
            # The objective function
            lambda p: (exponential_model(time_points, p) - raw_data_scaled) / errors,
            
            # The analytical Jacobian (divided by errors to match the objective)
            jac=lambda p: exponential_jacobian(time_points, p) / errors[:, None],
            
            x0=initial_guess,
            bounds=(lower_bounds, upper_bounds),
            
            # Relax tolerances to stop wasting time on statistical noise
            ftol=1e-5,
            xtol=1e-5,
            
            max_nfev=10000
        )

        # Updated parameters for ALL exponentials
        parameters = result.x
        fit = exponential_model(time_points, parameters)
        current_data = raw_data_scaled - fit
        mawr = np.mean(np.abs(current_data) / errors)
        metric_history.append(mawr)
        print(f"Weighted residual: {mawr:.6f}")

        # Calculate Covariance Matrix from the Jacobian
        # Use pseudo-inverse (pinv) to prevent crashes if the matrix is nearly singular
        cov = np.linalg.pinv(result.jac.T @ result.jac)
        param_errors = np.sqrt(np.diag(cov))

        # integrate under exponential for true amplitude
        #print(f"Iteration {iteration + 1} results:")
        found_components = {}
        for k in range(iteration + 1):
            A = parameters[2*k + 1]
            rate = parameters[2*k + 2]

            #integrated amplitude
            #A = np.trapezoid(A * np.exp(-rate * time_points))
            
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
            

            found_components[k + 1] = {
                "mev": mev,
                "amplitude": A,
                "amplitude err": A_err,
                "rate": rate,
                "rate err": rate_err,
                "lifetime": tau,
            }

        iteration += 1


    # -----------------------------------------------
    # STEP 6: Display the results
    # -----------------------------------------------
    if print_int:
        fig, (ax1, ax2, ax3) = plt.subplots(
            3, 1, figsize=(10, 9)
        )

        # Experimental data and global fit
        ax1.plot(
            time_points, raw_data_scaled,
            '.', markersize=3,
            label='Data'
        )

        ax1.plot(
            time_points, fit,
            '-', linewidth=2,
            label='Global fit'
        )

        # Show individual exponentials
        for k in range(iteration):

            A = parameters[2*k]
            rate = parameters[2*k + 1]

            component = A * np.exp(-rate * time_points)

            ax1.plot(
                time_points, component,
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
        ax2.plot(time_points, current_data, '.', markersize=3)

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
        plt.title(f'ILT at {mev} MEv')
        plt.show()

    return found_components, np.asarray(metric_history)




def determine_required_mev(pure_signal_counts, time_points, lambdas, elements_df, target_mawr=1.05):
    """
    Sweeps MEv up to a strict limit of 50.0. 
    Maps found rates to theoretical elements to determine if they are resolved or grouped.
    """
    mev_steps = [1.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0]
    
    print("\n--- Determining Required MEv for Resolution ---")
    
    for mev in mev_steps:
        print(f"\nTesting {mev:.1f} MEv... ==================================")
        noisy_counts, errors = scale_and_noise(pure_signal_counts, time_points, mev)
        
        # Run the iterative ILT solver
        found_components, _ = do_iterative_lasso_ILT(
            time_points, noisy_counts, errors, mev,
            lambdas=np.linspace(0.1, 15.0, 500), 
            target_mawr=target_mawr,
        )
        
        found_rates = [data['rate'] for data in found_components.values()]
        
        if not found_rates:
            print("Solver failed to find components.")
            continue
            
        # Map each theoretical element to the closest rate found by the solver
        groupings = {fr: [] for fr in found_rates}
        for _, row in elements_df.iterrows():
            closest_rate = min(found_rates, key=lambda x: abs(x - row['rate']))
            groupings[closest_rate].append(row['element'])
        
        # Check if any found component is forced to represent multiple theoretical elements
        has_grouped_factor = any(len(elements) > 1 for elements in groupings.values())
        
        if not has_grouped_factor:
            print(f"\nSUCCESS: All elements cleanly separated at {mev:.1f} MEv.")
            for fr, elements in groupings.items():
                if elements:
                    print(f"  {elements[0]} isolated at rate = {fr:.4f} us^-1")
            return mev, found_components

    # If the loop finishes without returning, the 50 MEv limit was reached
    print("\nLIMIT REACHED: Solver cannot fully separate components at 50.0 MEv.")
    print("The following components are definitively Grouped Factors:")
    for fr, elements in groupings.items():
        if len(elements) > 1:
            print(f"  -> GROUPED FACTOR: {', '.join(elements)} (Effective Rate: {fr:.4f} us^-1)")
        elif len(elements) == 1:
            print(f"  -> Separated: {elements[0]} (Rate: {fr:.4f} us^-1)")
            
    return 50.0, found_components

def check_resolution_limits(elements_df, resolution_ratio=1.15):
    """
    Checks if theoretical decay rates are too close to be resolved by the solver
    and predicts the amplitude-weighted grouped rates.
    """
    print("\n--- Theoretical Resolution Check ---")
    
    # Sort elements by rate to compare neighbors
    df = elements_df.sort_values(by='rate').reset_index(drop=True)
    
    groups = []
    current_group = [df.iloc[0]]
    
    for i in range(1, len(df)):
        row = df.iloc[i]
        # Compare current element's rate to the previous element's rate
        if row['rate'] / current_group[-1]['rate'] < resolution_ratio:
            current_group.append(row)
        else:
            groups.append(current_group)
            current_group = [row]
    groups.append(current_group)
    
    # Calculate and print predictions for each group
    for group in groups:
        elements = [r['element'] for r in group]
        total_amp = sum(r['amplitude'] for r in group)
        
        # The solver will find the amplitude-weighted average rate
        eff_rate = sum(r['amplitude'] * r['rate'] for r in group) / total_amp
        
        if len(group) > 1:
            print(f"WARNING: Unresolvable components -> {', '.join(elements)}")
            print(f"  These rates are too similar and will merge.")
            print(f"  Predicted solver result: Amplitude = {total_amp:.4f} | Rate = {eff_rate:.4f} us^-1\n")
        else:
            print(f"Clean separation expected -> {elements[0]}")
            print(f"  Predicted solver result: Amplitude = {total_amp:.4f} | Rate = {eff_rate:.4f} us^-1\n")
            
    print("------------------------------------\n")

def main():
    #parsed_formula = parse_formula_isotopes(FORMULA, available=DB)
    parsed_formula = chemparse.parse_formula(FORMULA)
    elements_df = pd.DataFrame(list(parsed_formula.items()), columns=['element', 'count'])
    df, avg_df = read_and_clean_data()
    elements_df = get_element_amplitudes(elements_df, avg_df, df)
    elements_df = get_element_rates(elements_df, avg_df, df)

    time_points, lambdas = generate_dynamic_grids(elements_df)
    pure_signal_counts, synthetic_data, time_points = generate_synethtic_data(elements_df, avg_df)

    plot_synthetic_data(synthetic_data, pure_signal_counts, time_points)

    # run sim
    required_mev, final_components = determine_required_mev(
        pure_signal_counts, 
        time_points,
        lambdas, 
        elements_df, 
        target_mawr=0.8  # Lower threshold to force the solver to try splitting elements
    )

main()