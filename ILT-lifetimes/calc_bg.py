import numpy as np
from tikhonov_ILT_Alex import load_data

t, raw_data,  errors = load_data('data/mantid/81801-bwd-counts.dat')

def estimate_background_fraction(t, raw_data, tail_start_us=15.0):
    # Isolate the late-time data where only random noise remains
    tail_mask = t > tail_start_us
    tail_counts = raw_data[tail_mask]

    # Calculate the average background counts per bin
    absolute_bg = np.mean(tail_counts)

    # Normalize it against the peak amplitude (t ~ 0)
    peak_counts = np.max(raw_data)
    print(peak_counts)
    bg_fraction = absolute_bg / peak_counts

    return bg_fraction

estimated_bg_fraction = estimate_background_fraction(t, raw_data)
print(f"Estimated background fraction: {estimated_bg_fraction:.4f}")