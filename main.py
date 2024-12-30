import os
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import stumpy
from tslearn.metrics import soft_dtw
from joblib import Parallel, delayed
from numba import njit
from bayes_opt import BayesianOptimization

# 1) Tell Modin to use Ray (if not already set in the environment):
os.environ["MODIN_ENGINE"] = "ray"
import modin.pandas as pd

plt.style.use('fivethirtyeight')

# ----------------------------
# Data Download & Preprocessing
# ----------------------------
print("Downloading SPY data (with volume)...")
spy = yf.download('SPY', start='2010-01-01', progress=False)

# Flatten MultiIndex columns (if any)
if isinstance(spy.columns, pd.MultiIndex):
    spy.columns = [f"{level0}_{level1}".strip('_') for level0, level1 in spy.columns]

# Drop rows with missing data
spy = spy.dropna(subset=["Open_SPY", "High_SPY", "Low_SPY", "Close_SPY", "Volume_SPY"])

# Calculate features
spy["range"] = spy["High_SPY"] - spy["Low_SPY"]
spy["direction"] = np.sign(spy["Close_SPY"] - spy["Open_SPY"])
spy["directional_range_pct"] = (spy["range"] / spy["Open_SPY"]) * 100 * spy["direction"]
spy["volume_pct"] = spy["Volume_SPY"].pct_change() * 100

# Optional: Basic filtering
spy = spy[spy["Open_SPY"] > 0].dropna()

print("Initial data head:")
print(spy.head())

# ----------------------------
# Helper Functions
# ----------------------------
@njit
def apply_decay_jit(distance, idx, segment_length, total_length, decay=0.9):
    """Exponentially inflate (or deflate) older matches; JIT-compiled."""
    steps_from_end = total_length - (idx + segment_length)
    return distance * (decay ** steps_from_end)

def zscore_2d(data_2d):
    """Z-score each dimension independently."""
    mean_vals = np.mean(data_2d, axis=0)
    std_vals = np.std(data_2d, axis=0) + 1e-9
    return (data_2d - mean_vals) / std_vals

def softdtw_distance_multidim(p1, p2, gamma=1.0):
    """Soft-DTW for multi-dimensional data, with z-scoring."""
    p1z = zscore_2d(p1)
    p2z = zscore_2d(p2)
    return soft_dtw(p1z, p2z, gamma=gamma)

# ----------------------------
# Main Backtesting Logic
# ----------------------------
def run_pattern_analysis_and_compare_no_future(
    df,
    pattern_length=5,
    forward_projection=3,
    gamma=1.0,
    decay=0.9,
    trade_threshold=0.5,
    timeframe='1D',
    top_k=10,
    rolling_window=3,
    outcome_metric=0,
    transaction_cost=0.0,
    initial_capital=1000
):
    """
    Compares Buy and Hold vs. Active Trading with pattern matching (Soft-DTW),
    ensuring no future data leakage and no self-matching.
    """

    # 1) Resample to desired timeframe
    resampled_df = df.resample(timeframe).mean().dropna()

    # 2) Rolling smoothing
    resampled_df["dr_pct_smooth"] = (
        resampled_df["directional_range_pct"].rolling(int(rolling_window)).mean()
    )
    resampled_df["vol_pct_smooth"] = (
        resampled_df["volume_pct"].rolling(int(rolling_window)).mean()
    )
    resampled_df.dropna(subset=["dr_pct_smooth", "vol_pct_smooth"], inplace=True)

    # 3) Convert to NumPy arrays
    features_2d = resampled_df[["dr_pct_smooth", "vol_pct_smooth"]].values
    dr_array = resampled_df["directional_range_pct"].values
    open_array = resampled_df["Open_SPY"].values
    close_array = resampled_df["Close_SPY"].values

    N = len(features_2d)
    if N < pattern_length + forward_projection:
        raise ValueError("Not enough data for the requested pattern length and forward projection.")

    # 4) Precompute Full Matrix Profile
    full_matrix_profile = stumpy.stump(features_2d.flatten(), m=pattern_length)

    # 5) Precompute Z-Scores for All Segments
    zscored_segments = np.array([zscore_2d(features_2d[i : i + pattern_length]) for i in range(N - pattern_length)])

    # 6) Precompute Valid Indices for Trivial Match Exclusion
    valid_indices_per_iteration = [
        np.setdiff1d(np.arange(i + pattern_length), np.arange(i, i + pattern_length)) 
        for i in range(N - pattern_length)
    ]

    # 7) Parallel Backtest Routine
    def process_single_iteration(i):
        current_pattern = zscored_segments[i]

        # Use precomputed distances, excluding trivial matches
        valid_indices = valid_indices_per_iteration[i]
        prefiltered_distances = full_matrix_profile[valid_indices, 0]

        # Apply decay to distances
        decayed_distances = np.array([
            apply_decay_jit(d, idx, pattern_length, len(valid_indices), decay)
            for idx, d in enumerate(prefiltered_distances)
        ])

        # Take top K candidates
        top_candidates = np.argsort(decayed_distances)[:top_k]

        # Refine with Soft-DTW
        refined_candidates = [
            (cand_idx, softdtw_distance_multidim(zscored_segments[cand_idx], current_pattern, gamma=gamma))
            for cand_idx in top_candidates
        ]

        # Sort by distance, keep top half
        refined_candidates.sort(key=lambda x: x[1])
        refined_candidates = refined_candidates[: max(1, len(refined_candidates) // 2)]

        # Calculate outcomes
        outcomes = []
        for cand_idx, _ in refined_candidates:
            future_start = cand_idx + pattern_length
            future_end = future_start + forward_projection
            if future_end <= len(dr_array):
                segment = dr_array[future_start:future_end]
                if len(segment) > 1:
                    if outcome_metric == 0:
                        outcomes.append(segment[-1] - segment[0])
                    elif outcome_metric == 1:
                        outcomes.append(np.sum(segment))
                    else:
                        outcomes.append(np.mean(segment))

        overall_ev = np.mean(outcomes) if outcomes else 0.0
        decision = 1 if overall_ev > trade_threshold else -1 if overall_ev < -trade_threshold else 0
        return i, decision, overall_ev

    results = Parallel(n_jobs=-1, verbose=0)(
        delayed(process_single_iteration)(i)
        for i in range(N - pattern_length - forward_projection)
    )

    # Final Capital Calculation
    capital_active = initial_capital
    shares_active = 0
    for i, decision, _ in results:
        current_open = open_array[i + pattern_length]
        if decision == 1 and shares_active == 0:
            shares_to_buy = capital_active // (current_open * (1 + transaction_cost))
            if shares_to_buy > 0:
                shares_active += shares_to_buy
                capital_active -= shares_to_buy * current_open * (1 + transaction_cost)
        elif decision == -1 and shares_active > 0:
            capital_active += shares_active * current_open * (1 - transaction_cost)
            shares_active = 0

    if shares_active > 0:
        capital_active += shares_active * close_array[-1] * (1 - transaction_cost)

    active_trading_return = capital_active - initial_capital

    print(f"Buy and Hold Return: {((close_array[-1] / open_array[0]) - 1) * initial_capital:.2f}")
    print(f"Active Trading Return: {active_trading_return:.2f}")
    return active_trading_return

# ----------------------------
# Bayesian Optimization Logic
# ----------------------------
def evaluate_hyperparams(pattern_length, forward_projection, gamma, decay, trade_threshold):
    return run_pattern_analysis_and_compare_no_future(
        df=spy,
        pattern_length=int(pattern_length),
        forward_projection=int(forward_projection),
        gamma=gamma,
        decay=decay,
        trade_threshold=trade_threshold
    )

# Define the search space for Bayesian Optimization
pbounds = {
    'pattern_length': (5, 60),          # Range for pattern length
    'forward_projection': (2, 15),     # Range for forward projection
    'gamma': (0.1, 10.0),              # Soft-DTW smoothing
    'decay': (0.8, 1.0),               # Decay factor
    'trade_threshold': (0.1, 1.0)      # Decision threshold
}

# Run Bayesian Optimization
optimizer = BayesianOptimization(
    f=evaluate_hyperparams,
    pbounds=pbounds,
    random_state=42
)

optimizer.maximize(init_points=16, n_iter=1000)

# Print the best parameters
print("Best Parameters:", optimizer.max)
