# Pattern-Based Trading System
![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![License](https://img.shields.io/badge/license-Apache-2)

## Description
The **Pattern-Based Trading System** is a Python-based library for analyzing financial time-series data. It utilizes advanced algorithms like **Soft Dynamic Time Warping (Soft-DTW)** and **Bayesian Optimization** to identify profitable trading patterns. The system compares **Buy-and-Hold** and **Active Trading** strategies to maximize returns, ensuring no data leakage or self-matching.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Features
- Multi-dimensional **Soft-DTW** for pattern matching with both price and volume data.
- Integration of **Bayesian Optimization** for hyperparameter tuning.
- **Precomputed Tables** for efficient computation of Soft-DTW distances and forward projections.
- Comparison of **Buy-and-Hold** vs. **Active Trading** strategies.
- Ensures no future data leakage or trivial self-matching during backtesting.

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/pattern-trading-system.git
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
### Example: Run the Backtesting Logic
```python
from trading_system import run_pattern_analysis_and_compare_no_future

# Run the backtesting logic
active_return = run_pattern_analysis_and_compare_no_future(
    df=spy,                 # DataFrame of SPY historical data
    pattern_length=10,      # Pattern length in days
    forward_projection=5,   # Days to project forward
    gamma=1.0,              # Soft-DTW smoothing factor
    decay=0.95,             # Decay factor for older matches
    trade_threshold=0.5     # Decision threshold
)
print(f"Active Trading Return: {active_return}")
```

### Example: Perform Hyperparameter Tuning
```python
from bayes_opt import BayesianOptimization

# Define the evaluation function for Bayesian Optimization
def evaluate_hyperparams(pattern_length, forward_projection, gamma, decay, trade_threshold):
    return run_pattern_analysis_and_compare_no_future(
        df=spy,
        pattern_length=int(pattern_length),
        forward_projection=int(forward_projection),
        gamma=gamma,
        decay=decay,
        trade_threshold=trade_threshold
    )

# Define hyperparameter bounds
pbounds = {
    'pattern_length': (5, 60),
    'forward_projection': (2, 15),
    'gamma': (0.1, 10.0),
    'decay': (0.8, 1.0),
    'trade_threshold': (0.1, 1.0)
}

# Run Bayesian Optimization
optimizer = BayesianOptimization(
    f=evaluate_hyperparams,
    pbounds=pbounds,
    random_state=42
)
optimizer.maximize(init_points=20, n_iter=100)

print("Best Parameters:", optimizer.max)
```

## Examples
### Comparing Buy-and-Hold vs. Active Trading
The system calculates returns for both strategies over the same period:
- **Buy-and-Hold**: Simulates a static investment from the start to the end of the dataset.
- **Active Trading**: Dynamically trades based on expected values derived from pattern matching.

#### Example Output:
```
Buy-and-Hold Return: $1250.50
Active Trading Return: $1780.30
```

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature or bug fix:
    ```bash
    git checkout -b feature-name
    ```
3. Commit your changes and push the branch:
    ```bash
    git push origin feature-name
    ```
4. Submit a pull request.
