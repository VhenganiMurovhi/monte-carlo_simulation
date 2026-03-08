import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load historical data
df = pd.read_csv("../data/asset_prices.csv")
df.set_index("date", inplace=True)

# Calculate daily returns all the time please
returns = df.pct_change().dropna()

# Portfolio weights
weights = np.array([0.4, 0.35, 0.25])

# Mean returns and covariance matrix
mean_returns = returns.mean().values
cov_matrix = returns.cov().values

# Simulation settings
num_simulations = 10000
time_horizon = 30  # days

# My initial portfolio value
initial_portfolio_value = 100000

# Store simulation results
portfolio_end_values = []

# Cholesky decomposition for correlated random variables
chol_matrix = np.linalg.cholesky(cov_matrix)

for _ in range(num_simulations):
    portfolio_value = initial_portfolio_value

    for _ in range(time_horizon):
        random_normals = np.random.normal(size=len(weights))
        correlated_randoms = chol_matrix @ random_normals
        simulated_daily_returns = mean_returns + correlated_randoms

        portfolio_return = np.dot(weights, simulated_daily_returns)
        portfolio_value *= (1 + portfolio_return)

    portfolio_end_values.append(portfolio_value)

portfolio_end_values = np.array(portfolio_end_values)

# Risk metrics
portfolio_losses = initial_portfolio_value - portfolio_end_values
var_95 = np.percentile(portfolio_losses, 95)
expected_shortfall_95 = portfolio_losses[portfolio_losses >= var_95].mean()

print("Initial Portfolio Value:", initial_portfolio_value)
print("Average Ending Portfolio Value:", round(portfolio_end_values.mean(), 2))
print("95% Value at Risk (VaR):", round(var_95, 2))
print("95% Expected Shortfall (ES):", round(expected_shortfall_95, 2))

# Plot distribution
plt.figure(figsize=(10, 6))
plt.hist(portfolio_end_values, bins=50, edgecolor="black")
plt.axvline(initial_portfolio_value, linestyle="--", label="Initial Value")
plt.axvline(np.percentile(portfolio_end_values, 5), linestyle="--", label="5th Percentile")
plt.title("Monte Carlo Simulation of Portfolio End Values")
plt.xlabel("Portfolio Value")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.show()