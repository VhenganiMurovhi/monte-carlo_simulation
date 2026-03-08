import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("Monte Carlo Portfolio Risk Dashboard")

# load data
from pathlib import Path
data_path = Path(__file__).resolve().parent.parent / "data" / "asset_prices.csv"
df = pd.read_csv(data_path)

df.set_index("date", inplace=True)

returns = df.pct_change().dropna()

weights = np.array([0.4, 0.35, 0.25])

mean_returns = returns.mean().values
cov_matrix = returns.cov().values

num_simulations = 10000
time_horizon = 30
initial_portfolio_value = 100000

chol = np.linalg.cholesky(cov_matrix)

results = []

for i in range(num_simulations):

    portfolio_value = initial_portfolio_value

    for t in range(time_horizon):

        rand = np.random.normal(size=len(weights))
        correlated = chol @ rand
        sim_returns = mean_returns + correlated

        portfolio_return = np.dot(weights, sim_returns)
        portfolio_value *= (1 + portfolio_return)

    results.append(portfolio_value)

results = np.array(results)

losses = initial_portfolio_value - results

var95 = np.percentile(losses, 95)
es95 = losses[losses >= var95].mean()

prob_loss = np.mean(results < initial_portfolio_value) * 100

st.subheader("Risk Metrics")

col1, col2, col3 = st.columns(3)

col1.metric("Value at Risk (95%)", f"${var95:,.2f}")
col2.metric("Expected Shortfall", f"${es95:,.2f}")
col3.metric("Probability of Loss", f"{prob_loss:.2f}%")

st.subheader("Simulation Distribution")

fig, ax = plt.subplots()

ax.hist(results, bins=50)

ax.axvline(initial_portfolio_value, linestyle="--", label="Initial Value")

ax.set_title("Monte Carlo Portfolio Distribution")

ax.set_xlabel("Portfolio Value")
ax.set_ylabel("Frequency")

ax.legend()

st.pyplot(fig)