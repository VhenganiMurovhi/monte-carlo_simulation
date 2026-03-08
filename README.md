# Advanced Monte Carlo Portfolio Stress Testing Engine

This is a Python-based Monte Carlo simulation engine designed to model portfolio risk under uncertainty.
The system simulates thousands of possible market scenarios and evaluates portfolio performance under normal and stressed conditions.

This project demonstrates how probabilistic modeling and simulation techniques can be used to estimate financial risk metrics such as Value at Risk (VaR) and Expected Shortfall (ES).

---

## Overview

Financial markets are uncertain. Institutions use Monte Carlo simulations to estimate potential future portfolio outcomes and understand downside risk.

This project builds a simplified version of such a system using Python.

The engine:

• simulates correlated asset returns  
• generates thousands of future portfolio scenarios  
• calculates downside risk metrics  
• visualizes the distribution of potential portfolio outcomes  

---

## Features

- Multi-asset portfolio simulation
- Correlated return modeling using covariance matrix
- Monte Carlo simulation with thousands of scenarios
- Value at Risk (VaR) calculation
- Expected Shortfall (ES) calculation
- Portfolio distribution visualization
- Stress-testing framework for adverse market conditions

---

## Technologies Used

Python
NumPy
Pandas
Matplotlib

---

## PURPOSE

This project aims to demonstrate how probabilistic simulation techniques are used in financial risk management and portfolio stress testing.

The main goal is to explore how data science, statistics and programming intersect in financial systems and decison-making.

## FUTURE IMPROVEMENTS

1. stress regime modelling
2. fat-tailed return distributions
3. drawdown tracking
4. scenario comparison dashboards.