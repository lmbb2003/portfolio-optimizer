# Portfolio Optimizer – Mean-Variance & Sharpe Ratio

This project simulates and optimizes a portfolio of stocks using daily returns, covariance matrices, and the Sharpe ratio. 
It relies on the classical mean-variance framework (Markowitz) with realistic constraints (no short-selling, weight sum = 1).

## Features
- Historical data loading via Yahoo Finance (`yfinance`)
- Monte Carlo simulation of 10,000+ random portfolios
- Portfolio optimization using `scipy.optimize`
- Visualization of the efficient frontier and max Sharpe point

## Tech Stack
- Python 3.10
- NumPy, Pandas, Matplotlib
- yfinance, SciPy

## Usage
Open the Jupyter notebook and run each cell.  
The code is modular and can be adapted to different assets or constraints.

## Author
Louis-Maxence Blanc-Bernard  
Master in Finance, IÉSEG  

This project was developed as part of my technical training in quantitative finance, with the goal of applying Python and optimization techniques to real financial data.

I'm currently deepening my skills in data-driven trading and portfolio construction.
