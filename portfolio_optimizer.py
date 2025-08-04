
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def load_data(tickers, start="2022-08-01", end="2025-08-01"):
    data = yf.download(tickers, start=start, end=end)['Adj Close'].dropna()
    returns = data.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    return data, returns, mean_returns, cov_matrix

def simulate_portfolios(mean_returns, cov_matrix, n_portfolios=10000):
    n_assets = len(mean_returns)
    results = np.zeros((3, n_portfolios))
    for i in range(n_portfolios):
        weights = np.random.dirichlet(np.ones(n_assets))
        ret = np.dot(weights, mean_returns) * 252
        vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
        sharpe = ret / vol
        results[0, i], results[1, i], results[2, i] = vol, ret, sharpe
    return results

def optimize_sharpe(mean_returns, cov_matrix):
    n_assets = len(mean_returns)
    def portfolio_perf(weights):
        ret = np.dot(weights, mean_returns) * 252
        vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
        return ret, vol

    def neg_sharpe(weights):
        ret, vol = portfolio_perf(weights)
        return -ret / vol

    bounds = tuple((0, 1) for _ in range(n_assets))
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    init_guess = np.ones(n_assets) / n_assets

    opt = minimize(neg_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    weights = opt.x
    ret, vol = portfolio_perf(weights)
    sharpe = ret / vol
    return weights, ret, vol, sharpe

def plot_results(results, opt_vol, opt_return):
    plt.figure(figsize=(10, 6))
    plt.scatter(results[0], results[1], c=results[2], cmap='viridis', alpha=0.5)
    plt.colorbar(label='Sharpe Ratio')
    plt.scatter(opt_vol, opt_return, c='red', marker='*', s=200, label='Max Sharpe')
    plt.xlabel('Volatility')
    plt.ylabel('Return')
    plt.title('Efficient Frontier')
    plt.legend()
    plt.show()

def run_optimizer(tickers):
    data, returns, mean_returns, cov_matrix = load_data(tickers)
    results = simulate_portfolios(mean_returns, cov_matrix)
    opt_weights, opt_return, opt_vol, opt_sharpe = optimize_sharpe(mean_returns, cov_matrix)
    plot_results(results, opt_vol, opt_return)

    weights_series = pd.Series(opt_weights, index=tickers).sort_values(ascending=False)
    perf_series = pd.Series({
        "Expected Return": opt_return,
        "Volatility": opt_vol,
        "Sharpe Ratio": opt_sharpe
    })

    return weights_series, perf_series

# Example usage
if __name__ == "__main__":
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
    weights, perf = run_optimizer(tickers)
    print("Optimal Portfolio Weights:\n", weights)
    print("\nPortfolio Performance:\n", perf)
