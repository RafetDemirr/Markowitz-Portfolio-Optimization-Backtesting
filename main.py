import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# 1. VERİ SETİ HAZIRLAMA
assets = ['THYAO.IS', 'EREGL.IS', 'ASELS.IS', 'TUPRS.IS', 'BIMAS.IS']
data = yf.download(assets, start='2020-01-01', end='2025-01-01', auto_adjust=True)['Close']
returns = data.pct_change().dropna()

# Yıllık bazda istatistikler
annual_returns = returns.mean() * 252
annual_cov = returns.cov() * 252

# 2. PORTFÖY İSTATİSTİKLERİ FONKSİYONU
def portfolio_stats(weights):
    weights = np.array(weights)
    p_ret = np.sum(annual_returns * weights)
    p_vol = np.sqrt(np.dot(weights.T, np.dot(annual_cov, weights)))
    sharpe = (p_ret - 0.02) / p_vol # 0.02 = Risksiz getiri oranı (varsayılan)
    return np.array([p_ret, p_vol, sharpe])

# 3. OPTİMİZASYON (Max Sharpe Ratio)
def min_func_sharpe(weights):
    return -portfolio_stats(weights)[2]

cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}) # Ağırlık toplamı 1 olmalı
bounds = tuple((0, 1) for _ in range(len(assets)))     # Açığa satış yok (0-1 arası)
init_guess = len(assets) * [1. / len(assets)]          # Eşit dağılımla başla

opt_results = minimize(min_func_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=cons)
optimal_weights = opt_results.x

# 4. BACKTESTING (Optimal Portföy vs BIST 100)
test_data = yf.download(assets, start='2025-01-01', auto_adjust=True)['Close']
benchmark_data = yf.download('XU100.IS', start='2025-01-01', auto_adjust=True)['Close']

test_returns = test_data.pct_change().dropna()
benchmark_returns = benchmark_data.pct_change().dropna()

# Portföy ve Endeks getirilerini hesapla
portfolio_daily_returns = (test_returns * optimal_weights).sum(axis=1)
portfolio_cum_returns = (1 + portfolio_daily_returns).cumprod()
benchmark_cum_returns = (1 + benchmark_returns).cumprod()

# 5. Grafik
fig, ax = plt.subplots(1, 2, figsize=(18, 7))

# Sol Grafik: Efficient Frontier (Etkin Sınır)
num_portfolios = 2000
p_ret, p_vol, p_shr = [], [], []
for _ in range(num_portfolios):
    w = np.random.random(len(assets))
    w /= np.sum(w)
    stats = portfolio_stats(w)
    p_ret.append(stats[0])
    p_vol.append(stats[1])
    p_shr.append(stats[2])

sc = ax[0].scatter(p_vol, p_ret, c=p_shr, cmap='viridis', marker='o', s=10, alpha=0.3)
opt_s = portfolio_stats(optimal_weights)
ax[0].scatter(opt_s[1], opt_s[0], color='red', marker='*', s=200, label='Optimal Portfolio')
ax[0].set_title('Efficient Frontier Analysis')
ax[0].set_xlabel('Volatility (Risk)')
ax[0].set_ylabel('Expected Return')
ax[0].legend()

# Backtesting karşılaştırma
ax[1].plot(portfolio_cum_returns, label='Portfolio Chosen', linewidth=2, color='blue')
ax[1].plot(benchmark_cum_returns, label='BIST 100 (Benchmark)', linestyle='--', color='black', alpha=0.7)
ax[1].set_title('Out-of-Sample Backtest (2024-Present)')
ax[1].set_xlabel('Date')
ax[1].set_ylabel('Cumulative Return')
ax[1].legend()
ax[1].grid(True)

plt.tight_layout()
plt.show()

# SONUÇLAR
print("\n--- OPTIMAL ASSET ALLOCATION (Markowitz) ---")
for asset, weight in zip(assets, optimal_weights):
    print(f"{asset}: %{weight*100:.2f}")

total_ret_val = float(total_ret)
bench_ret_val = float(bench_ret)

print(f"\nBacktest Sonucu (2025 Başından Beri):")
print(f"Portföy Getirisi: %{total_ret_val:.2f}")
print(f"BIST 100 Getirisi: %{bench_ret_val:.2f}")
