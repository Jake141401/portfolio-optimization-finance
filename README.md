import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Modelli dei Portafogli di Investimento
# Titoli considerati: JPMorgan, Goldman Sachs, Visa

print("Analisi dei Portafogli di Investimento")
print("Titoli: JPMorgan, Goldman Sachs, Visa")

# Dati reali sui prezzi (esempi fittizi simili a valori reali)
prices = {
    "JPMorgan": [269.53, 270.36, 264.95, 268.24, 268.15],
    "Goldman": [458.10, 462.85, 465.75, 461.30, 463.95],
    "Visa": [357.71, 355.48, 352.85, 371.40, 373.31]
}

# Creazione del DataFrame
data = pd.DataFrame(prices)

# Calcolo dei rendimenti
returns = data.pct_change().dropna()

# Calcolo delle statistiche per i titoli
stats = {}
for stock in data.columns:
    mean_return = returns[stock].mean()
    volatility = returns[stock].std()
    stats[stock] = {
        "Rendimento Medio": mean_return,
        "Volatilità": volatility
    }

stats_df = pd.DataFrame(stats).T

# Matrici di correlazione e covarianza
correlation_matrix = returns.corr()
covariance_matrix = returns.cov()

# Creazione di portafogli con pesi differenti
portfolios = [
    np.array([0.50, 0.30, 0.20]),  # 1 Forte esposizione su JPMorgan
    np.array([0.40, 0.40, 0.20]),  # 2 Bilanciato tra JPMorgan e Goldman
    np.array([0.25, 0.35, 0.40]),  # 3 Più peso su Visa
    np.array([0.20, 0.50, 0.30]),  # 4 Predominanza Goldman
    np.array([0.33, 0.33, 0.34])   # 5 Equamente distribuito 
]

portfolio_metrics = []

for weights in portfolios:
    portfolio_return = np.dot(weights, returns.mean())
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov(), weights)))
    portfolio_metrics.append({
        "Rendimento": portfolio_return,
        "Volatilità": portfolio_volatility
    })

portfolio_df = pd.DataFrame(portfolio_metrics)

# Calcolo dello Sharpe Ratio (assumiamo risk-free = 0)
portfolio_df["Sharpe Ratio"] = portfolio_df["Rendimento"] / portfolio_df["Volatilità"]

# Raccomandazione basata sul massimo Sharpe Ratio
recommendation = portfolio_df["Sharpe Ratio"].idxmax()

# Output dei risultati
print("\nStatistiche dei titoli (JPMorgan, Goldman Sachs, Visa):")
print(stats_df)

print("\nCorrelazioni:")
print(correlation_matrix)

print("\nCovarianze:")
print(covariance_matrix)

print("\nMetriche dei portafogli:")
print(portfolio_df)

print(f"\n Raccomandazione (Sharpe Ratio): Il portafoglio più efficiente è il Portafoglio {recommendation + 1}.\n")

# --------------------- GRAFICO RISCHIO-RENDIMENTO ---------------------

# Estrai dati per i titoli singoli
titoli_x = stats_df["Volatilità"]
titoli_y = stats_df["Rendimento Medio"]
titoli_nomi = stats_df.index.tolist()

# Estrai dati per i portafogli
portafogli_x = portfolio_df["Volatilità"]
portafogli_y = portfolio_df["Rendimento"]

#  figura
plt.figure(figsize=(10, 6))
plt.style.use('default')

# Punti dei titoli 
for i in range(len(titoli_nomi)):
    plt.scatter(titoli_x[i], titoli_y[i], color='orange', label=f"{titoli_nomi[i]} (Titolo)", marker='x')
    plt.text(titoli_x[i] + 0.00005, titoli_y[i], titoli_nomi[i], color='orange', fontsize=9)

# Punti dei portafogli 
plt.plot(portafogli_x, portafogli_y, marker='o', color='blue', label='Portafogli')

# Evidenzia il portafoglio consigliato 
plt.scatter(portafogli_x.iloc[recommendation], portafogli_y.iloc[recommendation], color='green', s=100, label=f'Portafoglio {recommendation + 1} Consigliato')

# Etichette dei portafogli
for i, (x, y) in enumerate(zip(portafogli_x, portafogli_y)):
    plt.text(x + 0.00005, y, f"Portafoglio {i+1}", color='blue', fontsize=8)

# Titoli e assi
plt.title("Efficienza dei Portafogli Finanziari")
plt.xlabel("Volatilità")
plt.ylabel("Rendimento")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.3)

# Mostra il grafico
plt.tight_layout()
plt.show()
