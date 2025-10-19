import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb

#Número total de partículas que llegan al sensor en un segundo
N_total = 2  

#Valores de r a estudiar
r_values = [50, 100, 500, 1000]

#Valores posibles de n (número de partículas en un pixel)
n_values = np.arange(0, N_total + 1)  

def binomial_pmf(n, N, p):
    return comb(N, n) * (p**n) * ((1-p)**(N-n))

plt.figure(figsize=(12, 8))

for r in r_values:
    p = 2 / r
    binom_probs = [binomial_pmf(n, N_total, p) for n in n_values]
    
    plt.plot(n_values, binom_probs, 'o-', label=f"Binomial r={r}")

plt.xlabel("Número de partículas en un pixel (n)")
plt.ylabel("Probabilidad P(X=n)")
plt.title("Distribución Binomial de partículas en un pixel")
plt.legend()
plt.grid(True)
plt.show()
""" Es posible usar Poisson cuando el número de pixeles es muy grande (r>>1)"""

# Nota: 7.0
