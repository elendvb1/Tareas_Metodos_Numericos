import numpy as np
import matplotlib.pyplot as plt

def f(t, y):
    return y - t**2 + 1

# Método de Euler
def euler(f, y0, t0, tf, h):
    N = int((tf - t0) / h)
    t = np.linspace(t0, tf, N + 1)
    y = np.zeros(N + 1)
    y[0] = y0
    for i in range(N):
        y[i + 1] = y[i] + h * f(t[i], y[i])
    return t, y

# Solución analítica
def y_exacta(t):
    return (t + 1)**2 - 0.5 * np.exp(t)

# Parámetros
h = 1e-4
t0, tf = 0, 2
delta_vals = np.linspace(-0.1, 0.1, 21)
y2_vals = []

# Cálculo de y(2) para varios valores de delta_0
for delta0 in delta_vals:
    y0 = 0.5 + delta0
    t, y = euler(f, y0, t0, tf, h)
    y2_vals.append(y[-1])

# Solución analítica para comparación
y2_analitico = y_exacta(2)

# Gráfico de y(2) en función de delta0
plt.figure(figsize=(8,5))
plt.plot(delta_vals, y2_vals, 'o-', label='Método de Euler')
plt.axhline(y2_analitico, color='r', linestyle='--', label='Solución analítica y(2)')
plt.xlabel('delta0')
plt.ylabel('y(2)')
plt.title('Dependencia de y(2) con el error inicial delta0')
plt.legend()
plt.grid(True)
plt.show()

print("Valor analítico de y(2):", y2_analitico)
print("Valores numéricos de y(2) con distintos delta0:")
for d, y2 in zip(delta_vals, y2_vals):
    print(f"delta0 = {d:+.3f} -> y(2) = {y2:.6f}")

# Se observa que y(2) varía linealmente con delta0.
# Esto ocurre porque la ecuación diferencial es lineal en y, y el error inicial delta0
# se propaga aproximadamente multiplicado por un factor e^{(t_final - t_inicial)} ≈ e^2 ≈ 7.39.
# Por tanto, pequeños errores en y(0) se amplifican de manera exponencial.
# El método de Euler, con paso h = 1e-4, reproduce muy bien la solución analítica.