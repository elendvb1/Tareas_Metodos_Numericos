import numpy as np
import matplotlib.pyplot as plt

# --- Definición del problema ---
def p(x):
    """Función medida: sin(x) + ruido gaussiano (media 0, sigma = 1e-5)."""
    ruido = np.random.normal(0, 1e-5, len(x))
    return np.sin(x) + ruido

def simpson_compuesto(f, a, b, n):
    """Regla compuesta de Simpson para n intervalos (n debe ser par)."""
    if n % 2 != 0:
        raise ValueError("n debe ser par")
    x = np.linspace(a, b, n + 1)
    y = f(x)
    h = (b - a) / n
    return (h / 3) * (y[0] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2]) + y[-1])

# --- Cálculo y comparación ---
a, b = 0, 1
valor_real = 1 - np.cos(1)

hs = []
errores = []

for n in [2**k for k in range(1, 13)]:  # de 2 a 4096 intervalos
    h = (b - a) / n
    I = simpson_compuesto(p, a, b, n)
    error = abs(I - valor_real)
    hs.append(h)
    errores.append(error)

# --- Gráfico del error ---
plt.figure(figsize=(7,5))
plt.loglog(hs, errores, 'o-', label='Error total')
plt.xlabel('h')
plt.ylabel('Error absoluto')
plt.title('Error del método de Simpson con ruido gaussiano')
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.legend()
plt.show()

# --- Resultados numéricos ---
print(f"Valor verdadero de la integral: {valor_real:.10f}")
print(f"Error mínimo obtenido: {min(errores):.2e}")

# --- Respuestas breves a las preguntas ---
print("\n--- Respuestas ---")
print("• El error disminuye al reducir h hasta que el ruido domina.")
print("• No se puede lograr una precisión de 1e-7 porque el ruido aleatorio")
print("  tiene una desviación estándar de 1e-5, lo que impone un límite físico.")
print("• El método de Simpson es de orden 4, pero en presencia de ruido")
print("  el error total deja de decrecer más allá del nivel del ruido.")