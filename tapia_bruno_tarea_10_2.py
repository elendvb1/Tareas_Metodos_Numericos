# %%

import numpy as np
import matplotlib.pyplot as plt

def rk4_system(f, a, b, y0, n_steps):
    """
    f : función f(t, Y) donde Y = [y, y', ..., y^(n-1)]
    a, b : intervalo
    y0 : vector con condiciones iniciales
    n_steps : número de pasos
    """
    t = np.linspace(a, b, n_steps+1)
    h = (b - a) / n_steps
    Y = np.zeros((n_steps+1, len(y0)))
    Y[0] = y0

    for i in range(n_steps):
        ti = t[i]
        Yi = Y[i]

        k1 = f(ti, Yi)
        k2 = f(ti + h/2, Yi + h*k1/2)
        k3 = f(ti + h/2, Yi + h*k2/2)
        k4 = f(ti + h,   Yi + h*k3)

        Y[i+1] = Yi + (h/6)*(k1 + 2*k2 + 2*k3 + k4)

    return t, Y


# Ejemplo del problema:
# y'' - 2 y' + y = t e^t - t
# Convertimos a sistema:
#
# y1 = y
# y2 = y'
# y1' = y2
# y2' = f = 2 y2 - y1 + t e^t - t

def f_example(t, Y):
    y, yp = Y
    return np.array([yp, 2*yp - y + t*np.exp(t) - t])

# Condiciones iniciales
y0 = [0, 0]  

# Parámetros
a = 0
b = 1
n_steps = 100

# Resolver
t, Y = rk4_system(f_example, a, b, y0, n_steps)

# Solución analítica
y_exact = (1/6)*t**3*np.exp(t) - t*np.exp(t) + 2*np.exp(t) - t - 2


# Graficar resultado

plt.plot(t, Y[:,0], label="RK4", linewidth=2)
plt.plot(t, y_exact, '--', label="Exacta")
plt.xlabel("t")
plt.ylabel("y(t)")
plt.title("Comparación solución RK4 vs exacta")
plt.legend()
plt.grid(True)
plt.show()

# Nota: 7.0