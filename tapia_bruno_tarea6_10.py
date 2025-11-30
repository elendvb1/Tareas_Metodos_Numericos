import numpy as np

# Funcion original
def f(x):
    return 4*x**3 - 3*x**2 + x - 5

# Construir spline natural: devuelve vector de segundas derivadas m
def cubic_spline_natural(x, y):
    n = len(x)
    h = np.diff(x)

    # Matriz del sistema y vector rhs
    A = np.zeros((n, n))
    b = np.zeros(n)

    # Condiciones naturales m[0]=0, m[n-1]=0
    A[0, 0] = 1
    A[-1, -1] = 1

    # Ecuaciones para nodos interiores
    for i in range(1, n - 1):
        A[i, i - 1] = h[i - 1]
        A[i, i]     = 2 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]
        b[i]        = 6 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])

    m = np.linalg.solve(A, b)
    return m

# Evaluar spline en punto xp dados nodos x,y y segundas derivadas m
def eval_spline(x, y, m, xp):
    # buscar intervalo
    i = np.searchsorted(x, xp) - 1
    if i < 0:
        i = 0
    if i >= len(x) - 1:
        i = len(x) - 2

    h = x[i+1] - x[i]
    A = (x[i+1] - xp) / h
    B = (xp - x[i]) / h

    S = (A*y[i] + B*y[i+1] +
         ((A**3 - A) * m[i] + (B**3 - B) * m[i+1]) * (h**2) / 6)
    return S

# --- Parte 1: spline con solo dos puntos ---
x2 = np.array([-1.0, 1.0])
y2 = f(x2)
m2 = cubic_spline_natural(x2, y2)

# --- Parte 2: spline con cuatro puntos equidistantes ---
x4 = np.linspace(-1.0, 1.0, 4)
y4 = f(x4)
m4 = cubic_spline_natural(x4, y4)

# Malla para comparar
xs = np.linspace(-1.0, 1.0, 200)
fs = f(xs)
s2 = np.array([eval_spline(x2, y2, m2, xi) for xi in xs])
s4 = np.array([eval_spline(x4, y4, m4, xi) for xi in xs])

# Calcular errores maximos
err2 = np.max(np.abs(fs - s2))
err4 = np.max(np.abs(fs - s4))

# Impresion de resultados (documentacion con solo primera letra en mayuscula)
print("Maximo error entre f y spline con dos puntos:", err2)
print("Maximo error entre f y spline con cuatro puntos:", err4)

# Breve respuesta a la pregunta del enunciado

"""
La spline cubica con solo dos puntos es diferente del polinomio original
porque la condicion natural impone que la segunda derivada en los extremos sea cero.
El polinomio original no satisface necesariamente esa condicion,
por lo que la spline produce otro cubico que pasa por los mismos puntos pero tiene curvatura distinta.
Al usar cuatro puntos equidistantes, la spline incorpora mas informacion local
y aproxima mejor al polinomio original, pero sigue siendo una funcion por tramos y no un unico polinomio global.
"""
