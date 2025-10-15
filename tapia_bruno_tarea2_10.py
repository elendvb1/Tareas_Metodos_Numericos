import numpy as np
import matplotlib.pyplot as plt

#Número de iteraciones.
N = 20

#Función, valor verdadero y error minimo.
f = lambda x: x**2 - 2.0
true = np.sqrt(2)
epsilon = 1.11e-16

#Método de Bisección.

a, b = 1.0, 2.0
x_bis_hist = []
err_bis = []

for k in range(N):
    c = 0.5 * (a + b)
    x_bis_hist.append(c)
    err_bis.append(abs(c - true))
    if f(a) * f(c) <= 0:
        b = c
    else:
        a = c

#Método de Newton.

x_newt_hist = []
err_newt = []

x = 2.0
for k in range(N):
    x_newt_hist.append(x)
    err_newt.append(abs(x - true))
    x = 0.5 * (x + 2.0 / x)


#Graficar los errores en escala logarítmica.

plt.figure(figsize=(8,5))
plt.semilogy(range(N), err_bis, 'o-', label='Bisección')
plt.semilogy(range(N), err_newt, 's-', label='Newton/Babilónico')
plt.xlabel('Iteración')
plt.ylabel('Error absoluto')
plt.title('Convergencia: Bisección vs Babilónico')
plt.axhline(epsilon, color='g', linestyle='--', label='Error minimo')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()

"""
Cuando el metodo de Newton llega a un error cercano a la máxima precisión de
cifras significativas que alcanza float64 (Error minimo en el gráfico entonces
el error deja de disminuir.
"""

# Nota: 7.0
