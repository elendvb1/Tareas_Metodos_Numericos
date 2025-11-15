import numpy as np
import matplotlib.pyplot as plt

class MinStep(Exception):
    pass

def rk4_step(func, fval, t_val, y_val, h_val):
    k1 = h_val * fval
    k2 = h_val * func(t_val + h_val/2, y_val + k1/2)
    k3 = h_val * func(t_val + h_val/2, y_val + k2/2)
    k4 = h_val * func(t_val + h_val,   y_val + k3)
    return (k1 + 2*k2 + 2*k3 + k4) / 6

def rkf_mod(fun, t0, t_end, y0, tol, h_min, h_max):
    t_current = t0
    y_current = y0
    h = h_max
    continue_flag = True

    t_arr = [t_current]
    y_arr = [y_current]
    h_arr = []
    r_arr = []
    eval_count = 0

    while continue_flag:
        # 6 function evaluations per step:
        k1 = h * fun(t_current, y_current)
        k2 = h * fun(t_current + h/4,     y_current + k1/4)
        k3 = h * fun(t_current + 3*h/8,   y_current + (3*k1/32 + 9*k2/32))
        k4 = h * fun(t_current + 12*h/13, y_current + (1932*k1/2197 - 7200*k2/2197 + 7296*k3/2197))
        k5 = h * fun(t_current + h,       y_current + (439*k1/216 - 8*k2 + 3680*k3/513 - 845*k4/4104))
        k6 = h * fun(t_current + h/2,     y_current + (-8*k1/27 + 2*k2 - 3544*k3/2565 + 1859*k4/4104 - 11*k5/40))
        eval_count += 6

        R = abs(k1/360 - 128*k3/4275 - 2197*k4/75240 + k5/50 + 2*k6/55)

        if R <= tol:
            t_current += h
            y_current += (25*k1/216 + 1408*k3/2565 + 2197*k4/4104 - k5/5)
            t_arr.append(t_current)
            y_arr.append(y_current)
            h_arr.append(h)
            r_arr.append(R)

        q = 0.84 * (tol / R)**(1/4)
        if q <= 0.1:
            h = 0.1 * h
        elif q >= 4:
            h = 4 * h
        else:
            h = q * h

        if h > h_max:
            h = h_max
        if t_current >= t_end:
            continue_flag = False
        elif t_current + h > t_end:
            h = t_end - t_current
        elif h < h_min:
            raise MinStep('Se excedió el paso mínimo.')

    return np.array(t_arr), np.array(y_arr), np.array(h_arr), np.array(r_arr), eval_count

def apce_mod(fun, t0, t_end, y0, tol, h_min, h_max):
    h = h_max
    continue_flag = True
    last_flag = False

    t_arr = [t0]
    y_arr = [y0]
    h_arr = [h]
    r_arr = [-1]
    eval_count = 0

    f_current = fun(t0, y0)
    eval_count += 1
    f_im1 = f_im2 = f_im3 = 0

    # Inicialización con 3 pasos de RK4
    for i in range(3):
        y_arr.append(y_arr[i] + rk4_step(fun, f_current, t_arr[i], y_arr[i], h))
        eval_count += 3
        t_arr.append(t_arr[i] + h)
        h_arr.append(h)
        r_arr.append(-1)
        f_im3 = f_im2
        f_im2 = f_im1
        f_im1 = f_current
        f_current = fun(t_arr[i+1], y_arr[i+1])
        eval_count += 1

    idx = 4
    t_next = t_arr[3] + h

    while continue_flag:
        # Predictor (Adams-Bashforth 4 pasos)
        y_predict = y_arr[idx-1] + h*(55*f_current - 59*f_im1 + 37*f_im2 - 9*f_im3)/24
        # Corrector (Adams-Moulton 3 pasos)
        y_correct = y_arr[idx-1] + h*(9*fun(t_next, y_predict) + 19*f_current - 5*f_im1 + f_im2)/24
        eval_count += 1

        error_est = 19 * abs(y_correct - y_predict) / (270 * h)

        if error_est <= tol:
            y_arr.append(y_correct)
            t_arr.append(t_next)
            h_arr.append(h)
            r_arr.append(error_est)
            f_im3 = f_im2
            f_im2 = f_im1
            f_im1 = f_current
            f_current = fun(t_arr[idx], y_arr[idx])
            eval_count += 1
            if last_flag:
                continue_flag = False
            else:
                idx += 1
                # Ajuste del tamaño de paso
                if error_est <= 0.1*tol or t_arr[idx-1] + h > t_end:
                    delta = (tol/(2*error_est))**(1/4)
                    if delta > 4:
                        h = 4*h
                    else:
                        h = delta*h
                    if h > h_max:
                        h = h_max
                    if t_arr[idx-1] + 4*h > t_end:
                        h = (t_end - t_arr[idx-1]) / 4
                        last_flag = True
                    # Recalcular 3 pasos iniciales de nuevo
                    for j in range(3):
                        y_arr.append(y_arr[idx+j-1] + rk4_step(fun, f_current, t_arr[idx+j-1], y_arr[idx+j-1], h))
                        eval_count += 3
                        t_arr.append(t_arr[idx+j-1] + h)
                        h_arr.append(h)
                        r_arr.append(-1)
                        f_im3 = f_im2
                        f_im2 = f_im1
                        f_im1 = f_current
                        f_current = fun(t_arr[idx+j], y_arr[idx+j])
                        eval_count += 1
                    idx += 3
        else:
            delta = (tol/(2*error_est))**(1/4)
            if delta < 0.1:
                h = 0.1*h
            else:
                h = delta*h
            if h < h_min:
                raise MinStep('Tamaño de paso mínimo excedido')
            else:
                if last_flag:
                    idx -= 3
                    f_current = f_im3
                    for j in range(3):
                        y_arr[idx+j] = y_arr[idx+j-1] + rk4_step(fun, f_current, t_arr[idx+j-1], y_arr[idx+j-1], h)
                        eval_count += 3
                        t_arr[idx+j] = t_arr[idx+j-1] + h
                        h_arr[idx+j] = h
                        r_arr[idx+j] = -1
                        f_im3 = f_im2
                        f_im2 = f_im1
                        f_im1 = f_current
                        f_current = fun(t_arr[idx+j], y_arr[idx+j])
                        eval_count += 1
                    idx += 3
                else:
                    for j in range(3):
                        y_arr.append(y_arr[idx+j-1] + rk4_step(fun, f_current, t_arr[idx+j-1], y_arr[idx+j-1], h))
                        eval_count += 3
                        t_arr.append(t_arr[idx+j-1] + h)
                        h_arr.append(h)
                        r_arr.append(-1)
                        f_im3 = f_im2
                        f_im2 = f_im1
                        f_im1 = f_current
                        f_current = fun(t_arr[idx+j], y_arr[idx+j])
                        eval_count += 1
                    idx += 3
        t_next = t_arr[idx-1] + h

    return np.array(t_arr), np.array(y_arr), np.array(h_arr), np.array(r_arr), eval_count

# Definición del problema:
def ode_fun(t, y):
    return (1/t) * (y**2 + y)

t0 = 1.0
t_final = 3.0
y_initial = -2.0
h_minimum = 0.001
h_maximum = 0.2
tolerances = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]

evals_rkf = [ rkf_mod(ode_fun, t0, t_final, y_initial, tol, h_minimum, h_maximum)[-1] for tol in tolerances ]
evals_apce = [ apce_mod(ode_fun, t0, t_final, y_initial, tol, h_minimum, h_maximum)[-1] for tol in tolerances ]

plt.figure(figsize=(8,6))
plt.loglog(tolerances, evals_rkf, label='RKF-Modificado', color='magenta', marker='o')
plt.loglog(tolerances, evals_apce, label='APC-Modificado', color='teal', marker='x')
plt.xlabel('Tolerancia')
plt.ylabel('Número de evaluaciones')
plt.title('Comparación métodos con control de error')
plt.legend()
plt.grid(True, which="both", ls="--", lw=0.5)
plt.show()