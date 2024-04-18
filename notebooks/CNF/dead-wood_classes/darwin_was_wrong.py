# ---
# jupyter:
#   jupytext:
#     formats: py,ipynb
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # The pursuit of eternal life in trees
#
# ... it results from the deeper concept of trying to produce as much offspring as possible during a lifetime.

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp, quad
from sympy import symbols, exp, solve, simplify


# +
def _logistic(x, x0, L, k):
    return L / (1 + np.exp(-k * (x - x0)))

logistic = np.vectorize(_logistic, excluded=["x0", "L", "k"])

# +
x0s = np.arange(0, 5, 1)
xs = np.arange(-5, 15, 0.1)

for x0 in x0s:
    l = plt.plot(xs, logistic(xs, x0, 1, 1), label=str(x0))
    plt.axvline(x0, c=l[0].get_color())

plt.legend()
# -

z, z0, L, k = symbols("z, z_0, L, k")
f = L / (1 + exp(-k * (z - z0)))
df = f.diff(z)
df

# +
for x0 in x0s:
    l = plt.plot(xs, [df.subs({z: x, z0: x0, L: 1, k: 1}) for x in xs], label=x0)
    plt.axvline(x0, c=l[0].get_color())

plt.legend()
# -

ddf = df.diff(z)
ddf

simplify(solve(ddf, z)[0])

# +
x_ast = 5600
r = 0.05
k = 0.095

#x_ast = 500
#r = 0.01
#k = 0.07


t0 = 45
tend = 80
ts = np.arange(0, tend + 1, 1)

x0 = x_ast / 2 # x(t0)
x0 = np.array(x0).reshape((1,))


# +
def g_x(t, x):
    return k / x_ast * x * (x_ast - x)

# actually, you would have to solve it from the mid point forward and compute the left hand side by symmetry
res_x = solve_ivp(g_x, t_span=(t0, np.maximum(tend, 2*t0)), y0=x0, dense_output=True)
res_x


# +
def _f_x(t):
    f_x_right = lambda t: res_x.sol(t).reshape(-1)
    f_x_left = lambda t: f_x_right(t0) - (f_x_right(t0 + (t0-t)) - f_x_right(t0))
    return f_x_right(t) if t >= t0 else f_x_left(t)

f_x = np.vectorize(_f_x)

plt.plot(ts, f_x(ts), label="x")


# +
def _phi(x):
    return k * (1 - x/x_ast) + r
#    return phi0 # simulate constant phi

phi = np.vectorize(_phi)

def g_y(t, y):
    x = f_x(t)
    return (phi(x) - r) * y
#    alpha = 0.4
#    r_ = 0.4 * np.exp(-0.1 * (t-1)) + 0.01
#    phi = (1 - alpha) / (alpha * t + 1) + r_
#    return (phi - r) * y



# -

y0 = np.array(f_x(0)).reshape((1,))
res_y = solve_ivp(g_y, t_span=(0, tend), y0=y0, dense_output=True)

f_y = lambda t: res_y.sol(t).reshape(-1)
plt.plot(ts, f_x(ts), label="x")
plt.plot(ts, f_y(ts), label="y")
plt.legend()

plt.plot(ts_, np.diff(f_y(ts)), label=r"$\dot{x}(t)$")
plt.legend()

plt.plot(f_a(ts), f_y(ts), label=r"$x(a)$")
plt.legend()

plt.plot(f_a(ts_), np.diff(f_y(ts)), label=r"$\dot{x}(a)$")
plt.legend()

plt.plot(g_a_t(ts_), np.diff(f_y(ts)), label=r"$\dot{x}(a)$")
plt.legend()

plt.plot(ts__, np.diff(np.diff(f_y(ts))), label=r"$\ddot{x}(t)$")
plt.legend()

plt.plot(ts_, np.diff(f_y(ts)) / f_y(ts_), label=r"$\dot{x}(t) / x(t)$")
plt.legend()

a0 = 0.
a0 = np.array(a0).reshape((1,))


def g_a(t, a):
    y = f_y(t)
    return 1 - a * phi(y) 


res_a = solve_ivp(g_a, t_span=(0, tend), y0=a0, dense_output=True)

f_a = lambda t: res_a.sol(t).reshape(-1)
plt.plot(ts, f_a(ts), label="a(t)")
plt.legend()

f_a = lambda t: res_a.sol(t).reshape(-1)
plt.plot(f_y(ts), f_a(ts), label="a(x)")
plt.legend()

f_a = lambda t: res_a.sol(t).reshape(-1)
plt.plot(phi_t(ts), f_a(ts), label=r"$a(\varphi)$")
plt.legend()

# +
plt.plot(ts, phi(f_y(ts)), label=r"$\varphi(t)$")
#plt.plot(ts, np.minimum(0.2, 1/f_a(ts) * (1 - np.exp(-phi(f_y(0)) * ts))) , label="phi_from_a")

phi_t = lambda t: phi(f_y(t))
g_a_t = lambda t: g_a(t, f_a(t))
phi0 = phi_t(0)[0]

ts_ = ts[:-1]
ts__ = ts[:-2]
#plt.plot(ts__, 1/np.diff(g_a_t(ts_)) * (- np.diff(np.diff(g_a_t(ts))) - f_a(ts__) * np.diff(phi_t(ts_))))
plt.plot(ts, (1 - g_a_t(ts)) / f_a(ts), ls = "--")

plt.legend()
# -

plt.plot(f_y(ts), phi_t(ts), label=r"$\varphi(x)$")
plt.legend()

plt.plot(f_a(ts), phi_t(ts), label=r"$\varphi(a)$")
plt.legend()

plt.plot(g_a_t(ts), phi_t(ts), label=r"$\varphi(\dot{a})$")
plt.legend()

plt.plot(ts, g_a_t(ts), label=r"$\dot{a}(t)$")
plt.plot(ts, np.exp(-phi0 * ts))
plt.plot(ts_, np.diff(phi_t(ts)) / (phi_t(ts_) - phi0))
plt.legend()

plt.plot(f_y(ts), g_a_t(ts), label=r"$\dot{a}(x)$")
plt.legend()

plt.plot(phi_t(ts), g_a_t(ts), label=r"$\dot{a}(\varphi)$")
plt.legend()

plt.plot(f_y(ts), phi_t(ts) / r, label=r"$\varphi/r$")
plt.xlabel("x");
plt.legend()

plt.plot(phi_t(ts__), np.diff(np.diff(f_a(ts))), label=r"$\varphi")
plt.xlabel("x");



plt.plot(ts_, np.diff(g_a_t(ts)))
plt.plot(ts_, - (g_a_t(ts_) * phi_t(ts_) + f_a(ts_) * np.diff(phi_t(ts))))

plt.plot(ts, f_a(ts) * phi_t(ts), label=r"$a \cdot \varphi$")
plt.legend()

plt.plot(f_y_t(ts), f_a_t(ts))



# +
def _dead_stem_wood(t):
#    return quad(lambda t: r * f_y(t), 0, t)[0]
    _ts = ts[ts <= t]
    r_discounted = r * np.exp(-0.15 * f_a(_ts))
    return np.trapz(r_discounted * f_y(_ts), _ts)
    
dead_stem_wood = np.vectorize(_dead_stem_wood)
# -

stem_wood = lambda t: f_y(t) + dead_stem_wood(t)

plt.plot(ts, f_y(ts), label="live stem wood")
plt.plot(ts, dead_stem_wood(ts), label="dead stem wood")
plt.plot(ts, stem_wood(ts), label="stem wood")
plt.legend()

r_discounted = r * np.exp(-0.2 * f_a(ts))
plt.plot(ts, r_discounted)


plt.plot(ts, phi(f_x(ts)), label=r"$\varphi$")
plt.plot(ts, (lambda t: 1/f_a(t))(ts), label="1/a")
plt.legend()

plt.plot(ts[:-1], - (np.diff(g_a(ts, f_a(ts))) / g_a(ts[:-1], f_a(ts[:-1]))))


def g(t, v):
    a, p = v
#    print(a, p)
    res = 0 * v
    res[0] = float(1 - a * p)
    a_dot_dot = - phi0 * res[0]
    res[1] = float(- res[0] * p - a_dot_dot)
#    print(res)

    return res


v0 = np.array([0, phi0])
print(v0)
res = solve_ivp(g, t_span=(0, 80), y0=v0, dense_output=True)

res

h_a = lambda t: res.sol(t)[0]
h_phi = lambda t: res.sol(t)[1]

plt.plot(ts, h_a(ts))

plt.plot(ts, h_phi(ts))

r

plt.plot(ts, 0.6/(0.4*ts))


