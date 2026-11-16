"""Depth four, rectangular G of full rank (n0=3, n4=1, w=1).

Builds the point the proof of Theorem 5.1 leaves over (W2=W4=0, W1 G^T=0,
W3 != 0) and checks by finite differences that it is a critical point of
H(theta)=Tr(G^T W4 W3 W2 W1) with zero Hessian and all D_l invertible.
Then checks that a random point with W2=0 and W4 != 0 is not a zero-Hessian
critical point with D invertible, as the proof says.
"""
import numpy as np

rng = np.random.default_rng(0)
n = [3, 1, 1, 1, 1]          # n0..n4
G = rng.standard_normal((n[4], n[0]))
shapes = [(n[l + 1], n[l]) for l in range(4)]
sizes = [a * b for a, b in shapes]


def unpack(t):
    out, k = [], 0
    for (a, b), s in zip(shapes, sizes):
        out.append(t[k:k + s].reshape(a, b))
        k += s
    return out


def H(t):
    W1, W2, W3, W4 = unpack(t)
    return np.trace(G.T @ W4 @ W3 @ W2 @ W1)


def grad(t, h=1e-6):
    g = np.zeros_like(t)
    for i in range(t.size):
        e = np.zeros_like(t); e[i] = h
        g[i] = (H(t + e) - H(t - e)) / (2 * h)
    return g


def hess(t, h=1e-4):
    N = t.size
    M = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            ei = np.zeros(N); ei[i] = h
            ej = np.zeros(N); ej[j] = h
            M[i, j] = (H(t + ei + ej) - H(t + ei - ej) - H(t - ei + ej) + H(t - ei - ej)) / (4 * h * h)
    return M


def D(t):
    W = unpack(t)
    return [W[l + 1].T @ W[l + 1] - W[l] @ W[l].T for l in range(3)]


# W1 orthogonal to the row of G
v = rng.standard_normal(3)
g = G[0]
v = v - (v @ g) / (g @ g) * g
W1 = v.reshape(1, 3)
W2 = np.zeros((1, 1)); W3 = np.array([[1.7]]); W4 = np.zeros((1, 1))
t = np.concatenate([W1.ravel(), W2.ravel(), W3.ravel(), W4.ravel()])
print("leftover point: |grad| =", np.abs(grad(t)).max(), " |Hess| =", np.abs(hess(t)).max())
print("  D_l:", [float(np.linalg.det(d)) for d in D(t)])

# a W2=0, W4!=0 point is not critical with zero Hessian
W4b = np.array([[0.9]])
tb = np.concatenate([W1.ravel(), W2.ravel(), W3.ravel(), W4b.ravel()])
print("W2=0, W4!=0: |grad| =", np.abs(grad(tb)).max(), " |Hess| =", np.abs(hess(tb)).max())
