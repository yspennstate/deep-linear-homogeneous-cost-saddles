"""Gradient flow of H(theta) = <G, W_L ... W_1> from Gaussian init.

Assumption 17 of Jacot et al. (arXiv 2106.15933) excludes initializations whose H-flow converges to a critical
point of H.  For L > 3 they prove such initializations have probability at most 1/2.  Here we count, for random
Gaussian init, how many flows (a) blow up in finite time, (b) then align with an optimal escape direction
(H/|theta|^L -> -s*), (c) stay bounded (converge to a critical point).

Usage: python h_flow_test.py L n0 nL w trials seed
"""
import sys
import numpy as np


def prod(Ws, a, b):
    """W_{b-1} ... W_a (layers a..b-1), identity-free: returns None for an empty product."""
    P = None
    for k in range(a, b):
        P = Ws[k] if P is None else Ws[k] @ P
    return P


def grads(Ws, G):
    L = len(Ws)
    out = []
    for l in range(L):
        A = prod(Ws, l + 1, L)  # layers above l
        B = prod(Ws, 0, l)      # layers below l
        M = G if A is None else A.T @ G
        out.append(M if B is None else M @ B.T)
    return out


def H(Ws, G):
    return float(np.sum(G * prod(Ws, 0, len(Ws))))


def run(L, n0, nL, w, G, rng, sstar):
    dims = [n0] + [w] * (L - 1) + [nL]
    Ws = [rng.standard_normal((dims[k + 1], dims[k])) for k in range(L)]
    h0 = H(Ws, G)
    # Flow on the sphere-plus-radius: integrate the direction with step scaled by |theta|^(L-2).
    t_steps = 0
    while True:
        gs = grads(Ws, G)
        nrm2 = sum(float(np.sum(W * W)) for W in Ws)
        g2 = sum(float(np.sum(g * g)) for g in gs)
        r = np.sqrt(nrm2)
        if r > 1e4:
            q = H(Ws, G) / r ** L
            return "escape", h0, q, t_steps
        if g2 < 1e-24 or (t_steps > 400000):
            return "bounded", h0, H(Ws, G) / max(r, 1e-300) ** L, t_steps
        # adaptive step: move at most 1e-3 of the radius
        dt = 1e-3 * r / max(np.sqrt(g2), 1e-300)
        Ws = [W - dt * g for W, g in zip(Ws, gs)]
        t_steps += 1


def main():
    L, n0, nL, w, trials, seed = (int(x) for x in sys.argv[1:7])
    rng = np.random.default_rng(seed)
    G = rng.standard_normal((nL, n0))
    s1 = np.linalg.svd(G, compute_uv=False)[0]
    sstar = s1  # H/|theta|^L at the optimum is -s1 / L^(L/2)
    qopt = -s1 / L ** (L / 2)
    counts = {"escape": 0, "bounded": 0}
    worst = 0.0
    for t in range(trials):
        kind, h0, q, steps = run(L, n0, nL, w, G, rng, sstar)
        counts[kind] += 1
        if kind == "escape":
            worst = max(worst, abs(q - qopt) / abs(qopt))
        else:
            print(f"bounded: trial {t} H0={h0:.3g} q={q:.3g} steps={steps}", flush=True)
    print(f"L={L} n0={n0} nL={nL} w={w} trials={trials} seed={seed} counts={counts} "
          f"qopt={qopt:.5g} worst_rel_dev_at_escape={worst:.3g}", flush=True)


if __name__ == "__main__":
    main()
