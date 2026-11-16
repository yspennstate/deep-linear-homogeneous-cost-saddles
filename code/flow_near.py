"""Start the H-flow near a zero-Hessian critical point theta* (built by chain_dcount.build) and see what happens.

If a positive-measure set of small perturbations flows back to the critical set, Assumption 17 fails with positive
probability for that architecture (the Gaussian density is positive near theta*).
Usage: python flow_near.py L w k j kill1 kill2 rk1 rk2 eps trials [layer:rank ...] [sSEED]
"""
import sys
import numpy as np
import chain_dcount as cd


def grads(Ws, G):
    L = len(Ws)
    out = []
    for l in range(L):
        A = None
        for m in range(l + 1, L):
            A = Ws[m] if A is None else Ws[m] @ A
        B = None
        for m in range(0, l):
            B = Ws[m] if B is None else Ws[m] @ B
        M = G if A is None else A.T @ G
        out.append(M if B is None else M @ B.T)
    return out


def main():
    a = sys.argv[1:]
    L, w, k, j, k1, k2, r1, r2 = (int(x) for x in a[:8])
    eps = float(a[8]); trials = int(a[9])
    seed = 0
    for tok in a[10:]:
        if tok.startswith("s"):
            seed = int(tok[1:])
        elif ":" in tok:
            lay, rr = tok.split(":"); cd.RANKS[int(lay)] = int(rr)
    rng = np.random.default_rng(seed)
    G = rng.standard_normal((w, w))
    _, n = cd.build(np.zeros(100000), L, w, k, j, (k1, k2), (r1, r2), G)
    p0 = rng.standard_normal(n)
    Ws0, _ = cd.build(p0, L, w, k, j, (k1, k2), (r1, r2), G)
    x0 = cd.flat(Ws0)
    r0 = np.linalg.norm(x0)
    counts = {"escape": 0, "stuck": 0}
    for t in range(trials):
        xi = rng.standard_normal(x0.size); xi *= eps * r0 / np.linalg.norm(xi)
        Ws = cd.unflat(x0 + xi, L, w)
        steps = 0
        mind = np.inf
        while True:
            gs = grads(Ws, G)
            x = cd.flat(Ws)
            r = np.linalg.norm(x)
            g = np.sqrt(sum(float(np.sum(q * q)) for q in gs))
            d = np.linalg.norm(x - x0)
            mind = min(mind, d)
            if r > 1e3 * r0:
                counts["escape"] += 1
                print(f"trial {t}: escape after {steps} steps, closest approach {mind / r0:.2e}", flush=True)
                break
            if steps >= 20000 or g < 1e-10 * r0 ** (L - 1):
                counts["stuck"] += 1 if d < 0.5 * r0 else 0
                print(f"trial {t}: STOPPED after {steps} steps, |g|={g:.2e}, dist/r0={d / r0:.2e}, "
                      f"r/r0={r / r0:.3f}, H={cd.H(Ws, G):.4e}", flush=True)
                break
            dt = 1e-2 * r / max(g, 1e-300)
            h0 = cd.H(Ws, G)
            while True:
                Wn = [W - dt * q for W, q in zip(Ws, gs)]
                if cd.H(Wn, G) <= h0 - 0.3 * dt * g * g or dt < 1e-300:
                    break
                dt *= 0.5
            Ws = Wn
            steps += 1
    print(f"L={L} w={w} eps={eps} trials={trials} counts={counts} H(theta*)={cd.H(Ws0, G):.2e} r0={r0:.2f}", flush=True)


if __name__ == "__main__":
    main()
