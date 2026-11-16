"""Leading-order escape test at a one-zero-layer, zero-Hessian critical point of H = Tr[G^T W_L ... W_1].

H is multilinear in the layers and W_k = 0 at the point, so every Taylor term H_m(delta) is LINEAR in delta_k:
H_m(x, y) = <x, B_m(y, ..., y)> with x = delta_k, y = the other layers.  Under the descent flow, the blow-up
delta = r u gives r' = -m H_m(u) r^{m-1} (to leading order), so the flow can converge to the point only along
directions u with H_m(u) >= 0 that attract the sphere-projected descent of H_m.  The scaling (x, y) -> (a x, b y)
on the sphere shows every spherical critical direction with H_m(u) > 0 is a strict local MAXIMUM along that curve,
hence not a local minimum.  This script checks it numerically: projected descent of H_3 on the unit sphere from
many random starts; we record the final values f = H_3(u) and the projected gradient norm.

Usage: python cubic_test.py <chain_dcount args...> [nstart=N]
"""
import sys
import numpy as np
import chain_dcount as cd


def prod(Ws, lo, hi, w):
    """W_hi ... W_lo (1-based inclusive); identity if empty."""
    P = np.eye(w)
    for l in range(lo, hi + 1):
        P = Ws[l - 1] @ P
    return P


def gradH(Ws, G):
    L = len(Ws); w = Ws[0].shape[0]
    out = []
    for l in range(1, L + 1):
        left = prod(Ws, l + 1, L, w)     # W_L .. W_{l+1}
        right = prod(Ws, 1, l - 1, w)    # W_{l-1} .. W_1
        out.append(left.T @ G @ right.T)
    return out


def coeffs(fun, deg, smax=0.5):
    """Polynomial coefficients c_0..c_deg of s -> fun(s) (array-valued), by Chebyshev-node interpolation."""
    s = smax * np.cos(np.pi * (np.arange(deg + 1) + 0.5) / (deg + 1))
    V = np.vander(s, deg + 1, increasing=True)
    Y = np.array([fun(si) for si in s])
    return np.linalg.solve(V, Y.reshape(deg + 1, -1)).reshape((deg + 1,) + Y.shape[1:])


def main():
    a = [t for t in sys.argv[1:] if not t.startswith("nstart=")]
    nstart = int(next((t.split("=")[1] for t in sys.argv[1:] if t.startswith("nstart=")), 200))
    L, w, k, j, k1, k2, r1, r2 = (int(x) for x in a[:8])
    seed = 0
    for tok in a[8:]:
        if tok.startswith("s"):
            seed = int(tok[1:])
        elif ":" in tok:
            lay, rr = tok.split(":"); cd.RANKS[int(lay)] = int(rr)
    rng = np.random.default_rng(seed)
    G = rng.standard_normal((w, w))
    _, n = cd.build(np.zeros(100000), L, w, k, j, (k1, k2), (r1, r2), G)
    p0 = rng.standard_normal(n)
    Ws, _ = cd.build(p0, L, w, k, j, (k1, k2), (r1, r2), G)
    x0 = cd.flat(Ws)
    N = x0.size
    g0 = np.max(np.abs(cd.flat(gradH(Ws, G))))

    def gpoly(d):  # coefficients of s -> grad H(theta* + s d)
        return coeffs(lambda s: cd.flat(gradH(cd.unflat(x0 + s * d, L, w), G)), L - 1)

    # order check: lowest nonvanishing order of grad along random directions
    lows = []
    for _ in range(5):
        d = rng.standard_normal(N); d /= np.linalg.norm(d)
        c = gpoly(d)
        nz = [m for m in range(L) if np.max(np.abs(c[m])) > 1e-8 * max(1.0, np.max(np.abs(c)))]
        lows.append(nz[0] + 1 if nz else -1)
    m = min(lows)                       # leading Taylor order of H itself

    kk = slice((k - 1) * w * w, k * w * w)

    def fg(u):
        c = gpoly(u)
        g = c[m - 1]                    # grad H_m(u)
        return float(g @ u) / m, g      # Euler: H_m(u) = <grad H_m(u), u>/m

    # linearity in delta_k: H_m(-x, y) = -H_m(x, y)
    u = rng.standard_normal(N); u /= np.linalg.norm(u)
    f1, _ = fg(u); u2 = u.copy(); u2[kk] *= -1; f2, _ = fg(u2)
    print(f"L={L} w={w} k={k} N={N} |gradH(theta*)|max={g0:.1e} leading order m={m} (per-dir {lows}) "
          f"H_m(u)={f1:.3e} H_m(-x,y)={f2:.3e}", flush=True)

    finals = []
    for t in range(nstart):
        u = rng.standard_normal(N); u /= np.linalg.norm(u)
        eta = 0.05
        f, g = fg(u)
        for it in range(4000):
            pg = g - (g @ u) * u
            if np.linalg.norm(pg) < 1e-7:
                break
            un = u - eta * pg; un /= np.linalg.norm(un)
            fn, gn = fg(un)
            if fn > f:
                eta *= 0.5; continue
            u, f, g = un, fn, gn
            eta = min(eta * 1.2, 1.0)
        pg = g - (g @ u) * u
        finals.append((f, np.linalg.norm(pg), np.linalg.norm(u[kk])))
        if t < 5 or t % 50 == 0:
            print(f"  start {t}: f={f:.4e} |pgrad|={np.linalg.norm(pg):.1e} |x|={np.linalg.norm(u[kk]):.3f} iters={it}", flush=True)
    F = np.array(finals)
    pos = int(np.sum(F[:, 0] > 1e-6))
    print(f"SUMMARY starts={nstart} final f: min={F[:,0].min():.4e} max={F[:,0].max():.4e} "
          f"#f>1e-6={pos} #|f|<=1e-6={int(np.sum(np.abs(F[:,0])<=1e-6))} #f<-1e-6={int(np.sum(F[:,0]<-1e-6))} "
          f"max|pgrad|={F[:,1].max():.1e} |x| range=({F[:,2].min():.3f},{F[:,2].max():.3f})")


if __name__ == "__main__":
    main()
