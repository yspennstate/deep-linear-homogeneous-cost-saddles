"""D-image dimension of a one-zero-layer, zero-Hessian component of the critical set of H = Tr[G^T W_L ... W_1].

Square case: every layer w x w, G w x w invertible.  Layer k is zero.  The cyclic path R that skips W_k is split
at layer j into
  block1 = W_{k+1} .. W_j              (product zero)
  block2 = W_{j+1} .. W_L, G^T, W_1 .. W_{k-1}   (product zero)
Each block is made zero by a "killer" layer whose row space is orthogonal to the range of the product before it.
We compute the Jacobian rank of theta -> (D_1 .. D_{L-1}) along the component; if it is below (L-1) w(w+1)/2,
the event "the limit lies in this component" is null.  We also check that the point is critical with zero Hessian.

Usage: python chain_dcount.py L w k j kill1 kill2 rank_kill1 rank_kill2 [layer:rank ...] [sSEED]
 (1-based layer indices; kill1 in block1, kill2 in block2 and not G)
"""
import sys
import numpy as np


def block_order(L, k, j):
    b1 = list(range(k + 1, j + 1))
    b2 = list(range(j + 1, L + 1)) + ["G"] + list(range(1, k))
    return b1, b2


RANKS = {}


def build(p, L, w, k, j, kills, kranks, G):
    b1, b2 = block_order(L, k, j)
    idx = 0
    W = {}
    # free layers first (all except k and the killers); layer l has rank RANKS.get(l, w)
    for l in range(1, L + 1):
        if l == k or l in kills:
            continue
        r = RANKS.get(l, w)
        A = p[idx: idx + w * r].reshape(w, r); idx += w * r
        B = p[idx: idx + w * r].reshape(w, r); idx += w * r
        W[l] = A @ B.T
    W[k] = np.zeros((w, w))
    for blk, kl, kr in ((b1, kills[0], kranks[0]), (b2, kills[1], kranks[1])):
        pos = blk.index(kl)
        P = np.eye(w)
        for e in blk[:pos]:
            M = G.T if e == "G" else W[e]
            P = M @ P
        u, s, _ = np.linalg.svd(P)
        r = int(np.sum(s > 1e-9 * max(1.0, s[0])))
        U = u[:, :r]
        A = p[idx: idx + w * kr].reshape(w, kr); idx += w * kr
        C = p[idx: idx + w * kr].reshape(w, kr); idx += w * kr
        B = C - U @ (U.T @ C)
        W[kl] = A @ B.T
    return [W[l] for l in range(1, L + 1)], idx


def Dfeat(Ws):
    w = Ws[0].shape[0]
    iu = np.triu_indices(w)
    return np.concatenate([(Ws[l + 1].T @ Ws[l + 1] - Ws[l] @ Ws[l].T)[iu] for l in range(len(Ws) - 1)])


def H(Ws, G):
    P = Ws[0]
    for W in Ws[1:]:
        P = W @ P
    return float(np.sum(G * P))


def flat(Ws):
    return np.concatenate([W.ravel() for W in Ws])


def unflat(x, L, w):
    return [x[i * w * w:(i + 1) * w * w].reshape(w, w) for i in range(L)]


def main():
    a = sys.argv[1:]
    L, w, k, j, k1, k2, r1, r2 = (int(x) for x in a[:8])
    seed = 0
    for tok in a[8:]:
        if tok.startswith("s"):
            seed = int(tok[1:])
        elif ":" in tok:
            lay, rr = tok.split(":"); RANKS[int(lay)] = int(rr)
    rng = np.random.default_rng(seed)
    G = rng.standard_normal((w, w))
    _, n = build(np.zeros(100000), L, w, k, j, (k1, k2), (r1, r2), G)
    p0 = rng.standard_normal(n)
    Ws, _ = build(p0, L, w, k, j, (k1, k2), (r1, r2), G)
    # checks
    x0 = flat(Ws)
    f = lambda x: H(unflat(x, L, w), G)
    h = 1e-4
    N = x0.size
    grad = np.array([(f(x0 + h * e) - f(x0 - h * e)) / (2 * h) for e in np.eye(N)])
    hess_max = 0.0
    for i in range(N):
        ei = np.eye(N)[i] * h
        for q in range(i, N):
            eq = np.eye(N)[q] * h
            v = (f(x0 + ei + eq) - f(x0 + ei - eq) - f(x0 - ei + eq) + f(x0 - ei - eq)) / (4 * h * h)
            hess_max = max(hess_max, abs(v))
    dets = [abs(np.linalg.det(Ws[l + 1].T @ Ws[l + 1] - Ws[l] @ Ws[l].T)) for l in range(L - 1)]
    ranks = [int(np.linalg.matrix_rank(W, tol=1e-9)) for W in Ws]
    adj = [np.linalg.norm(Ws[l + 1] @ Ws[l]) for l in range(L - 1) if l + 1 != k - 1 and l != k - 1]
    adjG = np.linalg.norm(Ws[0] @ G.T @ Ws[-1]) if k not in (1, L) else -1.0
    # Jacobian of D along the component
    hh = 1e-6
    J = np.zeros((Dfeat(Ws).size, n))
    for i in range(n):
        e = np.zeros(n); e[i] = hh
        J[:, i] = (Dfeat(build(p0 + e, L, w, k, j, (k1, k2), (r1, r2), G)[0])
                   - Dfeat(build(p0 - e, L, w, k, j, (k1, k2), (r1, r2), G)[0])) / (2 * hh)
    sv = np.linalg.svd(J, compute_uv=False)
    rk = int(np.sum(sv > 1e-7 * sv[0]))
    print(f"L={L} w={w} k={k} j={j} kill=({k1},{k2}) kr=({r1},{r2}) ranks={ranks} |grad|max={np.max(np.abs(grad)):.1e} "
          f"|hess|max={hess_max:.1e} min|detD|={min(dets):.2e} min|adj|={min(adj):.1e} |W1 G^T WL|={adjG:.1e} dimD={J.shape[0]} rankJ={rk} "
          f"{'NULL' if rk < J.shape[0] else 'FULL'}")


if __name__ == "__main__":
    main()
