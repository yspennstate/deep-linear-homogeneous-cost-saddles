"""Numerical checks of Lemma 4.3 (the split of D_l at a zero product) and of the recursion in the proof of Lemma 4.4."""
import numpy as np

rng = np.random.default_rng(7)
w, L = 4, 5

# Submersion recursion: dW_{l+1} = 1/2 W_{l+1}^{-T}(S_l + dW_l W_l^T + W_l dW_l^T) gives dD_l = S_l.
Ws = [rng.standard_normal((w, w)) for _ in range(L)]
S = [(lambda A: A + A.T)(rng.standard_normal((w, w))) for _ in range(L - 1)]
dW = [np.zeros((w, w)) for _ in range(L)]
for l in range(L - 1):  # 0-based: D_l uses Ws[l+1], Ws[l]
    X = S[l] + dW[l] @ Ws[l].T + Ws[l] @ dW[l].T
    dW[l + 1] = 0.5 * np.linalg.solve(Ws[l + 1].T, X)
err = 0.0
for l in range(L - 1):
    dD = dW[l + 1].T @ Ws[l + 1] + Ws[l + 1].T @ dW[l + 1] - dW[l] @ Ws[l].T - Ws[l] @ dW[l].T
    err = max(err, np.max(np.abs(dD - S[l])))
print(f"recursion max|dD - S| = {err:.2e}")

# Lemma 4.3: build W_{l+1} W_l = 0 with both nonzero (rank 2 each, complementary subspaces).
Q, _ = np.linalg.qr(rng.standard_normal((w, w)))
U, V = Q[:, :2], Q[:, 2:]
Wl = U @ rng.standard_normal((2, w))           # range(W_l) = span U
Wl1 = rng.standard_normal((w, 2)) @ V.T          # kernel(W_{l+1}) contains span U
assert np.max(np.abs(Wl1 @ Wl)) < 1e-12
D = Wl1.T @ Wl1 - Wl @ Wl.T
ev, Ev = np.linalg.eigh(D)
Dp = (Ev * np.maximum(ev, 0)) @ Ev.T
Dm = (Ev * np.maximum(-ev, 0)) @ Ev.T
print(f"Lemma 4.3: |P - D+| = {np.max(np.abs(Wl1.T @ Wl1 - Dp)):.2e}  |M - D-| = {np.max(np.abs(Wl @ Wl.T - Dm)):.2e}  "
      f"|W_l+1|^2 - TrD+ = {np.sum(Wl1**2) - np.trace(Dp):.2e}  |W_l|^2 - TrD- = {np.sum(Wl**2) - np.trace(Dm):.2e}")

# Wide middle layer: D_l rank <= n_{l-1} + n_{l+1} < n_l, singular identically.
A = rng.standard_normal((2, 5)); B = rng.standard_normal((5, 2))
print(f"wide middle (2,5,2): rank D = {np.linalg.matrix_rank(A.T @ A - B @ B.T)} of 5")
