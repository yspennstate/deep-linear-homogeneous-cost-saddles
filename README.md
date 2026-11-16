# Avoiding degenerate saddles of the homogeneous cost of deep linear networks

Yitzchak Shmalo

Jacot, Ged, Şimşek, Hongler and Gabriel (arXiv:2106.15933) describe small-initialization training of a deep linear
network through the gradient flow of the homogeneous cost H(W_1, ..., W_L) = Tr(G^T W_L ... W_1). Their description
needs the flow not to converge to a saddle. The paper shows that every critical point of H is a strict saddle or has
zero Hessian, and that the conserved imbalances between consecutive layers almost surely rule out every limit with two
zero layers, two zero adjacent products, or one of each. This settles depth three for every G of full rank, and depth
four when the hidden layers also have equal width. For larger depth the question reduces to critical points with at
most one zero layer, and an example at depth eight shows that the conserved quantities alone cannot exclude them.

- `paper/main.tex`, `paper/notation.tex`, `paper/main.pdf`: the paper.
- `code/`: the numerical checks of Section 7 and Example 6.3. Python 3 with NumPy; each script runs on its own, for
  example `python3 code/rect_depth4_check.py`. The proofs do not depend on them.
  - `h_flow_test.py`: the flow of H from Gaussian initializations.
  - `rect_depth4_check.py`: the depth-four point left over in the proof of Theorem 5.1, for a rectangular G.
  - `check_bd.py`: Lemma 4.3 and the recursion in the proof of Lemma 4.4.
  - `chain_dcount.py`, `cubic_test.py`, `flow_near.py`: the one-zero-layer critical points of
    Section 6, including the depth-eight example.

The numbers in the paper come from these commands, run from `code/`:

```
python3 check_bd.py
python3 rect_depth4_check.py
python3 h_flow_test.py 4 3 3 3 200 7      # also 4 4 4 2, 5 2 2 4, 5 3 3 3, 6 3 3 3 (L n0 nL w trials seed)
python3 chain_dcount.py 8 4 1 5 5 8 2 2 3:2 4:3 6:2 7:3                 # Example 6.3
python3 flow_near.py 8 4 1 5 5 8 2 2 0.05 12 3:2 4:3 6:2 7:3            # and with 0.005
python3 flow_near.py 8 4 1 5 5 8 2 1 0.05 12 3:2 4:3 6:3 7:3 s1         # point with W_8 W_7 = 0
python3 cubic_test.py 8 4 1 5 5 8 2 2 3:2 4:3 6:2 7:3 nstart=20
```

License: CC BY 4.0 for the paper, MIT for the code.
