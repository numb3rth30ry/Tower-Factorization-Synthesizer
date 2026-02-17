import math
import numpy as np

def spectra_report(n: int, tol=1e-8):
    edges, depth, label, parent = build_tower_tree(n)
    N = len(label)

    # Unweighted adjacency
    A = np.zeros((N, N), float)
    # Weighted adjacency (edge weights)
    Aw = np.zeros((N, N), float)

    for u, v, w in edges:
        A[u, v] = A[v, u] = 1.0
        Aw[u, v] = Aw[v, u] = w

    D = np.diag(A.sum(axis=1))
    Dw = np.diag(Aw.sum(axis=1))

    L  = D  - A
    Lw = Dw - Aw

    mask = [i for i in range(N) if i != 0]  # ground the root
    Lg  = L[np.ix_(mask, mask)]
    Lwg = Lw[np.ix_(mask, mask)]

    def spec(M):
        return np.sort(np.linalg.eigvalsh(M))

    return {
        "N": N,
        "deg": tuple(int(x) for x in np.sort(A.sum(axis=1))),
        "spec_A":  spec(A),
        "spec_L":  spec(L),
        "spec_Lw": spec(Lw),
        "spec_Lg": spec(Lg),
        "spec_Lwg": spec(Lwg),
    }

def compare(n1: int, n2: int, tol=1e-7):
    r1 = spectra_report(n1)
    r2 = spectra_report(n2)

    def same(x, y):
        return x.shape == y.shape and np.max(np.abs(x - y)) < tol

    print(f"n1={n1}  N={r1['N']}  deg={r1['deg']}")
    print(f"n2={n2}  N={r2['N']}  deg={r2['deg']}")
    print()

    print("Adjacency cospectral?        ", same(r1["spec_A"],  r2["spec_A"]))
    print("Laplacian cospectral?        ", same(r1["spec_L"],  r2["spec_L"]))
    print("Weighted Laplacian cospectral?", same(r1["spec_Lw"], r2["spec_Lw"]))
    print("Grounded Laplacian cospectral?", same(r1["spec_Lg"], r2["spec_Lg"]))
    print("Grounded weighted cospectral? ", same(r1["spec_Lwg"], r2["spec_Lwg"]))

# Example:
# compare(18480, 112742891520)

# -----------------------------
# 1) Factorization (sympy if available; fallback otherwise)
# -----------------------------
def factorint_fallback(n: int) -> dict[int, int]:
    """Trial division fallback. Fine for small/medium n; for large n use sympy."""
    n = int(n)
    f = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            f[d] = f.get(d, 0) + 1
            n //= d
        d = 3 if d == 2 else d + 2
    if n > 1:
        f[n] = f.get(n, 0) + 1
    return f

def factorint(n: int) -> dict[int, int]:
    try:
        from sympy import factorint as sympy_factorint
        return {int(p): int(e) for p, e in sympy_factorint(int(n)).items()}
    except Exception:
        return factorint_fallback(n)

# -----------------------------
# 2) Build a tower-factorization tree
#    Nodes are occurrences of primes; root is node 0.
#    For each factor p^e: add child "p", and recurse into exponent e.
# -----------------------------
def edge_weight(p: int, exp_multiplicity: int, depth: int) -> float:
    # Default stiffness: multiplicity * log(1+p), with a mild depth discount
    return exp_multiplicity * math.log1p(p) / (1.0 + 0.15 * depth)

def build_tower_tree(n: int):
    """
    Returns:
      edges: list of (u, v, w)
      depth: list where depth[v] is depth in the rooted tree
      label: list where label[v] is the prime label (root label=1)
      parent: list where parent[v] is parent index (root parent=-1)
    """
    edges = []
    depth = [0]
    label = [1]
    parent = [-1]

    def new_node(p_label: int, d: int, par: int) -> int:
        idx = len(label)
        label.append(p_label)
        depth.append(d)
        parent.append(par)
        return idx

    def rec(current: int, m: int, d: int):
        if m <= 1:
            return
        fac = factorint(m)
        for p, e in fac.items():
            child = new_node(p, d + 1, current)
            w = edge_weight(p, e, d)
            edges.append((current, child, w))
            rec(child, e, d + 1)  # recurse on exponent

    rec(0, int(n), 0)
    return edges, depth, label, parent

# -----------------------------
# 3) Laplacian and grounded Laplacian (clamp root)
# -----------------------------
def laplacian_from_edges(num_nodes: int, edges):
    L = np.zeros((num_nodes, num_nodes), dtype=float)
    for u, v, w in edges:
        L[u, u] += w
        L[v, v] += w
        L[u, v] -= w
        L[v, u] -= w
    return L

def grounded_laplacian(L: np.ndarray, ground_index: int = 0) -> np.ndarray:
    mask = [i for i in range(L.shape[0]) if i != ground_index]
    return L[np.ix_(mask, mask)], mask

# -----------------------------
# 4) Map spectrum to audible frequencies
# -----------------------------
def map_to_hz(omega: np.ndarray, fmin=80.0, fmax=4000.0) -> np.ndarray:
    # log-compressed mapping
    om = np.maximum(omega, 1e-12)
    o0, o1 = float(np.min(om)), float(np.max(om))
    if o1 <= o0 * (1.0 + 1e-9):
        return np.full_like(om, (fmin + fmax) / 2.0)
    x = (np.log1p(om) - np.log1p(o0)) / (np.log1p(o1) - np.log1p(o0))
    return fmin + (fmax - fmin) * x

# -----------------------------
# 5) Simple ADSR envelope
# -----------------------------
def adsr(t, attack=0.01, decay=0.15, sustain=0.6, release=0.35, dur=2.5):
    env = np.zeros_like(t)
    a_end = attack
    d_end = attack + decay
    r_start = max(dur - release, d_end)

    # Attack
    a = (t >= 0) & (t < a_end)
    env[a] = t[a] / max(attack, 1e-9)

    # Decay
    d = (t >= a_end) & (t < d_end)
    if np.any(d):
        x = (t[d] - a_end) / max(decay, 1e-9)
        env[d] = 1.0 - x * (1.0 - sustain)

    # Sustain
    s = (t >= d_end) & (t < r_start)
    env[s] = sustain

    # Release
    r = (t >= r_start) & (t <= dur)
    if np.any(r):
        x = (t[r] - r_start) / max(release, 1e-9)
        env[r] = sustain * (1.0 - x)

    return env
#------------------------------
# ?) Patch
#------------------------------
def adjacency_hz(n: int, fmin=80.0, fmax=4000.0):
    edges, depth, label, parent = build_tower_tree(n)
    N = len(label)
    if N <= 1:
        return np.array([220.0], dtype=float)

    A = np.zeros((N, N), float)
    for u, v, w in edges:
        A[u, v] = A[v, u] = 1.0

    mu = np.linalg.eigvalsh(A)
    omega = np.abs(mu)          # nonnegative "graph frequencies"
    omega = omega[omega > 1e-8] # drop near-zero
    if omega.size == 0:
        return np.array([220.0], dtype=float)

    return map_to_hz(np.sort(omega), fmin=fmin, fmax=fmax)


# -----------------------------
# 6) Synthesize one note from n
# -----------------------------
def synth_integer_note(n: int,
                       sr=44100,
                       dur=2.5,
                       fmin=80.0,
                       fmax=4000.0,
                       modes_max=80):
    edges, depth, label, parent = build_tower_tree(n)
    N = len(label)
    if N <= 1:
        # n=1 -> silence
        return np.zeros(int(sr * dur), dtype=np.float32)

    L = laplacian_from_edges(N, edges)
    Lg, mask = grounded_laplacian(L, ground_index=0)

    # --- Replace Laplacian modal section with adjacency-eigs partials ---
    hz = adjacency_hz(n, fmin=fmin, fmax=fmax)

    # limit number of partials
    hz = hz[:modes_max]

    t = np.arange(int(sr * dur)) / sr
    env = adsr(t, dur=dur)

    # simple amplitude law: spectral tilt (higher partials quieter)
    # (purely eigenvalue-based so cospectral -> identical sound)
    k = np.arange(1, len(hz) + 1)
    A = 1.0 / (k ** 1.1)

    alpha = 1.2
    beta  = 0.0008

    y = np.zeros_like(t, dtype=float)
    for ai, fi in zip(A, hz):
        y += ai * np.sin(2*np.pi*fi*t) * np.exp(-t*(alpha + beta*fi))

    y *= env
    y = y / (np.max(np.abs(y)) + 1e-9)
    return y.astype(np.float32)


def write_wav(filename: str, y: np.ndarray, sr=44100):
    from scipy.io.wavfile import write
    y16 = np.int16(np.clip(y, -1.0, 1.0) * 32767)
    write(filename, sr, y16)

if __name__ == "__main__":
    n = 36001  # try swapping n
    y = synth_integer_note(n)
    write_wav(f"tft_{n}.wav", y)
    print("Wrote", f"tft_{n}.wav")
