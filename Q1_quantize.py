import argparse
import glob
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def load_image(path):
    """Load an image as normalized RGB."""
    img = plt.imread(path)
    img = np.asarray(img)
    if img.dtype == np.uint8:
        img = img.astype(np.float64) / 255.0
    elif img.dtype == np.uint16:
        img = img.astype(np.float64) / 65535.0
    else:
        img = img.astype(np.float64)
        if img.max() > 1.0:
            img = img / 255.0
    if img.ndim == 2:                       # greyscale -> RGB
        img = np.stack([img] * 3, axis=-1)
    if img.shape[2] == 4:                   # drop alpha
        img = img[:, :, :3]
    return np.clip(img, 0.0, 1.0)


def sq_dists(X, C):
    """Calculate pairwise squared distances.

    X: (n, d), C: (m, d)  ->  (n, m) with ||x_i - c_j||^2.
    Uses ||x||^2 + ||c||^2 - 2 x.c so the work is one matrix product.
    """
    X = np.asarray(X, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    x2 = np.einsum("ij,ij->i", X, X)[:, None]
    c2 = np.einsum("ij,ij->i", C, C)[None, :]
    D = x2 + c2 - 2.0 * (X @ C.T)
    np.maximum(D, 0.0, out=D)               # remove tiny negative round-off
    return D


def downsample(img, max_side=80):
    """Downsample an image so that its longest side is at most max_side.

    Uses block averaging with an integer factor f = ceil(max(H, W) / max_side).
    """
    H, W = img.shape[:2]
    f = int(np.ceil(max(H, W) / max_side))
    if f <= 1:
        return img.copy()
    h, w = H // f, W // f
    c = img.shape[2]
    return img[:h * f, :w * f].reshape(h, f, w, f, c).mean(axis=(1, 3))


def image_features(img, lam=None):
    """Extract RGB and optional spatial features.

    lam=None -> (n, 3) RGB features.
    lam=float -> (n, 5) features (R, G, B, lam*x, lam*y) with x, y in [0, 1].
    """
    H, W = img.shape[:2]
    rgb = img.reshape(-1, 3).astype(np.float64)
    if lam is None:
        return rgb
    yy, xx = np.mgrid[0:H, 0:W]
    x = xx.reshape(-1) / max(W - 1, 1)
    y = yy.reshape(-1) / max(H - 1, 1)
    return np.column_stack([rgb, lam * x, lam * y])


def compression_ratio(k, n_pixels):
    """Calculate compression ratio CR(K) = 24 / (log2 K + 24K / n)."""
    return 24.0 / (np.log2(k) + 24.0 * k / n_pixels)


def _assign(X, C, chunk=1 << 18):
    """Nearest-centroid labels and squared distances, computed in row chunks
    so that the (n, K) distance matrix never has to be held in memory at once."""
    n = len(X)
    labels = np.empty(n, dtype=np.int64)
    d2 = np.empty(n, dtype=np.float64)
    for s in range(0, n, chunk):
        D = sq_dists(X[s:s + chunk], C)
        lab = D.argmin(axis=1)
        labels[s:s + chunk] = lab
        d2[s:s + chunk] = D[np.arange(len(lab)), lab]
    return labels, d2


def quantize_with(img, C):
    """Replace every pixel of img by its nearest colour in palette C."""
    H, W = img.shape[:2]
    X = img.reshape(-1, img.shape[2])
    labels, _ = _assign(X, C)
    return C[labels].reshape(H, W, -1)


def colour_by_labels(img, labels):
    """Paint every pixel with the mean RGB colour of its cluster (label -1 -> black)."""
    H, W = img.shape[:2]
    rgb = img.reshape(-1, 3)
    out = np.zeros_like(rgb)
    valid = labels >= 0
    if valid.any():
        k = labels[valid].max() + 1
        counts = np.bincount(labels[valid], minlength=k).astype(np.float64)
        means = np.stack([np.bincount(labels[valid], weights=rgb[valid, j],
                                      minlength=k) for j in range(3)], axis=1)
        means /= np.maximum(counts, 1)[:, None]
        out[valid] = means[labels[valid]]
    return out.reshape(H, W, 3)


def ari(a, b):
    """Adjusted Rand index between two labelings (used to score the toy data)."""
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    M = np.zeros((ai.max() + 1, bi.max() + 1))
    np.add.at(M, (ai, bi), 1)

    def comb2(x):
        return x * (x - 1) / 2.0

    s_ij = comb2(M).sum()
    s_a = comb2(M.sum(axis=1)).sum()
    s_b = comb2(M.sum(axis=0)).sum()
    total = comb2(len(a))
    expected = s_a * s_b / total
    max_idx = (s_a + s_b) / 2.0
    if max_idx == expected:
        return 1.0
    return float((s_ij - expected) / (max_idx - expected))


def major_clusters(labels, frac=0.01):
    """Number of clusters holding at least `frac` of all points."""
    labels = np.asarray(labels)
    valid = labels[labels >= 0]
    if valid.size == 0:
        return 0
    return int(np.sum(np.bincount(valid) >= frac * len(labels)))


def md_table(headers, rows):
    """Format a list of rows as a Markdown table."""
    def fmt(v):
        if isinstance(v, float):
            return f"{v:.4g}" if abs(v) < 1e5 else f"{v:.3e}"
        return str(v)
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(fmt(v) for v in r) + " |")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# K-Means
# ----------------------------------------------------------------------

class KMeans:
    def __init__(self, k, init="random", max_iter=50, tol=0.001,
                 seed=0, snapshot_iters=()):
        self.k = k
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.rng = np.random.default_rng(seed)
        self.snapshot_iters = set(snapshot_iters)

    def _init_random(self, X):
        """K distinct pixels chosen uniformly at random."""
        idx = self.rng.choice(len(X), size=self.k, replace=False)
        return X[idx].copy()

    def _init_kmeanspp(self, X):
        """K-Means++: first centre uniform, every next centre sampled with
        probability proportional to D(x)^2, the squared distance to the
        closest centre chosen so far."""
        n, d = X.shape
        C = np.empty((self.k, d))
        C[0] = X[self.rng.integers(n)]
        d2 = np.einsum("ij,ij->i", X - C[0], X - C[0])
        for j in range(1, self.k):
            total = d2.sum()
            if total <= 0:                  # all points identical to a centre
                idx = self.rng.integers(n)
            else:
                idx = self.rng.choice(n, p=d2 / total)
            C[j] = X[idx]
            diff = X - C[j]
            np.minimum(d2, np.einsum("ij,ij->i", diff, diff), out=d2)
        return C

    def _update(self, X, labels, C):
        """Centroid update (vectorised with bincount) + empty-cluster re-seeding."""
        n, d = X.shape
        counts = np.bincount(labels, minlength=self.k)
        sums = np.stack([np.bincount(labels, weights=X[:, j], minlength=self.k)
                         for j in range(d)], axis=1)
        newC = C.copy()
        nz = counts > 0
        newC[nz] = sums[nz] / counts[nz, None]
        empty = np.flatnonzero(~nz)
        if empty.size:
            # re-seed each empty cluster with the pixel farthest from its centroid
            diff = X - newC[labels]
            far = np.einsum("ij,ij->i", diff, diff)
            m = empty.size
            idx = np.argpartition(far, -m)[-m:]
            newC[empty] = X[idx]
            self.n_reseeds += m
        return newC

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        n = len(X)
        t0 = time.perf_counter()
        if self.init in ("kmeans++", "k-means++", "kmeanspp", "++"):
            C = self._init_kmeanspp(X)
        else:
            C = self._init_random(X)
        self.init_centroids = C.copy()
        self.sse_history = []
        self.changed_history = []
        self.snapshots = {}
        self.n_reseeds = 0

        labels = None
        sse = np.inf
        it = 0
        for it in range(1, self.max_iter + 1):
            # assignment step
            new_labels, _ = _assign(X, C)
            changed = n if labels is None else int(np.count_nonzero(new_labels != labels))
            labels = new_labels
            # update step
            C = self._update(X, labels, C)
            diff = X - C[labels]
            sse = float(np.einsum("ij,ij->", diff, diff))
            self.sse_history.append(sse)
            self.changed_history.append(changed)
            if it in self.snapshot_iters:
                self.snapshots[it] = C.copy()
            # stopping rule: fewer than tol (0.1%) of pixels changed cluster
            if changed < self.tol * n:
                break

        self.centroids = C
        self.labels_ = labels
        self.n_iter = it
        self.sse = sse
        self.snapshots["final"] = C.copy()
        self.fit_time = time.perf_counter() - t0
        return self

    def predict(self, X):
        labels, _ = _assign(np.asarray(X, dtype=np.float64), self.centroids)
        return labels

    def quantize(self, img):
        """Replace every pixel by its nearest palette colour."""
        return quantize_with(img, self.centroids)


# ----------------------------------------------------------------------
# DBSCAN
# ----------------------------------------------------------------------

CORE, BORDER, NOISE = 0, 1, 2


class DBSCAN:
    def __init__(self, eps, min_pts, chunk=1024):
        self.eps = eps
        self.min_pts = min_pts
        self.chunk = chunk

    def _neighbours(self, X):
        """eps-neighbourhood (including the point itself) of every point.
        Distances are computed in chunks of rows to bound memory."""
        n = len(X)
        eps2 = self.eps ** 2
        out = []
        for s in range(0, n, self.chunk):
            mask = sq_dists(X[s:s + self.chunk], X) <= eps2
            rows, cols = np.nonzero(mask)
            counts = np.bincount(rows, minlength=mask.shape[0])
            out.extend(np.split(cols.astype(np.int64), np.cumsum(counts)[:-1]))
        return out

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        n = len(X)
        t0 = time.perf_counter()
        nbrs = self._neighbours(X)
        counts = np.fromiter((len(a) for a in nbrs), dtype=np.int64, count=n)
        core = counts >= self.min_pts

        labels = np.full(n, -1, dtype=np.int64)
        cid = 0
        for p in np.flatnonzero(core):
            if labels[p] != -1:
                continue
            # grow a new cluster from core point p (only core points expand)
            labels[p] = cid
            stack = [p]
            while stack:
                q = stack.pop()
                nb = nbrs[q]
                new = nb[labels[nb] == -1]
                labels[new] = cid
                stack.extend(new[core[new]].tolist())
            cid += 1

        types = np.full(n, NOISE, dtype=np.int64)
        types[labels >= 0] = BORDER
        types[core] = CORE

        self.labels_ = labels
        self.types_ = types
        self.counts_ = counts
        self.n_clusters_ = cid
        self.noise_frac_ = float(np.mean(labels < 0))
        self.fit_time = time.perf_counter() - t0
        return self


def k_distance(X, k, chunk=1024):
    """Calculate sorted k-nearest-neighbour distances.

    The point itself counts as its own neighbour (same convention as DBSCAN),
    so a point is core for (eps, minPts=k) exactly when its value is <= eps.
    """
    X = np.asarray(X, dtype=np.float64)
    out = np.empty(len(X))
    for s in range(0, len(X), chunk):
        D = sq_dists(X[s:s + chunk], X)
        out[s:s + chunk] = np.partition(D, k - 1, axis=1)[:, k - 1]
    return np.sort(np.sqrt(out))


def elbow_index(y):
    """Find the elbow point of an increasing curve.

    Both axes are scaled to [0, 1]; the elbow is the point that lies farthest
    below the straight line joining the first and the last point.
    """
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    if n < 3 or y[-1] == y[0]:
        return n - 1
    x = np.linspace(0.0, 1.0, n)
    yn = (y - y[0]) / (y[-1] - y[0])
    return int(np.argmax(x - yn))


def choose_eps(X, min_pts, ax=None, title=""):
    """Pick eps at the elbow of the k-distance plot (k = min_pts)."""
    kd = k_distance(X, min_pts)
    i = elbow_index(kd)
    eps = float(kd[i])
    if ax is not None:
        ax.plot(np.arange(len(kd)), kd, lw=1.5)
        ax.axhline(eps, color="tab:red", ls="--", lw=1)
        ax.plot(i, eps, "o", color="tab:red")
        ax.set_xlabel("points sorted by distance")
        ax.set_ylabel(f"distance to {min_pts}-th nearest neighbour")
        ax.set_title(f"{title}  k-distance, eps = {eps:.3f}")
        ax.grid(alpha=0.3)
    return eps


def dbscan_image(img_small, labels):
    """Generate the clustered image: each cluster gets its mean colour,
    noise pixels are black."""
    return colour_by_labels(img_small, np.asarray(labels))


# ----------------------------------------------------------------------
# Toy data
# ----------------------------------------------------------------------

def make_rings(n=800, seed=0):
    """Generate concentric rings (radius 1 and 2.5, equal point density).

    Angles are stratified (one random angle per equal slice of the circle) so
    that the rings have no large random gaps; radial noise has std 0.08.
    """
    rng = np.random.default_rng(seed)
    radii = np.array([1.0, 2.5])
    n1 = int(round(n * radii[0] / radii.sum()))
    sizes = [n1, n - n1]
    X, y = [], []
    for lab, (r, m) in enumerate(zip(radii, sizes)):
        t = (np.arange(m) + rng.uniform(0, 1, m)) * 2 * np.pi / m
        rr = r + rng.normal(0, 0.08, m)
        X.append(np.column_stack([rr * np.cos(t), rr * np.sin(t)]))
        y.append(np.full(m, lab))
    return np.vstack(X), np.concatenate(y)


def make_blobs_outliers(n=600, seed=0):
    """Generate three Gaussian blobs plus 5% uniform outliers (label -1)."""
    rng = np.random.default_rng(seed)
    centres = np.array([[0.0, 4.0], [-4.0, -2.0], [4.0, -2.0]])
    n_out = int(round(0.05 * n))
    n_in = n - n_out
    sizes = [n_in // 3, n_in // 3, n_in - 2 * (n_in // 3)]
    X, y = [], []
    for lab, (c, m) in enumerate(zip(centres, sizes)):
        X.append(c + rng.normal(0, 0.8, (m, 2)))
        y.append(np.full(m, lab))
    X.append(rng.uniform(-10, 10, (n_out, 2)))
    y.append(np.full(n_out, -1))
    return np.vstack(X), np.concatenate(y)


# ----------------------------------------------------------------------
# Experiments
# ----------------------------------------------------------------------

def save(fig, out, name):
    """Save a figure."""
    fig.tight_layout()
    fig.savefig(os.path.join(out, name), dpi=130)
    plt.close(fig)


def _show(ax, img, title):
    ax.imshow(np.clip(img, 0, 1), interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def part_b_init(img, out, k=8, runs=5):
    """Random vs K-Means++ initialisation, `runs` seeds each."""
    X = image_features(img)
    rows = []
    res = {"random": [], "kmeans++": []}
    for init in ("random", "kmeans++"):
        for seed in range(runs):
            km = KMeans(k, init=init, seed=seed).fit(X)
            res[init].append(km)
            rows.append([init, seed, km.sse, km.n_iter, km.fit_time, km.n_reseeds])

    summary = []
    for init in ("random", "kmeans++"):
        s = np.array([m.sse for m in res[init]])
        it = np.array([m.n_iter for m in res[init]])
        summary.append([init, s.mean(), s.std(), s.min(), s.max(), it.mean()])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for init, c in (("random", "tab:orange"), ("kmeans++", "tab:blue")):
        for i, m in enumerate(res[init]):
            axes[0].plot(np.arange(1, m.n_iter + 1), m.sse_history, color=c,
                         alpha=0.8, label=init if i == 0 else None)
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel("SSE")
    axes[0].set_title(f"SSE per iteration, K={k}, {runs} seeds each")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    x = np.arange(runs)
    axes[1].bar(x - 0.2, [m.sse for m in res["random"]], 0.4, label="random",
                color="tab:orange")
    axes[1].bar(x + 0.2, [m.sse for m in res["kmeans++"]], 0.4, label="kmeans++",
                color="tab:blue")
    axes[1].set_xticks(x)
    axes[1].set_xlabel("seed")
    axes[1].set_ylabel("final SSE")
    lo = min(m.sse for v in res.values() for m in v)
    hi = max(m.sse for v in res.values() for m in v)
    axes[1].set_ylim(lo - 0.15 * (hi - lo + 1e-9), hi + 0.15 * (hi - lo + 1e-9))
    axes[1].set_title("Final SSE per run")
    axes[1].legend()
    save(fig, out, "b_init_comparison.png")
    return rows, summary


def part_c_convergence(img, out, name, k=8):
    """Quantized image after iterations 1, 2, 5, final + SSE curve."""
    X = image_features(img)
    km = KMeans(k, init="random", seed=0, snapshot_iters=(1, 2, 5)).fit(X)
    sse = np.array(km.sse_history)
    monotone = bool(np.all(np.diff(sse) <= 1e-9 * sse[0]))

    keys = [it for it in (1, 2, 5) if it in km.snapshots] + ["final"]
    fig = plt.figure(figsize=(3.2 * (len(keys) + 1), 6.4))
    gs = fig.add_gridspec(2, len(keys) + 1)
    _show(fig.add_subplot(gs[0, 0]), img, "original")
    for j, key in enumerate(keys):
        q = quantize_with(img, km.snapshots[key])
        it = km.n_iter if key == "final" else key
        lab = f"final (iter {it})" if key == "final" else f"after iter {it}"
        _show(fig.add_subplot(gs[0, j + 1]), q, f"{lab}\nSSE={sse[it - 1]:.1f}")
    ax = fig.add_subplot(gs[1, :])
    ax.plot(np.arange(1, len(sse) + 1), sse, "o-")
    ax.set_xlabel("iteration")
    ax.set_ylabel("SSE")
    ax.set_title(f"{name}: SSE vs iteration (K={k}, random init) - "
                 f"{'never increases' if monotone else 'INCREASES (bug!)'}")
    ax.grid(alpha=0.3)
    save(fig, out, f"c_convergence_{name}.png")
    return [name, k, km.n_iter, sse[0], sse[-1], "yes" if monotone else "NO"]


def part_d_choose_k(img, out, name, ks=(2, 4, 8, 16, 32)):
    """Elbow plot, compression ratios and quantized images for each K."""
    X = image_features(img)
    n = len(X)
    rows, sses, quants = [], [], []
    for k in ks:
        km = KMeans(k, init="kmeans++", seed=0).fit(X)
        cr = compression_ratio(k, n)
        sses.append(km.sse)
        quants.append(km.quantize(img))
        rows.append([name, k, km.sse, km.n_iter, cr, km.fit_time])

    fig = plt.figure(figsize=(3.0 * (len(ks) + 1), 6.6))
    gs = fig.add_gridspec(2, len(ks) + 1)
    _show(fig.add_subplot(gs[0, 0]), img, "original (24 bit)")
    for j, (k, q, r) in enumerate(zip(ks, quants, rows)):
        _show(fig.add_subplot(gs[0, j + 1]), q, f"K={k}  CR={r[4]:.1f}x")
    ax = fig.add_subplot(gs[1, :(len(ks) + 1) // 2 + 1])
    ax.plot(ks, sses, "o-")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("K")
    ax.set_ylabel("final SSE")
    ax.set_title(f"{name}: elbow plot")
    ax.grid(alpha=0.3)
    ax2 = fig.add_subplot(gs[1, (len(ks) + 1) // 2 + 1:])
    ax2.plot(ks, [r[4] for r in rows], "s-", color="tab:green")
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(ks)
    ax2.set_xticklabels([str(k) for k in ks])
    ax2.set_xlabel("K")
    ax2.set_ylabel("compression ratio")
    ax2.set_title("CR(K) = 24 / (log2 K + 24K/n)")
    ax2.grid(alpha=0.3)
    save(fig, out, f"d_choose_k_{name}.png")
    return rows


def part_e_dbscan(img, out, name, min_pts=6):
    """DBSCAN on the downsampled image with eps from the k-distance elbow."""
    small = downsample(img, 80)
    X = image_features(small)
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
    eps = choose_eps(X, min_pts, ax=axes[0], title=name)
    db = DBSCAN(eps, min_pts).fit(X)
    _show(axes[1], small, f"downsampled {small.shape[1]}x{small.shape[0]}")
    _show(axes[2], dbscan_image(small, db.labels_),
          f"DBSCAN: {db.n_clusters_} clusters, noise (black) {100 * db.noise_frac_:.1f}%")
    type_cols = np.array([[0.15, 0.45, 0.85], [1.0, 0.75, 0.1], [0.0, 0.0, 0.0]])
    _show(axes[3], type_cols[db.types_].reshape(small.shape),
          "point type: core (blue) / border (yellow) / noise (black)")
    save(fig, out, f"e_dbscan_{name}.png")
    n_core = int(np.sum(db.types_ == CORE))
    n_border = int(np.sum(db.types_ == BORDER))
    n_noise = int(np.sum(db.types_ == NOISE))
    return [name, f"{small.shape[1]}x{small.shape[0]}", min_pts, eps,
            db.n_clusters_, major_clusters(db.labels_), n_core, n_border, n_noise,
            100 * db.noise_frac_, db.fit_time]


def part_f_shapes(out):
    """Rings and blobs+outliers: K-Means vs DBSCAN side by side."""
    datasets = [("rings", *make_rings(800, 0), 2),
                ("blobs+5% outliers", *make_blobs_outliers(600, 0), 3)]
    min_pts = 5
    rows = []
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5))
    for i, (dname, X, y, k) in enumerate(datasets):
        km = KMeans(k, init="kmeans++", seed=0).fit(X)
        eps = choose_eps(X, min_pts, ax=axes[i, 2], title=dname)
        db = DBSCAN(eps, min_pts).fit(X)

        axes[i, 0].scatter(X[:, 0], X[:, 1], c=km.labels_, s=7, cmap="tab10",
                           vmin=0, vmax=9)
        axes[i, 0].scatter(km.centroids[:, 0], km.centroids[:, 1], marker="X",
                           s=160, c="k", edgecolor="w")
        axes[i, 0].set_title(f"{dname}: K-Means (K={k}), ARI={ari(y, km.labels_):.2f}")
        lab = db.labels_
        axes[i, 1].scatter(X[lab >= 0, 0], X[lab >= 0, 1], c=lab[lab >= 0], s=7,
                           cmap="tab10", vmin=0, vmax=9)
        axes[i, 1].scatter(X[lab < 0, 0], X[lab < 0, 1], c="k", marker="x", s=18,
                           label="noise")
        axes[i, 1].set_title(f"{dname}: DBSCAN (eps={eps:.2f}, minPts={min_pts}), "
                             f"ARI={ari(y, lab):.2f}")
        axes[i, 1].legend(loc="upper right")
        for a in axes[i, :2]:
            a.set_aspect("equal")

        if (y < 0).any():
            out_recall = float(np.mean(lab[y < 0] < 0))
            # K-Means centroid displacement from true blob centres
            true_c = np.array([X[y == j].mean(0) for j in range(k)])
            shift = sq_dists(true_c, km.centroids).min(axis=1) ** 0.5
            km_note = f"max centroid shift {shift.max():.2f}"
            db_note = f"{100 * out_recall:.0f}% of true outliers flagged as noise"
        else:
            def ring_split(labels):
                # how many distinct clusters each true ring is spread over
                return [int(np.unique(labels[(y == j) & (labels >= 0)]).size)
                        for j in range(k)]
            km_note = (f"clusters per ring {ring_split(km.labels_)}; "
                       "straight cut through both rings")
            db_note = f"clusters per ring {ring_split(lab)}"
        rows.append([f"(a) {dname}", "K-Means", f"K={k}", k, 0.0,
                     ari(y, km.labels_), km.fit_time, km_note])
        rows.append([f"(a) {dname}", "DBSCAN", f"eps={eps:.3f}, minPts={min_pts}",
                     db.n_clusters_, 100 * db.noise_frac_, ari(y, lab),
                     db.fit_time, db_note])
    save(fig, out, "f_shapes.png")
    return rows


def part_f_images(img, out, name, k, lam=0.5):
    """K-Means vs DBSCAN with RGB and RGB+xy features (on the downsampled image)."""
    small = downsample(img, 80)
    min_pts = 6
    rows = []
    fig, axes = plt.subplots(2, 4, figsize=(17, 8.5))
    eps_used = {}
    for i, (fname, flam) in enumerate((("RGB", None), (f"RGB+{lam}xy", lam))):
        X = image_features(small, flam)
        km = KMeans(k, init="kmeans++", seed=0).fit(X)
        eps = choose_eps(X, min_pts, ax=axes[i, 3], title=fname)
        db = DBSCAN(eps, min_pts).fit(X)
        eps_used[fname] = eps
        _show(axes[i, 0], small, f"{name} downsampled")
        _show(axes[i, 1], colour_by_labels(small, km.labels_),
              f"K-Means K={k}, {fname}")
        _show(axes[i, 2], dbscan_image(small, db.labels_),
              f"DBSCAN eps={eps:.3f}, {fname}\n{db.n_clusters_} clusters, "
              f"noise {100 * db.noise_frac_:.1f}%")
        rows.append([f"(b) {name}", "K-Means", f"{fname}, K={k}", k, 0.0, "-",
                     km.fit_time, f"SSE={km.sse:.2f}"])
        rows.append([f"(b) {name}", "DBSCAN", f"{fname}, eps={eps:.3f}",
                     db.n_clusters_, 100 * db.noise_frac_, "-", db.fit_time,
                     f"minPts={min_pts}, {major_clusters(db.labels_)} clusters >=1%"])
    save(fig, out, f"f_images_{name}.png")
    return rows, eps_used


def part_f_sensitivity(img, name, k, eps, min_pts=6):
    """eps +/- 30% for DBSCAN, K +/- 2 for K-Means (RGB, downsampled image)."""
    small = downsample(img, 80)
    X = image_features(small)
    rows = []
    for f, tag in ((0.7, "-30%"), (1.0, "base"), (1.3, "+30%")):
        db = DBSCAN(eps * f, min_pts).fit(X)
        rows.append([f"(c) {name}", "DBSCAN", f"eps={eps * f:.4f} ({tag})",
                     db.n_clusters_, 100 * db.noise_frac_, "-", db.fit_time,
                     f"{major_clusters(db.labels_)} clusters >=1%"])
    for dk in (-2, 0, 2):
        kk = max(1, k + dk)
        km = KMeans(kk, init="kmeans++", seed=0).fit(X)
        n_nonempty = int(np.unique(km.labels_).size)
        rows.append([f"(c) {name}", "K-Means", f"K={kk} ({dk:+d})",
                     n_nonempty, 0.0, "-", km.fit_time, f"SSE={km.sse:.2f}"])
    return rows


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def _find_images(folder):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    if not os.path.isdir(folder):
        # fall back to any folder called "images" below the working directory
        cands = [d for d in glob.glob("**/images", recursive=True)
                 if "__MACOSX" not in d and os.path.isdir(d)]
        if not cands:
            raise FileNotFoundError(f"image folder '{folder}' not found")
        folder = cands[0]
    paths = sorted(p for e in exts for p in glob.glob(os.path.join(folder, e)))
    if not paths:
        raise FileNotFoundError(f"no images in '{folder}'")
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default="images")
    parser.add_argument("--out", default="out/q1")
    parser.add_argument("--k", type=int, default=16)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    paths = _find_images(args.images)
    imgs = {os.path.splitext(os.path.basename(p))[0]: load_image(p) for p in paths}
    names = list(imgs)
    print(f"Loaded {len(imgs)} images: " +
          ", ".join(f"{n} {imgs[n].shape[1]}x{imgs[n].shape[0]}" for n in names))
    report = []

    def emit(title, headers, rows):
        txt = f"## {title}\n\n{md_table(headers, rows)}\n"
        print("\n" + txt)
        report.append(txt)

    # (b) random vs K-Means++ initialisation on one image (sample2 if present)
    b_name = "sample2" if "sample2" in imgs else names[0]
    rows_b, summary_b = part_b_init(imgs[b_name], args.out, k=8, runs=5)
    emit(f"(b) Initialisation, {b_name}, K=8",
         ["init", "seed", "final SSE", "iterations", "time (s)", "empty re-seeds"],
         rows_b)
    emit(f"(b) Initialisation summary, {b_name}",
         ["init", "mean SSE", "std SSE", "min SSE", "max SSE", "mean iterations"],
         summary_b)
    # extra evidence: the same 5+5 runs on every image
    rows_ball = []
    for n in names:
        X = image_features(imgs[n])
        for init in ("random", "kmeans++"):
            ms = [KMeans(8, init=init, seed=s).fit(X) for s in range(5)]
            s = np.array([m.sse for m in ms])
            rows_ball.append([n, init, s.mean(), s.std(), s.min(), s.max(),
                              float(np.mean([m.n_iter for m in ms]))])
    emit("(b) Initialisation on all images (K=8, 5 seeds each)",
         ["image", "init", "mean SSE", "std SSE", "min SSE", "max SSE",
          "mean iterations"], rows_ball)

    # (c) convergence on every image
    rows_c = [part_c_convergence(imgs[n], args.out, n, k=8) for n in names]
    emit("(c) Convergence (K=8, random init)",
         ["image", "K", "iterations", "SSE iter 1", "final SSE", "monotone"], rows_c)

    # (d) choosing K on every image
    rows_d = []
    for n in names:
        rows_d += part_d_choose_k(imgs[n], args.out, n)
    emit("(d) Choosing K (K-Means++ init)",
         ["image", "K", "final SSE", "iterations", "compression ratio", "time (s)"],
         rows_d)

    # (e) DBSCAN on every image
    rows_e = [part_e_dbscan(imgs[n], args.out, n) for n in names]
    eps_rgb = {r[0]: r[3] for r in rows_e}
    emit("(e) DBSCAN on downsampled images (RGB)",
         ["image", "size", "minPts", "eps (elbow)", "clusters", "clusters >=1%",
          "core", "border",
          "noise", "noise %", "time (s)"], rows_e)

    # (f) K-Means vs DBSCAN
    rows_f = part_f_shapes(args.out)
    cmp_names = [n for n in ("sample1", "sample2") if n in imgs] or names[:2]
    for n in cmp_names:
        r, _ = part_f_images(imgs[n], args.out, n, args.k)
        rows_f += r
    for n in cmp_names:
        rows_f += part_f_sensitivity(imgs[n], n, args.k, eps_rgb[n])
    emit("(f) K-Means vs DBSCAN summary",
         ["experiment", "method", "setting", "clusters", "noise %", "ARI",
          "time (s)", "note"], rows_f)

    with open(os.path.join(args.out, "tables.md"), "w", encoding="utf-8") as fh:
        fh.write("# Q1 results\n\n" + "\n".join(report))
    print(f"\nFigures and tables written to {args.out}")


if __name__ == "__main__":
    main()