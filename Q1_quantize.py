import glob
import os
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Image helpers
def load_image(path):
    """Load an image as float RGB in [0, 1] with shape (H, W, 3)."""
    img = plt.imread(path)
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    else:
        img = img.astype(np.float32)
        if img.max() > 1.0:
            img /= 255.0
    if img.ndim == 2:  # greyscale -> 3 channels
        img = np.stack([img] * 3, axis=-1)
    return img[..., :3]  # drop alpha if present


def image_to_points(img):
    """(H, W, 3) image -> (H*W, 3) array of RGB points."""
    return img.reshape(-1, 3)


def downsample(img, max_side=80):
    """Block-average an image so that neither side exceeds max_side pixels."""
    H, W, C = img.shape
    f = int(np.ceil(max(H, W) / max_side))
    if f <= 1:
        return img.copy()
    H2, W2 = H // f, W // f
    img = img[: H2 * f, : W2 * f]
    return img.reshape(H2, f, W2, f, C).mean(axis=(1, 3))


def rgbxy_features(img, lam=0.5):
    """(R, G, B, lam*x, lam*y) features with x, y scaled to [0, 1]."""
    H, W, _ = img.shape
    ys, xs = np.mgrid[0:H, 0:W]
    x = xs / max(W - 1, 1)
    y = ys / max(H - 1, 1)
    return np.column_stack([image_to_points(img), lam * x.ravel(), lam * y.ravel()])


# 1. K-Means from scratch
def assign_clusters(X, C, chunk=262144):
    """
    Assignment step. Returns (labels, sqdist) where labels[i] is the index of
    the nearest centroid and sqdist[i] is the squared distance to it.

    Uses ||x - c||^2 = ||x||^2 - 2 x.c + ||c||^2 as one matrix product per chunk,
    so the only Python loop is over chunks of ~260k pixels, never over pixels.
    """
    n = X.shape[0]
    C = np.asarray(C, dtype=np.float64)
    C2 = np.sum(C * C, axis=1)
    labels = np.empty(n, dtype=np.int64)
    sqdist = np.empty(n, dtype=np.float64)
    for s in range(0, n, chunk):
        Xc = X[s : s + chunk].astype(np.float64)
        d = np.sum(Xc * Xc, axis=1)[:, None] - 2.0 * Xc @ C.T + C2[None, :]
        idx = np.argmin(d, axis=1)
        labels[s : s + chunk] = idx
        # exact distance to the chosen centroid (avoids round-off from the expansion)
        sqdist[s : s + chunk] = np.sum((Xc - C[idx]) ** 2, axis=1)
    return labels, sqdist


def update_centroids(X, labels, K):
    """Centroid update: the mean of the points in each cluster (vectorised with bincount)."""
    counts = np.bincount(labels, minlength=K).astype(np.float64)
    sums = np.column_stack(
        [np.bincount(labels, weights=X[:, j], minlength=K) for j in range(X.shape[1])]
    )
    return sums / np.maximum(counts, 1.0)[:, None], counts


def compute_sse(X, labels, C, chunk=262144):
    """Sum of squared distances from each point to its own centroid."""
    total = 0.0
    for s in range(0, X.shape[0], chunk):
        Xc = X[s : s + chunk].astype(np.float64)
        total += np.sum((Xc - C[labels[s : s + chunk]]) ** 2)
    return float(total)


def init_random(X, K, rng):
    """Random initialisation: K distinct pixels chosen uniformly at random."""
    idx = rng.choice(X.shape[0], size=K, replace=False)
    return X[idx].astype(np.float64)


# 2. K-Means++ seeding
def init_kmeans_pp(X, K, rng):
    """
    K-Means++: the first centroid is a random pixel; each further centroid is a
    pixel sampled with probability proportional to D(x)^2, the squared distance
    to the nearest centroid chosen so far.
    """
    n = X.shape[0]
    C = np.empty((K, X.shape[1]), dtype=np.float64)
    C[0] = X[rng.integers(n)]
    d2 = np.sum((X.astype(np.float64) - C[0]) ** 2, axis=1)
    for k in range(1, K):
        total = d2.sum()
        if total <= 0:  # every pixel already coincides with a centroid
            idx = rng.integers(n)
        else:
            r = rng.random() * total
            idx = min(int(np.searchsorted(np.cumsum(d2), r)), n - 1)
        C[k] = X[idx]
        d2 = np.minimum(d2, np.sum((X.astype(np.float64) - C[k]) ** 2, axis=1))
    return C


def kmeans(X, K, init="random", max_iter=50, tol=0.001, seed=0, snapshot_iters=()):
    """
    K-Means with
      - stopping rule: fewer than tol (0.1%) of points change cluster, or max_iter (50) iterations
      - empty clusters re-seeded with the pixel farthest from its centroid

    Returns a dict with keys:
      centroids (K, d), labels (n,), sse_history (list), n_iter (int),
      snapshots {iteration: (centroids, labels)} for the iterations in snapshot_iters
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=np.float32)
    n = X.shape[0]
    C = init_kmeans_pp(X, K, rng) if init == "kmeans++" else init_random(X, K, rng)

    labels_prev = None
    sse_history, snapshots = [], {}
    it = 0
    for it in range(1, max_iter + 1):
        # assignment step
        labels, sqdist = assign_clusters(X, C)

        # empty clusters: re-seed with the pixel(s) farthest from their centroid
        counts = np.bincount(labels, minlength=K)
        empty = np.flatnonzero(counts == 0)
        if empty.size:
            far = np.argpartition(-sqdist, empty.size - 1)[: empty.size]
            labels[far] = empty  # each far pixel becomes the seed of one empty cluster

        # update step
        C, _ = update_centroids(X, labels, K)
        sse_history.append(compute_sse(X, labels, C))
        if it in snapshot_iters:
            snapshots[it] = (C.copy(), labels.copy())

        # stopping rule
        if labels_prev is not None:
            changed = np.count_nonzero(labels != labels_prev) / n
            if changed < tol:
                break
        labels_prev = labels

    return {
        "centroids": C,
        "labels": labels,
        "sse_history": sse_history,
        "n_iter": it,
        "snapshots": snapshots,
    }


def quantize(img, centroids, labels):
    """Replace each pixel by its palette colour."""
    return np.clip(centroids[labels][:, :3].reshape(img.shape), 0, 1)


def compression_ratio(K, n):
    """CR(K) = 24 / (log2 K + 24K / n)."""
    return 24.0 / (np.log2(K) + 24.0 * K / n)


def knee_index(x, y):
    """Index of the 'elbow' point: the point farthest from the chord joining first and last."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xn = (x - x.min()) / (x.max() - x.min() + 1e-12)
    yn = (y - y.min()) / (y.max() - y.min() + 1e-12)
    p1, p2 = np.array([xn[0], yn[0]]), np.array([xn[-1], yn[-1]])
    line = (p2 - p1) / np.linalg.norm(p2 - p1)
    vecs = np.column_stack([xn, yn]) - p1
    proj = np.outer(vecs @ line, line)
    return int(np.argmax(np.linalg.norm(vecs - proj, axis=1)))


# 5. DBSCAN from scratch
CORE, BORDER, NOISE = 0, 1, 2


def pairwise_sqdist_chunk(Xc, X, X2):
    """Squared distances from a chunk of points to all points."""
    d = np.sum(Xc * Xc, axis=1)[:, None] - 2.0 * Xc @ X.T + X2[None, :]
    return np.maximum(d, 0.0)


def region_query_all(X, eps, chunk=512):
    """
    epsilon-neighbourhood of every point (the point counts as its own neighbour),
    computed in chunks so the full n x n distance matrix is never stored.
    """
    X = np.asarray(X, dtype=np.float64)
    X2 = np.sum(X * X, axis=1)
    eps2 = eps * eps
    nbrs = []
    for s in range(0, X.shape[0], chunk):
        within = pairwise_sqdist_chunk(X[s : s + chunk], X, X2) <= eps2 + 1e-12
        nbrs.extend(np.flatnonzero(row) for row in within)
    return nbrs


def dbscan(X, eps, min_pts):
    """
    DBSCAN. Returns (labels, point_type):
      labels[i]     = cluster id (0, 1, ...) or -1 for noise
      point_type[i] = CORE, BORDER or NOISE
    """
    n = X.shape[0]
    nbrs = region_query_all(X, eps)
    counts = np.fromiter((len(a) for a in nbrs), dtype=np.int64, count=n)
    is_core = counts >= min_pts

    labels = np.full(n, -1, dtype=np.int64)
    cid = 0
    for i in range(n):
        if labels[i] != -1 or not is_core[i]:
            continue
        labels[i] = cid
        stack = [i]
        while stack:  # grow the cluster through core points only
            p = stack.pop()
            q = nbrs[p]
            new = q[labels[q] == -1]
            labels[new] = cid
            stack.extend(new[is_core[new]].tolist())
        cid += 1

    point_type = np.full(n, NOISE, dtype=np.int64)
    point_type[labels >= 0] = BORDER
    point_type[is_core] = CORE
    return labels, point_type


def k_distance(X, k, chunk=512):
    """
    Distance from every point to its k-th nearest neighbour, counting the point
    itself (consistent with minPts counting the point itself). Returned sorted ascending.
    """
    X = np.asarray(X, dtype=np.float64)
    X2 = np.sum(X * X, axis=1)
    out = np.empty(X.shape[0])
    for s in range(0, X.shape[0], chunk):
        d = pairwise_sqdist_chunk(X[s : s + chunk], X, X2)
        out[s : s + chunk] = np.sqrt(np.partition(d, k - 1, axis=1)[:, k - 1])
    return np.sort(out)


def choose_eps(X, min_pts, ax=None, title="", percentile=None):
    """
    Choose epsilon from the sorted k-distance curve (and plot it if ax is given).
    Default: automatic elbow (point farthest from the chord). If percentile is given,
    epsilon is read off the curve at that percentile, i.e. where it turns sharply upward
    (used for the toy data, where the bend sits at the very top of the curve).
    """
    kd = k_distance(X, min_pts)
    if percentile is None:
        i = knee_index(np.arange(kd.size), kd)
    else:
        i = min(int(round(percentile / 100 * (kd.size - 1))), kd.size - 1)
    eps = float(kd[i])
    if ax is not None:
        ax.plot(kd, lw=1.5)
        ax.axhline(eps, color="r", ls="--", lw=1, label=f"elbow: ε = {eps:.4f}")
        ax.scatter([i], [eps], color="r", zorder=3)
        ax.set_xlabel("points sorted by k-distance")
        ax.set_ylabel(f"distance to {min_pts}-th nearest neighbour")
        ax.set_title(title)
        ax.legend()
    return eps


def dbscan_image(labels, X_rgb, shape):
    """Colour each DBSCAN cluster by its mean RGB; noise pixels are black."""
    out = np.zeros((labels.size, 3))
    for c in np.unique(labels[labels >= 0]):
        m = labels == c
        out[m] = X_rgb[m].mean(axis=0)
    return out.reshape(shape)


# Experiments
IMAGE_DIR = "images"
OUT_DIR = "results_q1"
K_CHOICE = 16  # chosen in Part 4


def save(fig, name):
    fig.savefig(os.path.join(OUT_DIR, name), dpi=130, bbox_inches="tight")
    plt.close(fig)


def print_table(header, rows):
    print(" | ".join(header))
    for r in rows:
        print(" | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in r))


def show(ax, img, title):
    ax.imshow(np.clip(img, 0, 1), interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def part2_init_comparison(img, name, K=8):
    print(f"\n Part 2: random vs K-Means++ ({name}, K={K}) ")
    X = image_to_points(img)
    rows = []
    for init in ("random", "kmeans++"):
        for seed in range(5):
            r = kmeans(X, K, init=init, seed=seed)
            rows.append([init, seed, r["sse_history"][-1], r["n_iter"]])
    print_table(["Init", "Seed", "Final SSE", "Iterations"], rows)


def part3_convergence(img, name, K=8):
    print(f"\n Part 3: convergence ({name}, K={K}) ")
    X = image_to_points(img)
    r = kmeans(X, K, init="random", seed=0, snapshot_iters=(1, 2, 5))
    sse = np.array(r["sse_history"])
    print(f"Iterations: {r['n_iter']}, final SSE: {sse[-1]:.4f}")
    print("SSE never increases:", bool(np.all(np.diff(sse) <= 1e-9 * sse[0])))

    shots = [(k, *r["snapshots"][k]) for k in (1, 2, 5) if k in r["snapshots"]]
    shots.append((r["n_iter"], r["centroids"], r["labels"]))
    fig, axes = plt.subplots(1, len(shots) + 1, figsize=(3.2 * (len(shots) + 1), 3.2))
    show(axes[0], img, "Original")
    for ax, (k, C, lab) in zip(axes[1:], shots):
        show(ax, quantize(img, C, lab), f"Iteration {k}" + (" (final)" if k == r["n_iter"] else ""))
    save(fig, f"p3_iterations_{name}.png")

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(np.arange(1, len(sse) + 1), sse, "o-")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("SSE")
    ax.set_title(f"SSE vs iteration ({name}, K={K})")
    save(fig, f"p3_sse_{name}.png")


def part4_choose_k(images, Ks=(2, 4, 8, 16, 32)):
    print("\n Part 4: choosing K ")
    fig_e, ax_e = plt.subplots(figsize=(6, 4))
    for name, img in images.items():
        X = image_to_points(img)
        n = X.shape[0]
        rows, sses, quants = [], [], []
        for K in Ks:
            r = kmeans(X, K, init="kmeans++", seed=0)
            sses.append(r["sse_history"][-1])
            rows.append([K, sses[-1], compression_ratio(K, n)])
            quants.append(quantize(img, r["centroids"], r["labels"]))
        print(f"\n{name} (n = {n})")
        print_table(["K", "Final SSE", "CR(K)"], rows)
        ax_e.plot(Ks, sses, "o-", label=name)

        fig, axes = plt.subplots(1, len(Ks) + 1, figsize=(2.8 * (len(Ks) + 1), 3))
        show(axes[0], img, "Original")
        for ax, K, q, row in zip(axes[1:], Ks, quants, rows):
            show(ax, q, f"K={K}, CR={row[2]:.2f}")
        save(fig, f"p4_quantized_{name}.png")
    ax_e.set_xscale("log", base=2)
    ax_e.set_xticks(Ks)
    ax_e.set_xticklabels(Ks)
    ax_e.set_xlabel("K")
    ax_e.set_ylabel("Final SSE")
    ax_e.set_title("Elbow plot")
    ax_e.legend(fontsize=8)
    save(fig_e, "p4_elbow.png")


def part5_dbscan(images, min_pts=6):
    print("\n Part 5: DBSCAN (downsampled, RGB) ")
    rows, eps_used = [], {}
    n_img = len(images)
    fig_k, axes_k = plt.subplots(1, n_img, figsize=(3.8 * n_img, 3.2), squeeze=False)
    fig_s, axes_s = plt.subplots(2, n_img, figsize=(3.2 * n_img, 6.4), squeeze=False)
    for j, (name, img) in enumerate(images.items()):
        small = downsample(img, 80)
        X = image_to_points(small).astype(np.float64)
        eps = choose_eps(X, min_pts, ax=axes_k[0, j], title=f"k-distance: {name}")
        eps_used[name] = eps
        labels, ptype = dbscan(X, eps, min_pts)
        rows.append([name, eps, int(labels.max() + 1), int((ptype == CORE).sum()),
                     int((ptype == BORDER).sum()), int((ptype == NOISE).sum())])
        show(axes_s[0, j], small, f"{name} (downsampled)")
        show(axes_s[1, j], dbscan_image(labels, X, small.shape), "DBSCAN (noise = black)")
    save(fig_k, "p5_kdistance.png")
    save(fig_s, "p5_dbscan.png")
    print_table(["Image", "eps", "Clusters", "Core", "Border", "Noise"], rows)
    return eps_used


def make_rings(rng, n=1000, noise=0.05):
    """Two concentric rings (r = 1 and r = 3), points split in proportion to circumference."""
    n_in = n // 4
    r = np.where(np.arange(n) < n_in, 1.0, 3.0)
    t = np.concatenate([np.linspace(0, 2 * np.pi, n_in, endpoint=False),
                        np.linspace(0, 2 * np.pi, n - n_in, endpoint=False)])
    return np.column_stack([r * np.cos(t), r * np.sin(t)]) + rng.normal(0, noise, (n, 2))


def make_blobs_outliers(rng, n=600, frac_out=0.05):
    """Three round blobs plus 5% uniform random outliers."""
    centers = np.array([[0, 0], [5, 5], [0, 6]])
    n_out = int(round(frac_out * n))
    X_in = centers[rng.integers(0, 3, n - n_out)] + rng.normal(0, 0.6, (n - n_out, 2))
    X_out = rng.uniform(X_in.min(0) - 2, X_in.max(0) + 2, (n_out, 2))
    return np.vstack([X_in, X_out])


def part6_compare(images, eps_used, K_choice=K_CHOICE):
    print("\nPart 6: K-Means vs DBSCAN ")
    rows = []
    rng = np.random.default_rng(0)

    # (a) shapes. eps is read off the k-distance curve where it bends upward.
    datasets = [("Rings", make_rings(rng), 2, 99.5), ("Blobs + outliers", make_blobs_outliers(rng), 3, 95)]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for i, (dname, X, K, pct) in enumerate(datasets):
        t = time.perf_counter()
        km = kmeans(X, K, init="kmeans++", seed=0)
        t_km = time.perf_counter() - t
        eps = choose_eps(X, 4, ax=axes[i, 2], title=f"k-distance: {dname}", percentile=pct)
        t = time.perf_counter()
        db, _ = dbscan(X, eps, 4)
        t_db = time.perf_counter() - t
        axes[i, 0].scatter(X[:, 0], X[:, 1], c=km["labels"], cmap="tab10", s=8)
        axes[i, 0].set_title(f"K-Means (K={K}): {dname}")
        axes[i, 1].scatter(X[db >= 0, 0], X[db >= 0, 1], c=db[db >= 0], cmap="tab10", s=8)
        axes[i, 1].scatter(X[db < 0, 0], X[db < 0, 1], c="k", marker="x", s=14)
        axes[i, 1].set_title(f"DBSCAN (eps={eps:.2f}, minPts=4): {dname}")
        for ax in axes[i, :2]:
            ax.set_aspect("equal")
        rows.append([f"(a) {dname}", "K-Means", f"K={K}", K, 0.0, t_km])
        rows.append([f"(a) {dname}", "DBSCAN", f"eps={eps:.3f}", int(db.max() + 1), 100.0 * (db < 0).mean(), t_db])
    save(fig, "p6a_shapes.png")

    # (b) two images, RGB vs (R, G, B, lambda*x, lambda*y)
    names = list(images)[:2]
    fig, axes = plt.subplots(len(names), 5, figsize=(16, 3.4 * len(names)), squeeze=False)
    for i, name in enumerate(names):
        small = downsample(images[name], 80)
        X_rgb = image_to_points(small).astype(np.float64)
        show(axes[i, 0], small, name)
        for j, (fname, X) in enumerate([("RGB", X_rgb), ("RGBxy", rgbxy_features(small, 0.5))]):
            t = time.perf_counter()
            km = kmeans(X, K_choice, init="kmeans++", seed=0)
            t_km = time.perf_counter() - t
            min_pts = 2 * X.shape[1]
            eps = eps_used[name] if fname == "RGB" else choose_eps(X, min_pts)
            t = time.perf_counter()
            db, _ = dbscan(X, eps, min_pts)
            t_db = time.perf_counter() - t
            show(axes[i, 1 + j], quantize(small, km["centroids"], km["labels"]), f"K-Means {fname}")
            show(axes[i, 3 + j], dbscan_image(db, X_rgb, small.shape), f"DBSCAN {fname}")
            rows.append([f"(b) {name} {fname}", "K-Means", f"K={K_choice}", K_choice, 0.0, t_km])
            rows.append([f"(b) {name} {fname}", "DBSCAN", f"eps={eps:.3f}", int(db.max() + 1),
                         100.0 * (db < 0).mean(), t_db])
    save(fig, "p6b_images.png")

    # (c) sensitivity: eps +-30%, K +-2
    name = names[0]
    X = image_to_points(downsample(images[name], 80)).astype(np.float64)
    for f in (0.7, 1.0, 1.3):
        t = time.perf_counter()
        db, _ = dbscan(X, eps_used[name] * f, 6)
        rows.append([f"(c) {name}", "DBSCAN", f"eps={eps_used[name] * f:.3f}", int(db.max() + 1),
                     100.0 * (db < 0).mean(), time.perf_counter() - t])
    for K in (K_choice - 2, K_choice, K_choice + 2):
        t = time.perf_counter()
        kmeans(X, K, init="kmeans++", seed=0)
        rows.append([f"(c) {name}", "K-Means", f"K={K}", K, 0.0, time.perf_counter() - t])

    print_table(["Experiment", "Method", "Parameters", "Clusters", "Noise %", "Time (s)"], rows)


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.*")))
    images = {os.path.splitext(os.path.basename(p))[0]: load_image(p) for p in paths}
    first = next(iter(images))

    part2_init_comparison(images[first], first)
    for name, img in images.items():
        part3_convergence(img, name)
    part4_choose_k(images)
    eps_used = part5_dbscan(images)
    part6_compare(images, eps_used)