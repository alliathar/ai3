"""
CS 351 - Assignment 03, Part B Q1
Starter skeleton: Colour quantization with K-Means and DBSCAN.

Complete the TODO sections according to the assignment instructions.
Only NumPy and Matplotlib should be used for the algorithms.
"""

import argparse
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
    # TODO
    raise NotImplementedError


def sq_dists(X, C):
    """Calculate pairwise squared distances."""
    # TODO
    raise NotImplementedError


def downsample(img, max_side=80):
    """Downsample an image."""
    # TODO
    raise NotImplementedError


def image_features(img, lam=None):
    """Extract RGB and optional spatial features."""
    # TODO
    raise NotImplementedError


def compression_ratio(k, n_pixels):
    """Calculate compression ratio."""
    # TODO
    raise NotImplementedError


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
        # TODO
        raise NotImplementedError

    def _init_kmeanspp(self, X):
        # TODO
        raise NotImplementedError

    def fit(self, X):
        # TODO
        raise NotImplementedError

    def predict(self, X):
        # TODO
        raise NotImplementedError

    def quantize(self, img):
        # TODO
        raise NotImplementedError


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
        # TODO
        raise NotImplementedError

    def fit(self, X):
        # TODO
        raise NotImplementedError


def k_distance(X, k, chunk=1024):
    """Calculate sorted k-nearest-neighbour distances."""
    # TODO
    raise NotImplementedError


def elbow_index(y):
    """Find the elbow point."""
    # TODO
    raise NotImplementedError


def choose_eps(X, min_pts, ax=None, title=""):
    # TODO
    raise NotImplementedError


def dbscan_image(img_small, labels):
    """Generate the clustered image."""
    # TODO
    raise NotImplementedError


# ----------------------------------------------------------------------
# Toy data
# ----------------------------------------------------------------------

def make_rings(n=800, seed=0):
    """Generate concentric rings."""
    # TODO
    raise NotImplementedError


def make_blobs_outliers(n=600, seed=0):
    """Generate Gaussian blobs with outliers."""
    # TODO
    raise NotImplementedError


# ----------------------------------------------------------------------
# Experiments
# ----------------------------------------------------------------------

def save(fig, out, name):
    """Save a figure."""
    fig.tight_layout()
    fig.savefig(os.path.join(out, name), dpi=130)
    plt.close(fig)


def part_b_init(img, out, k=8, runs=5):
    # TODO
    raise NotImplementedError


def part_c_convergence(img, out, name, k=8):
    # TODO
    raise NotImplementedError


def part_d_choose_k(img, out, name, ks=(2, 4, 8, 16, 32)):
    # TODO
    raise NotImplementedError


def part_e_dbscan(img, out, name, min_pts=6):
    # TODO
    raise NotImplementedError


def part_f_shapes(out):
    # TODO
    raise NotImplementedError


def part_f_images(img, out, name, k, lam=0.5):
    # TODO
    raise NotImplementedError


def part_f_sensitivity(img, name, k, eps, min_pts=6):
    # TODO
    raise NotImplementedError


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default="images")
    parser.add_argument("--out", default="out/q1")
    parser.add_argument("--k", type=int, default=16)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # TODO
    raise NotImplementedError


if __name__ == "__main__":
    main()