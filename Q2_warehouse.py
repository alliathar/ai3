"""
CS 351 - Assignment 03, Part B Q2
Starter skeleton: Warehouse robot planning with Value Iteration
(and Policy Iteration for the bonus).

Map characters:
    . = free, # = shelf, G = packing station (+100),
    X = spill zone (-100), S = start

Complete the TODO sections according to the assignment instructions.
"""

import argparse
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


ACTIONS = ["U", "D", "L", "R"]
MOVES = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
PERP = {"U": ("L", "R"), "D": ("L", "R"), "L": ("U", "D"), "R": ("U", "D")}
ARROW = {"U": (0, -0.35), "D": (0, 0.35), "L": (-0.35, 0), "R": (0.35, 0)}


# ----------------------------------------------------------------------
# Map loading
# ----------------------------------------------------------------------

def load_grid(path):
    """Load the warehouse map."""
    # TODO
    raise NotImplementedError


# ----------------------------------------------------------------------
# Warehouse environment
# ----------------------------------------------------------------------

class Warehouse:
    def __init__(self, grid, goal_reward=100.0, spill_reward=-100.0,
                 p_intended=0.8):
        self.grid = grid
        self.H, self.W = grid.shape
        self.p = p_intended

        # TODO: Initialize states, terminals, and transitions.
        raise NotImplementedError

    def _move(self, c, action):
        # TODO
        raise NotImplementedError

    def _transitions(self, stochastic):
        # TODO
        raise NotImplementedError

    def q_values(self, V, stochastic, gamma, r):
        # TODO
        raise NotImplementedError

    def value_iteration(self, stochastic=False, gamma=0.9, r=-1.0,
                        theta=1e-4, max_iter=100000):
        # TODO
        raise NotImplementedError

    def greedy(self, V, stochastic, gamma, r):
        # TODO
        raise NotImplementedError

    def policy_evaluation(self, policy, stochastic, gamma, r):
        # TODO
        raise NotImplementedError

    def policy_iteration(self, stochastic=True, gamma=0.9, r=-1.0,
                         max_iter=1000):
        # TODO
        raise NotImplementedError

    def simulate(self, policy, episodes=1000, r=-1.0,
                 max_steps=500, seed=0):
        """Simulate episodes and return performance statistics."""
        # TODO
        raise NotImplementedError

    def plot(self, V, policy, title, ax, mark=None):
        """Plot values and policy."""
        # TODO
        raise NotImplementedError


# ----------------------------------------------------------------------
# Random warehouse map
# ----------------------------------------------------------------------

def random_map(n=20, shelf=0.22, spills=12, seed=3):
    """Generate a reachable random warehouse map."""
    # TODO
    raise NotImplementedError


# ----------------------------------------------------------------------
# Main experiments
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", default="maps/warehouse.txt")
    parser.add_argument("--out", default="out/q2")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # TODO: Load the map and initialize the environment.

    # Part (a): Deterministic Value Iteration
    # TODO

    # Part (b): Stochastic Value Iteration
    # TODO

    # Part (c): Policy simulation
    # TODO

    # Part (d): Parameter sweep
    # TODO

    # Bonus: Policy Iteration vs Value Iteration
    # TODO

    raise NotImplementedError


if __name__ == "__main__":
    main()