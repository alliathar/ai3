"""
CS 351 - Assignment 03, Part B Q2
Warehouse robot planning with Value Iteration
(and Policy Iteration for the bonus).

Map characters:
    . = free, # = shelf, G = packing station (+100),
    X = spill zone (-100), S = start

Model (same convention as Part A2):
    Q(s, a) = r + gamma * sum_s' P(s' | s, a) V(s')
    V(G) = +100 and V(X) = -100 are fixed (terminal), r is the living reward
    paid on every move. Moving into a shelf or off the grid leaves the robot
    in place.

Usage:
    python Q2_warehouse.py --map warehouse-1.txt --out out/q2
"""

import argparse
import glob
import os
import time
from collections import deque

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle


ACTIONS = ["U", "D", "L", "R"]
MOVES = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
PERP = {"U": ("L", "R"), "D": ("L", "R"), "L": ("U", "D"), "R": ("U", "D")}
ARROW = {"U": (0, -0.35), "D": (0, 0.35), "L": (-0.35, 0), "R": (0.35, 0)}

VALID = set(".#GXS")


# ----------------------------------------------------------------------
# Map loading
# ----------------------------------------------------------------------

def load_grid(path):
    """Load the warehouse map as a 2-D array of characters.

    Blank lines are ignored; ragged rows are padded with shelves ('#').
    """
    with open(path, "r", encoding="utf-8") as fh:
        lines = [ln.rstrip("\r\n").rstrip() for ln in fh]
    lines = [ln for ln in lines if ln.strip()]
    if not lines:
        raise ValueError(f"{path}: empty map")
    W = max(len(ln) for ln in lines)
    grid = np.array([list(ln.ljust(W, "#")) for ln in lines])
    bad = set(np.unique(grid)) - VALID
    if bad:
        raise ValueError(f"{path}: unknown map symbols {sorted(bad)}")
    if not (grid == "S").any():
        raise ValueError(f"{path}: no start cell 'S'")
    if not (grid == "G").any():
        raise ValueError(f"{path}: no packing station 'G'")
    return grid


# ----------------------------------------------------------------------
# Warehouse environment
# ----------------------------------------------------------------------

class Warehouse:
    def __init__(self, grid, goal_reward=100.0, spill_reward=-100.0,
                 p_intended=0.8):
        self.grid = grid
        self.H, self.W = grid.shape
        self.p = p_intended

        # states = every non-shelf cell (terminals included, their value is fixed)
        self.cells = [(int(r), int(c)) for r, c in np.argwhere(grid != "#")]
        self.nS = len(self.cells)
        self.index = -np.ones((self.H, self.W), dtype=np.int64)
        for i, (r, c) in enumerate(self.cells):
            self.index[r, c] = i

        sym = np.array([grid[r, c] for r, c in self.cells])
        self.is_goal = sym == "G"
        self.is_spill = sym == "X"
        self.terminal = self.is_goal | self.is_spill
        self.term_value = np.zeros(self.nS)
        self.term_value[self.is_goal] = goal_reward
        self.term_value[self.is_spill] = spill_reward
        self.start = int(np.flatnonzero(sym == "S")[0])

        # transitions: next-state index for (action, state, outcome) and outcome probs
        self.T = {False: self._transitions(False), True: self._transitions(True)}

    def _move(self, c, action):
        """Deterministic result of trying `action` from cell c=(row, col)."""
        dr, dc = MOVES[action]
        r, k = c[0] + dr, c[1] + dc
        if 0 <= r < self.H and 0 <= k < self.W and self.grid[r, k] != "#":
            return (r, k)
        return c

    def _transitions(self, stochastic):
        """Return (nxt, probs): nxt[a, s, j] is the state reached by outcome j
        (j=0 intended, j=1,2 the two perpendicular slips) and probs[j] its
        probability. Terminal states map to themselves."""
        nxt = np.empty((len(ACTIONS), self.nS, 3), dtype=np.int64)
        for a, act in enumerate(ACTIONS):
            outcomes = (act,) + PERP[act]
            for s, cell in enumerate(self.cells):
                for j, o in enumerate(outcomes):
                    nxt[a, s, j] = s if self.terminal[s] else self.index[self._move(cell, o)]
        if stochastic:
            slip = (1.0 - self.p) / 2.0
            probs = np.array([self.p, slip, slip])
        else:
            probs = np.array([1.0, 0.0, 0.0])
        return nxt, probs

    def q_values(self, V, stochastic, gamma, r):
        """Q[s, a] = r + gamma * sum_s' P(s'|s,a) V(s'), shape (nS, 4).
        Rows of terminal states hold their fixed value."""
        nxt, probs = self.T[stochastic]
        Q = (r + gamma * (V[nxt] @ probs)).T
        Q[self.terminal] = self.term_value[self.terminal][:, None]
        return Q

    def value_iteration(self, stochastic=False, gamma=0.9, r=-1.0,
                        theta=1e-4, max_iter=100000):
        """Synchronous Value Iteration. Stops when max_s |V_{k+1}(s) - V_k(s)| < theta.
        Returns (V, number of iterations)."""
        V = np.zeros(self.nS)
        V[self.terminal] = self.term_value[self.terminal]
        self.vi_deltas = []
        it = 0
        for it in range(1, max_iter + 1):
            V_new = self.q_values(V, stochastic, gamma, r).max(axis=1)
            delta = float(np.max(np.abs(V_new - V)))
            V = V_new
            self.vi_deltas.append(delta)
            if delta < theta:
                break
        return V, it

    def greedy(self, V, stochastic, gamma, r):
        """Greedy policy (action index per state, -1 for terminals).
        Ties (within 1e-9) go to the first action in ACTIONS order."""
        Q = self.q_values(V, stochastic, gamma, r)
        best = Q.max(axis=1, keepdims=True)
        policy = np.argmax(Q >= best - 1e-9, axis=1)
        policy[self.terminal] = -1
        return policy

    def _policy_matrix(self, policy, stochastic):
        nxt, probs = self.T[stochastic]
        a = np.where(policy >= 0, policy, 0)
        s = np.arange(self.nS)
        P = np.zeros((self.nS, self.nS))
        for j in range(3):
            np.add.at(P, (s, nxt[a, s, j]), probs[j])
        return P

    def policy_evaluation(self, policy, stochastic, gamma, r):
        """Exact evaluation of a fixed policy by solving the linear system
        V_N = r + gamma (P_NN V_N + P_NT V_T) for the non-terminal states N."""
        P = self._policy_matrix(policy, stochastic)
        N = np.flatnonzero(~self.terminal)
        T = np.flatnonzero(self.terminal)
        A = np.eye(len(N)) - gamma * P[np.ix_(N, N)]
        b = r + gamma * P[np.ix_(N, T)] @ self.term_value[T]
        V = np.zeros(self.nS)
        V[T] = self.term_value[T]
        V[N] = np.linalg.solve(A, b)
        return V

    def policy_iteration(self, stochastic=True, gamma=0.9, r=-1.0,
                         max_iter=1000):
        """Policy Iteration starting from 'always Up'.
        Returns (V, policy, number of policy-improvement iterations)."""
        policy = np.zeros(self.nS, dtype=np.int64)
        policy[self.terminal] = -1
        s = np.arange(self.nS)
        it = 0
        V = None
        for it in range(1, max_iter + 1):
            V = self.policy_evaluation(policy, stochastic, gamma, r)
            Q = self.q_values(V, stochastic, gamma, r)
            cur = Q[s, np.where(policy >= 0, policy, 0)]
            best = Q.argmax(axis=1)
            improve = (Q[s, best] > cur + 1e-9) & ~self.terminal
            if not improve.any():
                break
            policy[improve] = best[improve]
        return V, policy, it

    def simulate(self, policy, episodes=1000, r=-1.0,
                 max_steps=500, seed=0):
        """Simulate episodes and return performance statistics.

        Runs the STOCHASTIC world from S. The return of an episode is the
        undiscounted sum r * steps + terminal value (+100 at G, -100 at X).
        Episodes still running after max_steps count as failures.
        """
        rng = np.random.default_rng(seed)
        nxt, probs = self.T[True]
        pos = np.full(episodes, self.start, dtype=np.int64)
        alive = np.ones(episodes, dtype=bool)
        steps = np.zeros(episodes, dtype=np.int64)
        ret = np.zeros(episodes)
        goal = np.zeros(episodes, dtype=bool)
        spill = np.zeros(episodes, dtype=bool)
        for _ in range(max_steps):
            idx = np.flatnonzero(alive)
            if idx.size == 0:
                break
            s = pos[idx]
            a = policy[s]
            j = rng.choice(3, size=idx.size, p=probs)
            s2 = nxt[a, s, j]
            pos[idx] = s2
            steps[idx] += 1
            ret[idx] += r
            done = self.terminal[s2]
            ret[idx[done]] += self.term_value[s2[done]]
            goal[idx[done]] = self.is_goal[s2[done]]
            spill[idx[done]] = self.is_spill[s2[done]]
            alive[idx[done]] = False
        return {
            "avg_return": float(ret.mean()),
            "avg_steps": float(steps.mean()),
            "success_rate": float(goal.mean()),
            "spill_rate": float(spill.mean()),
            "timeout_rate": float(alive.mean()),
            "avg_steps_success": float(steps[goal].mean()) if goal.any() else float("nan"),
        }

    def plot(self, V, policy, title, ax, mark=None):
        """Plot values and policy: heatmap of V, arrows for the policy,
        shelves dark grey, optional outlined cells in `mark`."""
        img = np.full((self.H, self.W), np.nan)
        for i, (r, c) in enumerate(self.cells):
            img[r, c] = V[i]
        cmap = plt.get_cmap("RdYlGn").copy()
        cmap.set_bad("#3a3a3a")
        vmin = min(float(np.nanmin(img)), -1.0)
        vmax = max(float(np.nanmax(img)), 1.0)
        im = ax.imshow(img, cmap=cmap, norm=TwoSlopeNorm(0.0, vmin, vmax))
        small = self.H * self.W <= 150
        for i, (r, c) in enumerate(self.cells):
            ch = self.grid[r, c]
            if self.terminal[i]:
                ax.text(c, r, ch, ha="center", va="center", fontsize=11 if small else 7,
                        fontweight="bold", color="k")
                continue
            act = ACTIONS[policy[i]]
            dx, dy = ARROW[act]
            ax.arrow(c - dx * 0.6, r - dy * 0.6, dx * 1.2, dy * 1.2,
                     head_width=0.2 if small else 0.25, head_length=0.15,
                     length_includes_head=True, color="k", lw=1.0)
            if small:
                ax.text(c, r + 0.38, f"{V[i]:.1f}", ha="center", va="center",
                        fontsize=6.5, color="#222")
            if ch == "S":
                ax.text(c - 0.38, r - 0.3, "S", fontsize=8, fontweight="bold",
                        color="blue", ha="center", va="center")
        if mark is not None:
            for (r, c) in mark:
                ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False,
                                       ec="magenta", lw=2.5))
        ax.set_xticks(range(self.W))
        ax.set_yticks(range(self.H))
        ax.tick_params(labelsize=6)
        ax.set_title(title, fontsize=9)
        return im


# ----------------------------------------------------------------------
# Random warehouse map
# ----------------------------------------------------------------------

def _reachable(grid, src, dst):
    H, W = grid.shape
    seen = np.zeros((H, W), dtype=bool)
    q = deque([src])
    seen[src] = True
    while q:
        r, c = q.popleft()
        if (r, c) == dst:
            return True
        for dr, dc in MOVES.values():
            nr, nc = r + dr, c + dc
            if (0 <= nr < H and 0 <= nc < W and not seen[nr, nc]
                    and grid[nr, nc] not in "#X"):
                seen[nr, nc] = True
                q.append((nr, nc))
    return False


def random_map(n=20, shelf=0.22, spills=12, seed=3):
    """Generate a reachable random warehouse map (S bottom-left, G top-right)."""
    rng = np.random.default_rng(seed)
    S, G = (n - 1, 0), (0, n - 1)
    while True:
        grid = np.full((n, n), ".")
        grid[rng.random((n, n)) < shelf] = "#"
        grid[S] = "S"
        grid[G] = "G"
        free = [tuple(c) for c in np.argwhere(grid == ".")]
        for i in rng.choice(len(free), size=min(spills, len(free)), replace=False):
            grid[free[i]] = "X"
        if _reachable(grid, S, G):
            return grid


# ----------------------------------------------------------------------
# Main experiments
# ----------------------------------------------------------------------

def md_table(headers, rows):
    def fmt(v):
        if isinstance(v, float):
            return f"{v:.4g}"
        return str(v)
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    lines += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in rows]
    return "\n".join(lines)


def _find_map(path):
    if os.path.isfile(path):
        return path
    for pat in ("maps/*.txt", "warehouse*.txt", "*/warehouse*.txt"):
        c = sorted(glob.glob(pat))
        if c:
            return c[0]
    raise FileNotFoundError(f"map file '{path}' not found")


def _save(fig, out, name):
    fig.tight_layout()
    fig.savefig(os.path.join(out, name), dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", default="maps/warehouse.txt")
    parser.add_argument("--out", default="out/q2")
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--episodes", type=int, default=1000)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    report = []

    def emit(title, headers, rows):
        txt = f"## {title}\n\n{md_table(headers, rows)}\n"
        print("\n" + txt)
        report.append(txt)

    # Load the map and initialize the environment.
    map_path = _find_map(args.map)
    grid = load_grid(map_path)
    env = Warehouse(grid)
    g, r0 = args.gamma, -1.0
    print(f"Map {map_path}: {env.H}x{env.W}, {env.nS} non-shelf cells, "
          f"start {env.cells[env.start]}")
    print("\n".join("".join(row) for row in grid))

    def cell_name(i):
        return f"({env.cells[i][0]},{env.cells[i][1]})"

    # Part (a): Deterministic Value Iteration
    t = time.perf_counter()
    V_det, it_det = env.value_iteration(stochastic=False, gamma=g, r=r0)
    t_det = time.perf_counter() - t
    pi_det = env.greedy(V_det, False, g, r0)
    print(f"\n(a) Deterministic VI converged in {it_det} iterations "
          f"({1000 * t_det:.1f} ms), V*(S) = {V_det[env.start]:.3f}")
    fig, ax = plt.subplots(figsize=(6.5, 6))
    im = env.plot(V_det, pi_det, f"Deterministic V* and policy "
                  f"(gamma={g}, r={r0}, {it_det} iterations)", ax)
    fig.colorbar(im, ax=ax, shrink=0.8)
    _save(fig, args.out, "a_deterministic.png")

    # Part (b): Stochastic Value Iteration
    t = time.perf_counter()
    V_sto, it_sto = env.value_iteration(stochastic=True, gamma=g, r=r0)
    t_sto = time.perf_counter() - t
    pi_sto = env.greedy(V_sto, True, g, r0)
    print(f"(b) Stochastic VI converged in {it_sto} iterations "
          f"({1000 * t_sto:.1f} ms), V*(S) = {V_sto[env.start]:.3f}")
    diff = np.flatnonzero((pi_det != pi_sto) & ~env.terminal)
    diff_cells = [env.cells[i] for i in diff]
    Q_det = env.q_values(V_det, False, g, r0)
    Q_sto = env.q_values(V_sto, True, g, r0)
    rows_b = []
    for i in diff:
        rows_b.append([cell_name(i), ACTIONS[pi_det[i]], ACTIONS[pi_sto[i]],
                       V_det[i], V_sto[i],
                       " ".join(f"{a}:{q:.2f}" for a, q in zip(ACTIONS, Q_sto[i]))])
    emit("(b) Cells where the stochastic policy differs from the deterministic one",
         ["cell", "det action", "stoch action", "V det", "V stoch",
          "stochastic Q-values"], rows_b)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    env.plot(V_det, pi_det, f"(a) deterministic, {it_det} iterations", axes[0],
             mark=diff_cells)
    im = env.plot(V_sto, pi_sto, f"(b) stochastic (0.8/0.1/0.1), {it_sto} iterations",
                  axes[1], mark=diff_cells)
    fig.colorbar(im, ax=axes, shrink=0.8)
    fig.suptitle("Magenta boxes: cells where the two policies differ")
    fig.savefig(os.path.join(args.out, "b_stochastic_vs_deterministic.png"), dpi=140)
    plt.close(fig)

    # Part (c): Policy simulation (both policies in the stochastic world)
    rows_c = []
    for pname, pol, V in (("deterministic-world policy", pi_det, None),
                          ("stochastic-world policy", pi_sto, V_sto)):
        st = env.simulate(pol, episodes=args.episodes, r=r0)
        v_true = env.policy_evaluation(pol, True, g, r0)[env.start]
        rows_c.append([pname, st["avg_return"], st["avg_steps"],
                       st["avg_steps_success"], 100 * st["success_rate"],
                       100 * st["spill_rate"], 100 * st["timeout_rate"], v_true])
    emit(f"(c) Simulation in the stochastic world ({args.episodes} episodes, r={r0})",
         ["policy", "avg return", "avg steps", "avg steps (successes)",
          "success %", "spill %", "timeout %", "V^pi(S), gamma=" + str(g)], rows_c)

    # Part (d): Parameter sweep
    gammas = (0.5, 0.9, 0.99)
    rewards = (-5.0, -1.0, 0.0, 1.0)
    rows_d = []
    sweep = {}
    for gg in gammas:
        for rr in rewards:
            V, it = env.value_iteration(stochastic=True, gamma=gg, r=rr)
            pol = env.greedy(V, True, gg, rr)
            st = env.simulate(pol, episodes=args.episodes, r=rr)
            sweep[(gg, rr)] = (V, pol, it, st)
            rows_d.append([gg, rr, it, 100 * st["success_rate"],
                           100 * st["spill_rate"], 100 * st["timeout_rate"],
                           st["avg_steps"], st["avg_return"], V[env.start]])
    emit("(d) Stochastic VI sweep over gamma and living reward r",
         ["gamma", "r", "iterations", "success %", "spill %", "timeout %",
          "avg steps", "avg return", "V*(S)"], rows_d)
    fig, axes = plt.subplots(len(gammas), len(rewards),
                             figsize=(4.2 * len(rewards), 4.0 * len(gammas)))
    for i, gg in enumerate(gammas):
        for j, rr in enumerate(rewards):
            V, pol, it, st = sweep[(gg, rr)]
            env.plot(V, pol, f"gamma={gg}, r={rr:+g}: success "
                     f"{100 * st['success_rate']:.0f}%, {st['avg_steps']:.1f} steps",
                     axes[i, j])
    _save(fig, args.out, "d_sweep_policies.png")

    # Bonus: Policy Iteration vs Value Iteration on a 20x20 map
    big = random_map(20)
    benv = Warehouse(big)
    with open(os.path.join(args.out, "random_map_20x20.txt"), "w") as fh:
        fh.write("\n".join("".join(row) for row in big) + "\n")
    rows_x = []
    res = {}
    for gg in (0.9, 0.99):
        tv, tp = [], []
        for _ in range(5):
            t = time.perf_counter()
            Vv, itv = benv.value_iteration(stochastic=True, gamma=gg, r=r0)
            tv.append(time.perf_counter() - t)
            t = time.perf_counter()
            Vp, pp, itp = benv.policy_iteration(stochastic=True, gamma=gg, r=r0)
            tp.append(time.perf_counter() - t)
        pv = benv.greedy(Vv, True, gg, r0)
        same = int(np.sum((pv == pp) & ~benv.terminal))
        n_nt = int(np.sum(~benv.terminal))
        res[gg] = (Vv, pv, Vp, pp, itv, itp)
        rows_x.append([gg, "Value Iteration", itv, 1000 * np.median(tv), "-"])
        rows_x.append([gg, "Policy Iteration", itp, 1000 * np.median(tp),
                       f"max|V_PI - V_VI| = {np.max(np.abs(Vp - Vv)):.2e}, "
                       f"same action in {same}/{n_nt} states"])
    emit("Bonus: Policy Iteration vs Value Iteration, random 20x20 map "
         "(stochastic, r=-1, median of 5 runs)",
         ["gamma", "method", "iterations", "time (ms)", "agreement"], rows_x)
    Vv, pv, Vp, pp, itv, itp = res[0.9]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.6))
    benv.plot(Vv, pv, f"Value Iteration ({itv} iterations), gamma=0.9", axes[0])
    benv.plot(Vp, pp, f"Policy Iteration ({itp} iterations), gamma=0.9", axes[1])
    _save(fig, args.out, "bonus_pi_vs_vi.png")

    with open(os.path.join(args.out, "tables.md"), "w", encoding="utf-8") as fh:
        fh.write("# Q2 results\n\n"
                 f"(a) deterministic VI: {it_det} iterations, V*(S)={V_det[env.start]:.3f}\n\n"
                 f"(b) stochastic VI: {it_sto} iterations, V*(S)={V_sto[env.start]:.3f}\n\n"
                 + "\n".join(report))
    print(f"\nFigures and tables written to {args.out}")


if __name__ == "__main__":
    main()
