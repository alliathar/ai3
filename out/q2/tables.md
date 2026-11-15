# Q2 results

(a) deterministic VI: 17 iterations, V*(S)=32.616

(b) stochastic VI: 38 iterations, V*(S)=4.441

## (b) Cells where the stochastic policy differs from the deterministic one

| cell | det action | stoch action | V det | V stoch | stochastic Q-values |
|---|---|---|---|---|---|
| (0,0) | D | R | 10.38 | 3.792 | U:2.58 D:3.18 L:2.49 R:3.79 |
| (0,3) | D | R | 17.96 | 10.24 | U:8.22 D:9.63 L:6.61 R:10.24 |
| (0,6) | D | R | 28.35 | 19.58 | U:16.62 D:19.25 L:14.35 R:19.58 |
| (4,0) | D | R | 21.07 | 11.62 | U:7.89 D:7.89 L:8.98 R:11.62 |
| (5,0) | D | U | 24.52 | 8.982 | U:8.98 D:5.42 L:7.11 R:7.11 |
| (6,0) | D | U | 28.35 | 6.667 | U:6.67 D:3.40 L:5.01 R:5.01 |
| (7,0) | R | U | 32.62 | 4.441 | U:4.44 D:2.84 L:3.20 R:1.93 |
| (7,1) | R | L | 37.35 | 2.68 | U:1.64 D:1.64 L:2.68 R:2.00 |

## (c) Simulation in the stochastic world (1000 episodes, r=-1.0)

| policy | avg return | avg steps | avg steps (successes) | success % | spill % | timeout % | V^pi(S), gamma=0.9 |
|---|---|---|---|---|---|---|---|
| deterministic-world policy | 25.78 | 9.424 | 11.04 | 67.6 | 32.4 | 0 | 0.2651 |
| stochastic-world policy | 80.28 | 19.72 | 19.72 | 100 | 0 | 0 | 4.441 |

## (d) Stochastic VI sweep over gamma and living reward r

| gamma | r | iterations | success % | spill % | timeout % | avg steps | avg return | V*(S) |
|---|---|---|---|---|---|---|---|---|
| 0.5 | -5 | 17 | 100 | 0 | 0 | 64.06 | -220.3 | -9.999 |
| 0.5 | -1 | 18 | 100 | 0 | 0 | 64.06 | 35.94 | -1.999 |
| 0.5 | 0 | 18 | 100 | 0 | 0 | 64.06 | 100 | 0.0004889 |
| 0.5 | 1 | 18 | 100 | 0 | 0 | 64.06 | 164.1 | 2 |
| 0.9 | -5 | 37 | 72.1 | 27.9 | 0 | 10.42 | -7.885 | -24.36 |
| 0.9 | -1 | 38 | 100 | 0 | 0 | 19.72 | 80.28 | 4.441 |
| 0.9 | 0 | 49 | 100 | 0 | 0 | 19.72 | 100 | 13.13 |
| 0.9 | 1 | 51 | 100 | 0 | 0 | 19.72 | 119.7 | 21.82 |
| 0.99 | -5 | 42 | 96 | 4 | 0 | 18.26 | 0.71 | -6.396 |
| 0.99 | -1 | 104 | 100 | 0 | 0 | 19.72 | 80.28 | 64.17 |
| 0.99 | 0 | 162 | 100 | 0 | 0 | 19.72 | 100 | 82.08 |
| 0.99 | 1 | 168 | 0 | 0 | 100 | 500 | 500 | 100 |

## Bonus: Policy Iteration vs Value Iteration, random 20x20 map (stochastic, r=-1, median of 5 runs)

| gamma | method | iterations | time (ms) | agreement |
|---|---|---|---|---|
| 0.9 | Value Iteration | 89 | 1.806 | - |
| 0.9 | Policy Iteration | 9 | 359.8 | max|V_PI - V_VI| = 8.46e-04, same action in 303/303 states |
| 0.99 | Value Iteration | 918 | 17.42 | - |
| 0.99 | Policy Iteration | 9 | 189.5 | max|V_PI - V_VI| = 9.84e-03, same action in 302/303 states |
