# Q1 results

## (b) Initialisation, sample2, K=8

| init | seed | final SSE | iterations | time (s) | empty re-seeds |
|---|---|---|---|---|---|
| random | 0 | 9491 | 9 | 2.192 | 7 |
| random | 1 | 8394 | 12 | 2.183 | 11 |
| random | 2 | 5714 | 6 | 0.9054 | 4 |
| random | 3 | 1.426e+04 | 8 | 1.139 | 9 |
| random | 4 | 2680 | 7 | 1.07 | 7 |
| kmeans++ | 0 | 696.9 | 2 | 0.6307 | 0 |
| kmeans++ | 1 | 599.2 | 2 | 0.4983 | 0 |
| kmeans++ | 2 | 585 | 2 | 0.5551 | 0 |
| kmeans++ | 3 | 585 | 2 | 0.6623 | 0 |
| kmeans++ | 4 | 589.3 | 2 | 0.5125 | 0 |

## (b) Initialisation summary, sample2

| init | mean SSE | std SSE | min SSE | max SSE | mean iterations |
|---|---|---|---|---|---|
| random | 8108 | 3875 | 2680 | 1.426e+04 | 8.4 |
| kmeans++ | 611.1 | 43.22 | 585 | 696.9 | 2 |

## (b) Initialisation on all images (K=8, 5 seeds each)

| image | init | mean SSE | std SSE | min SSE | max SSE | mean iterations |
|---|---|---|---|---|---|---|
| sample1 | random | 1112 | 0.2564 | 1112 | 1113 | 42.6 |
| sample1 | kmeans++ | 1171 | 75.9 | 1112 | 1305 | 32.4 |
| sample2 | random | 8108 | 3875 | 2680 | 1.426e+04 | 8.4 |
| sample2 | kmeans++ | 611.1 | 43.22 | 585 | 696.9 | 2 |
| sample3 | random | 317 | 0.2536 | 316.8 | 317.5 | 48.6 |
| sample3 | kmeans++ | 317.5 | 2.653 | 315.4 | 322.7 | 35.8 |
| sample4 | random | 356.8 | 29.01 | 320.8 | 384.3 | 45 |
| sample4 | kmeans++ | 334.2 | 26.83 | 320.5 | 387.8 | 39.8 |
| sample5 | random | 1009 | 26.36 | 967.3 | 1051 | 36.2 |
| sample5 | kmeans++ | 967.3 | 0.01196 | 967.3 | 967.3 | 33 |

## (c) Convergence (K=8, random init)

| image | K | iterations | SSE iter 1 | final SSE | monotone |
|---|---|---|---|---|---|
| sample1 | 8 | 50 | 2072 | 1113 | yes |
| sample2 | 8 | 9 | 6.843e+04 | 9491 | yes |
| sample3 | 8 | 48 | 732.6 | 316.8 | yes |
| sample4 | 8 | 50 | 1520 | 372.1 | yes |
| sample5 | 8 | 49 | 1218 | 1051 | yes |

## (d) Choosing K (K-Means++ init)

| image | K | final SSE | iterations | compression ratio | time (s) |
|---|---|---|---|---|---|
| sample1 | 2 | 6012 | 8 | 23.98 | 0.05029 |
| sample1 | 4 | 2473 | 37 | 11.99 | 0.2611 |
| sample1 | 8 | 1205 | 15 | 7.992 | 0.1591 |
| sample1 | 16 | 621.4 | 40 | 5.991 | 0.8096 |
| sample1 | 32 | 375.4 | 50 | 4.789 | 1.539 |
| sample2 | 2 | 5.375e+04 | 3 | 24 | 0.2828 |
| sample2 | 4 | 1.196e+04 | 6 | 12 | 0.6936 |
| sample2 | 8 | 696.9 | 2 | 7.999 | 0.4951 |
| sample2 | 16 | 249.9 | 6 | 5.999 | 3.401 |
| sample2 | 32 | 87.34 | 8 | 4.799 | 5.337 |
| sample3 | 2 | 1720 | 10 | 23.98 | 0.07217 |
| sample3 | 4 | 662.3 | 17 | 11.99 | 0.1613 |
| sample3 | 8 | 316.8 | 20 | 7.992 | 0.3088 |
| sample3 | 16 | 161.3 | 31 | 5.991 | 0.6071 |
| sample3 | 32 | 85.91 | 50 | 4.789 | 1.494 |
| sample4 | 2 | 3628 | 11 | 23.98 | 0.0542 |
| sample4 | 4 | 835.6 | 12 | 11.99 | 0.08681 |
| sample4 | 8 | 387.8 | 50 | 7.992 | 0.5281 |
| sample4 | 16 | 166.2 | 38 | 5.991 | 0.8469 |
| sample4 | 32 | 90.47 | 50 | 4.789 | 1.529 |
| sample5 | 2 | 3860 | 8 | 23.98 | 0.03104 |
| sample5 | 4 | 2137 | 9 | 11.99 | 0.04725 |
| sample5 | 8 | 967.3 | 46 | 7.99 | 0.3362 |
| sample5 | 16 | 517.3 | 50 | 5.989 | 0.5979 |
| sample5 | 32 | 309.7 | 50 | 4.785 | 1.071 |

## (e) DBSCAN on downsampled images (RGB)

| image | size | minPts | eps (elbow) | clusters | clusters >=1% | core | border | noise | noise % | time (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| sample1 | 64x64 | 6 | 0.04903 | 5 | 1 | 3658 | 138 | 300 | 7.324 | 0.2575 |
| sample2 | 78x59 | 6 | 0.003216 | 20 | 7 | 4345 | 14 | 243 | 5.28 | 0.48 |
| sample3 | 64x64 | 6 | 0.02625 | 7 | 1 | 3887 | 91 | 118 | 2.881 | 0.2446 |
| sample4 | 75x54 | 6 | 0.02561 | 4 | 1 | 3750 | 118 | 182 | 4.494 | 0.2382 |
| sample5 | 75x75 | 6 | 0.04776 | 10 | 1 | 5128 | 300 | 197 | 3.502 | 0.4495 |

## (f) K-Means vs DBSCAN summary

| experiment | method | setting | clusters | noise % | ARI | time (s) | note |
|---|---|---|---|---|---|---|---|
| (a) rings | K-Means | K=2 | 2 | 0 | -0.001025 | 0.001676 | clusters per ring [2, 2]; straight cut through both rings |
| (a) rings | DBSCAN | eps=0.148, minPts=5 | 2 | 1.875 | 0.9607 | 0.01076 | clusters per ring [1, 1] |
| (a) blobs+5% outliers | K-Means | K=3 | 3 | 0 | 0.9232 | 0.001447 | max centroid shift 0.16 |
| (a) blobs+5% outliers | DBSCAN | eps=0.721, minPts=5 | 3 | 6 | 0.9712 | 0.008311 | 93% of true outliers flagged as noise |
| (b) sample1 | K-Means | RGB, K=16 | 16 | 0 | - | 0.02488 | SSE=37.91 |
| (b) sample1 | DBSCAN | RGB, eps=0.049 | 5 | 7.324 | - | 0.2553 | minPts=6, 1 clusters >=1% |
| (b) sample1 | K-Means | RGB+0.5xy, K=16 | 16 | 0 | - | 0.04206 | SSE=114.92 |
| (b) sample1 | DBSCAN | RGB+0.5xy, eps=0.094 | 9 | 5.566 | - | 0.2592 | minPts=6, 1 clusters >=1% |
| (b) sample2 | K-Means | RGB, K=16 | 16 | 0 | - | 0.006007 | SSE=3.86 |
| (b) sample2 | DBSCAN | RGB, eps=0.003 | 20 | 5.28 | - | 0.4547 | minPts=6, 7 clusters >=1% |
| (b) sample2 | K-Means | RGB+0.5xy, K=16 | 16 | 0 | - | 0.01655 | SSE=33.76 |
| (b) sample2 | DBSCAN | RGB+0.5xy, eps=0.014 | 7 | 8.735 | - | 0.3064 | minPts=6, 7 clusters >=1% |
| (c) sample1 | DBSCAN | eps=0.0343 (-30%) | 12 | 15.31 | - | 0.2307 | 3 clusters >=1% |
| (c) sample1 | DBSCAN | eps=0.0490 (base) | 5 | 7.324 | - | 0.241 | 1 clusters >=1% |
| (c) sample1 | DBSCAN | eps=0.0637 (+30%) | 10 | 2.881 | - | 0.2405 | 2 clusters >=1% |
| (c) sample1 | K-Means | K=14 (-2) | 14 | 0 | - | 0.03697 | SSE=41.65 |
| (c) sample1 | K-Means | K=16 (+0) | 16 | 0 | - | 0.02479 | SSE=37.91 |
| (c) sample1 | K-Means | K=18 (+2) | 18 | 0 | - | 0.02358 | SSE=32.04 |
| (c) sample2 | DBSCAN | eps=0.0023 (-30%) | 24 | 5.758 | - | 0.4308 | 7 clusters >=1% |
| (c) sample2 | DBSCAN | eps=0.0032 (base) | 20 | 5.28 | - | 0.432 | 7 clusters >=1% |
| (c) sample2 | DBSCAN | eps=0.0042 (+30%) | 22 | 4.781 | - | 0.4276 | 7 clusters >=1% |
| (c) sample2 | K-Means | K=14 (-2) | 14 | 0 | - | 0.007958 | SSE=4.68 |
| (c) sample2 | K-Means | K=16 (+0) | 16 | 0 | - | 0.005705 | SSE=3.86 |
| (c) sample2 | K-Means | K=18 (+2) | 18 | 0 | - | 0.005977 | SSE=2.62 |
