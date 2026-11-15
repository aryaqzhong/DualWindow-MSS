# Oracle reference on the valid tracks (14 tracks)

Mono, 30 s max-vocal excerpts, uSDR of vocals (dB), mixture phase.

| config | N | M | latency (ms) | IRM1 median | IRM2 median | IRM2 mean |
|---|---:|---:|---:|---:|---:|---:|
| sym1024 | 1024 | 1024 | 23.2 | 9.84 | 10.90 | 10.57 |
| sym512 | 512 | 512 | 11.6 | 8.56 | 9.52 | 9.30 |
| sym256 | 256 | 256 | 5.8 | 7.65 | 8.57 | 8.26 |
| sym128 | 128 | 128 | 2.9 | 6.67 | 7.77 | 7.46 |
| dual1024/256 | 1024 | 256 | 5.8 | 8.71 | 9.75 | 9.49 |
| dual2048/256 | 2048 | 256 | 5.8 | 9.04 | 10.06 | 9.80 |
| dual4096/256 | 4096 | 256 | 5.8 | 8.94 | 9.95 | 9.75 |
| dual1024/128 | 1024 | 128 | 2.9 | 8.29 | 9.26 | 8.94 |
| dual2048/128 | 2048 | 128 | 2.9 | 8.58 | 9.50 | 9.22 |

Selected analysis window for C (highest IRM2 median among dual1024/256, dual2048/256, dual4096/256): **dual2048/256** (N = 2048).
