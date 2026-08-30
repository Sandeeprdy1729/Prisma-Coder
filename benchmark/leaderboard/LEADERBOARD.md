# PrismaBench Leaderboard

Primary metric: **pass@1** (gate-chain verdict; execution-weighted composite governs ties).
Static-mode rows are ranked separately - they do not use the full V1-V5 gates.

| Rank | Model | Mode | Tasks | pass@1 | 95% CI | pass@5 | CodeBLEU | Latency avg (ms) | Peak mem (MB) |
|---:|---|---|---:|---:|---|---:|---:|---:|---:|
| s1 | submission | static | 26 | 0.000 | [0.000, 0.129] | - | 0.0 | - | - |

_Submission guide: see docs/BENCHMARK_CARD.md - latency, memory and token cost columns are mandatory for full-gate entries (O5 cost honesty)._
