| Setup | Runs | Success (all reached) | People reached | Time to reach all (s) | Ghost rescues | Robot-caused collisions | Walk-ins | Path (m) |
|---|---|---|---|---|---|---|---|---|
| rule | 20 | 80% | 75/80 | 92.5 ± 9.8 | 1 | 2 | 8 | 163.0 |
| rule_v2 | 20 | 100% | 80/80 | 100.9 ± 13.5 | 0 | 8 | 13 | 52.9 |
| rule_v3 | 20 | 100% | 80/80 | 104.5 ± 12.5 | 1 | 0 | 14 | 51.5 |

rule    = first version (moving car in the scene)
rule_v2 = car parked (rescue area closed to traffic); 1 s rescue confirmation; same-person radius 1.0 m
rule_v3 = + careful final approach (slow within 3 m, stop at 1.4 m, hard stop if target within 1.0 m)  <- final baseline
All: 20 seeded missions (seed = layout + thermal conditions), scored against true positions, 2.0 m rescue radius.
Collisions: robot within 0.6 m of a person; robot-caused = robot moved >0.2 m toward the person in the last second.
