# Status reconciliation and chronology

Several artifacts were generated at different times and therefore disagree. The older Q1 readiness audit was created at `2026-07-21T21:13:32.502992+00:00` and recorded `7/12 complete (8 run directories discovered)`. The later development snapshot records 12/12 complete runs, and the filesystem contains 12 main run manifests.

The pipeline state file was last modified at `2026-07-23T07:15:29.847532+09:00` and still reports `RUNNING`. At documentation time, 0 matching live training processes were observed. The bidirectional specialist has 2/3 completed manifests; seed 4307 contains partial training evidence but no completed run manifest.

The correct interpretation is therefore:

1. Main M0–M3 development training is complete for three seeds each.
2. The manuscript and older audit are stale with respect to those main runs.
3. The end-to-end pipeline is not complete.
4. No immutable protocol-freeze manifest or locked neural evaluation set is present.
5. The bidirectional branch is partially complete.
6. Human validation remains unstarted.

All later claims in this monograph use this reconciled chronology.
