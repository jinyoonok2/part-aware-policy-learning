# Two separate evaluation sets

Updated November 8, 2026. The user requested an independent 400-episode benchmark
version alongside the original 197 released demonstrations. This replaces the
earlier proposal to merge the 197 with additions toward a total of 400.

| Cohort | Membership | Expert reference | Current status |
| --- | --- | --- | --- |
| Eval-A: official Val197 | All 197 released door-opening demonstrations | Original actions/states; 187 early-pose labels | Offline pose errors complete; matched rollouts preparing |
| Eval-B: Benchmark400 v1 | 400 fixed starts from the 2,000-start simulator benchmark | Newly generated native-expert records | Starts frozen; asset staging and gated collection queued |

The new set contains zero cases from Eval-A and conservatively excludes its
numeric house IDs as well as prepared Train houses. Selection uses seed 6198,
200 houses and one push/one pull start per house. Membership is fixed before
expert or model outcomes are observed. Parent episode IDs, exact initial states,
hashes and future Train exclusions are recorded.

## Recording and pose labels

The implemented `benchmark_expert_witness_v2` collector records all three RGB
views; expert commands and applied controls; robot qpos/qvel and simulator state;
timestamps; both TCP/base poses; current visible handle points; camera calibration;
expert phases, opening progress and contact/force evidence. Command row zero is
padding; command i produces observation i. The HDF5, videos and EpisodeSpec are
stored together with explicit provenance.

The adapter uses the existing source-phase/closure rule, selects the earliest
eligible grasp before prior opening exceeds 10%, and exports conditioning-base
TCP targets using the same four-frame sampling convention as the trained head.
Source-phase pose targets and force-qualified grasp-onset references stay
separate. A source grasp flag does not certify continuous firm grasp.

Every selected start remains in the task-performance population, including
expert failures and technical unknowns. Pose-label coverage and successful
force-qualified witnesses have their own reported counts. Failed experts are
not replaced by easier successes. A failed expert attempt does not prove that
the task is impossible.

## Gated collection

Execute one full expert episode, then the next 19. A CPU review checks every
video/state/point/camera sequence and the existing pose-loader contract. At least
one usable native early-pose target is required to demonstrate compatibility.
The gate has no learned-policy outcome or expert success-rate threshold. The
remaining 380 run after that review passes. The pilot is included in the 400.

Twelve focused CPU checks passed, including actual synthetic HDF5/MP4 export
through the existing pose loader. Native GPU execution remains pending. Asset
writes for both cohorts are serialized; one expert GPU and one policy-evaluation
GPU may run concurrently after staging.

## Later comparison

Keep the same frozen MolmoBot and step-20,000 pose checkpoint, seed 42, 400-action
budget and fixed opening/retention metrics on both sets. Report baseline versus
pose-guided performance and paired gains within each set. Also report pose-error
label coverage. Differences between sets reflect their task/start distributions;
they do not represent a change in model weights. Model evaluation on the new
400 has not been submitted yet.

Dataset: `data/benchmark_eval400_v1_20261108`.
[Collection jobs and outputs](BENCHMARK400_RUN.md),
[original-set evaluation](../pose_learning/EVALUATION_RUN.md).
