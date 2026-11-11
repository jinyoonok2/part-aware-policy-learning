# Current protocol and reservations

- `train_test_protocol_v1.json`: Train/Test roles and the fixed 20,000-update recipe.
- `final_evaluation_reservations.json`: houses excluded from future Train collection.
- `official_val_test_house_reservations_v1.json`: retained official Val/Test identities.
- `test_candidate_house_reservations_v1.json`: preserved identities from the earlier candidate plan.

Data collection settings live separately in `data_preparation/configs/`.
The current evaluation uses separate official Val197 and Benchmark400 cohorts.
`data/test_400_plan` is a historical merged-cohort proposal, not an active
finalization workflow. Its house reservations remain excluded from future Train
collection, alongside the independent Benchmark400 reservations.
No reservation or candidate list establishes successful physical grasping.
