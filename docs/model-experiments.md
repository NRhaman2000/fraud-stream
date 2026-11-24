# Model experiments (Phase 3b)

Every choice below was made on the **validation** split (last 15% of fraudTrain by time).
The test split (fraudTest, Jun to Dec 2020) was evaluated once, after all choices were fixed.

## 1. Feature additions (XGBoost defaults, target precision 0.92)

| Variant | Val PR-AUC | Val recall at P>=0.92 | Missed frauds (val) |
|---|---|---|---|
| Baseline, 16 features | 0.9724 | 0.928 | 82 |
| + amount_sum_1h | 0.9738 | 0.930 | 79 |
| + tx_count_7d | 0.9733 | 0.928 | 82 |
| + amount_sum_7d | 0.9741 | 0.931 | 78 |
| + max_amount_24h | 0.9788 | 0.950 | 57 |
| + distinct_categories_24h | 0.9728 | 0.929 | 80 |
| + amount_to_card_mean | 0.9741 | 0.925 | 85 |
| + amount_to_7d_mean | 0.9748 | 0.935 | 74 |
| + hour_deviation | 0.9783 | 0.951 | 56 |
| **+ all 8 (kept)** | **0.9871** | **0.971** | **33** |

Each feature helps a little alone; together they cut missed frauds by 60%.

## 2. Hyperparameters (all 24 features)

| max_depth | min_child_weight | Val PR-AUC |
|---|---|---|
| 6 | 5 (default, kept) | 0.9871 |
| 6 | 1 | 0.9873 |
| 8 | 5 | 0.9864 |
| 8 | 1 | 0.9868 |
| 4 | 5 | 0.9860 |

All within noise, so the defaults stay.

## 3. Alert threshold target (validation)

| Val precision target | Best val recall |
|---|---|
| 0.92 | 0.969 |
| 0.95 (kept) | 0.961 |
| 0.97 | 0.948 |

Raising the target from 0.92 to 0.95 costs under 1 point of recall, and leaves a margin for
the precision drop seen from validation to test in v1 (0.92 -> 0.90). Chosen: 0.95.

## 4. Final, single test-set evaluation

| Model | Test precision (95% CI) | Test recall (95% CI) | Test PR-AUC |
|---|---|---|---|
| v1: 16 features, target 0.92 | 0.902 (0.889-0.915) | 0.884 (0.870-0.898) | 0.952 |
| **v2: 24 features, target 0.95** | **0.940 (0.929-0.950)** | **0.931 (0.920-0.942)** | **0.976** |

Test: 555,719 transactions, 2,145 frauds. Results vary by about +-0.005 between machines
(XGBoost multithreading on different CPUs); rerun `make train` to reproduce on yours.
