| dataset | model | threshold_mode | n_series | auc_pr | oracle_best_f1 | precision | recall | f1 | alarm_rate | event_recall | mean_delay_steps |
|---|---|---|---|---|---|---|---|---|---|---|---|
| nab | isolation_forest | adaptive | 16 | 0.186 | 0.282 | 0.269 | 0.202 | 0.173 | 0.115 | 0.885 | 56.944 |
| nab | isolation_forest | fixed | 16 | 0.186 | 0.282 | 0.281 | 0.327 | 0.183 | 0.278 | 0.979 | 62.802 |
| nab | lstm_autoencoder | adaptive | 16 | 0.211 | 0.296 | 0.233 | 0.077 | 0.092 | 0.046 | 1.0 | 41.521 |
| nab | lstm_autoencoder | fixed | 16 | 0.211 | 0.296 | 0.284 | 0.275 | 0.189 | 0.196 | 0.979 | 50.833 |
| smd | isolation_forest | adaptive | 3 | 0.235 | 0.274 | 0.17 | 0.078 | 0.097 | 0.02 | 0.554 | 165.365 |
| smd | isolation_forest | fixed | 3 | 0.235 | 0.274 | 0.247 | 0.376 | 0.22 | 0.223 | 0.747 | 117.2 |
| smd | lstm_autoencoder | adaptive | 3 | 0.376 | 0.437 | 0.243 | 0.136 | 0.145 | 0.017 | 0.917 | 40.385 |
| smd | lstm_autoencoder | fixed | 3 | 0.376 | 0.437 | 0.226 | 0.581 | 0.218 | 0.24 | 1.0 | 2.244 |
