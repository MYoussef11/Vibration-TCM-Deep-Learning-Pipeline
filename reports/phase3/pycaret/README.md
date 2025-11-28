# PyCaret Baseline Results

- Training size fraction: **0.8**
- Session ID: **42**
- Best model: **Pipeline(memory=Memory(location=None),
         steps=[('numerical_imputer',
                 TransformerWrapper(exclude=None,
                                    include=['acceleration_xg_time_mean',
                                             'acceleration_xg_time_std',
                                             'acceleration_xg_time_rms',
                                             'acceleration_xg_time_mad',
                                             'acceleration_xg_time_skewness',
                                             'acceleration_xg_time_kurtosis',
                                             'acceleration_xg_time_crest_factor',
                                             'acceleration_xg_freq_spectral_ene...
                                            criterion='friedman_mse', init=None,
                                            learning_rate=0.1, loss='log_loss',
                                            max_depth=3, max_features=None,
                                            max_leaf_nodes=None,
                                            min_impurity_decrease=0.0,
                                            min_samples_leaf=1,
                                            min_samples_split=2,
                                            min_weight_fraction_leaf=0.0,
                                            n_estimators=100,
                                            n_iter_no_change=None,
                                            random_state=42, subsample=1.0,
                                            tol=0.0001, validation_fraction=0.1,
                                            verbose=0, warm_start=False))],
         verbose=False)**

See `pycaret_comparison.csv` for the full leaderboard and `best_model_report.json` for precision/recall/F1 details.