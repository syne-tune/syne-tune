#!/bin/sh

echo "Generating paper plots"

echo "Preprocess results for plotting"
python preprocess_results.py SimOpt
python preprocess_results.py XGBoost
python preprocess_results.py YAHPO_auc_svm_1220
python preprocess_results.py YAHPO_auc_svm_458
python preprocess_results.py YAHPO_auc_aknn_4538
python preprocess_results.py YAHPO_auc_aknn_41138
python preprocess_results.py YAHPO_auc_ranger_4154
python preprocess_results.py YAHPO_auc_ranger_40978
python preprocess_results.py YAHPO_auc_glmnet_375
python preprocess_results.py YAHPO_auc_glmnet_40981

echo "Generate hyperparameter landscape plots"
python plotting/plot_xgboost_landscapes.py
python plotting/plot_simopt_landscapes.py
python plotting/plot_yahpo_landscapes.py

echo "Generate result plots"
python plotting/plot_normalised_score_bars.py
python plotting/plot_iteration_curves.py
python plotting/plot_compare_simpleordered_cts.py
python plotting/plot_different_scenarios.py
python plotting/plot_sampling_locations.py
python plotting/plot_rankings.py

echo "Calculating downstream performance"
python calculate_downstream_performance.py