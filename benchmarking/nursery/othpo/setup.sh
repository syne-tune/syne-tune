#!/bin/sh

echo "Generating files for SimOpt"
python simopt/generate_simopt_context.py
python simopt/generate_simopt_fixed_factors.py

echo "Generating folder for XGBoost results"
mkdir -p xgboost_experiment_results/random-mnist
