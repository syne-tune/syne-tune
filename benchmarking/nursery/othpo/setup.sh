#!/bin/sh

echo "Generating files for SimOpt"
python simopt/generate_simopt_context.py
python simopt/generate_simopt_fixed_factors.py
