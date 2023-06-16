# Obeying the Order: Introducing Ordered Transfer Hyperparameter Optimisation
This folder contains the code needed to reproduce the results in the above paper. 

The generated files are not included here. To use these, please go to [this fork](https://github.com/sighellan/syne-tune/tree/othpo-results).
By running `bash setup.sh` you can generate the required SimOpt (NewsVendor) setting files, and collect the XGBoost result json file.


## Naming
Different names were used for the methods in the code and in the paper. These are defined in `plotting/plotting_helper.py`, but also summarised here. We also refer to the NewsVendor benchmark in the code as simopt.

| Name in code          | Name in paper         |
|-----------------------|-----------------------|
| RandomSearch          | RandomSearch          |
| BayesianOptimization  | BO                    |
| BoundingBox           | BoundingBox           |
| ZeroShot              | ZeroShot              |
| Quantiles             | CTS                   |
| BoTorchTransfer       | TransferBO            |
| WarmBOShuffled        | SimpleOrderedShuffled |
| WarmBO                | SimpleOrdered         |
| PrevBO                | SimplePrevious        |
| PrevNoBO              | SimplePreviousNoBO    |


## To reproduce everything
All of the steps after setting up the virtual environments are optional. In particular, the raw results are included so you can skip straight to Plot results.

#### Set up virtual environments
* Change folder to `xgboost` folder
* `conda create --name for_xgboost python=3.8`
* `conda activate for_xgboost`
* `pip install -r requirements.txt`
* Change folder to main folder
* `conda create --name for_experiments python=3.8`
* `conda activate for_experiments`
* `pip install -r requirements_locally.txt`
* If running remotely on AWS:
* `conda create --name for_sagemaker python=3.8`
* `conda activate for_sagemaker`
* `pip install -r requirements_for_sagemaker.txt`

#### Generate the SimOpt/NewsVendor files
* `bash setup.sh`

#### Generate the XGBoost hyperparameters
* Change folder to `xgboost` folder
* `python generate_hyperparameter_files.py`

#### [Slow] Collect XGBoost evaluations (see below)
* Change folder to `xgboost` folder

##### Remote
* Activate the `for_sagemaker` virtual environment
* `python launch_xgboost.py`

##### Local
* Activate the `for_xgboost` virtual environment
* `python launch_xgboost.py --local`

##### Post-process
* Change folder to main folder
* `python aggregate_experiment.py`

#### [Slow] Run experiments (see below)
* Change folder to main folder

##### Remote
* Activate the `for_sagemaker` virtual environment
* `python launch_collect_results.py`

##### Local
* Activate the `for_experiments` virtual environment
* `python launch_collect_results.py --local`

##### Post-process
* [MANUAL] Update `experiment_master_file.py` with the files generated through `launch_collect_results.py`

#### [Slow] Generate the YAHPO evaluations for plotting
* `python plotting/collect_yahpo_evaluations_for_plotting.py`

#### Plot results (see below)
* `bash generate_paper_plots.sh`

#### Generate downstream improvement values
* `python calculate_downstream_performance.py` (also included in previous)


## Requirements
We provide four requirement files, depending on how the experiments are run and what part is run.
* `xgboost/requirements.txt`: for collecting XGBoost evaluations
* `requirements_on_sagemaker.txt`: requirements file for the sagemaker instance
* `requirements_for_sagemaker.txt`: local requirements file for running experiments on sagemaker
* `requirements_locally.txt`: local requirements file for running experiments locally


## Seeds
We specify the `random_seed` going into Syne Tune. In our experiments we used the 50 seeds between 0 and 49 (inclusive).


## Running experiments
The experiments are launched using `launch_collect_results.py`, which runs experiments based on `collect_results.py`


### Remotely on AWS
The experiments were run on AWS Sagemaker, this requires setting up credentials, and specifying the
* `role` (str): AWS role, in the form "arn:aws:iam::[UPDATE]:role/service-role/AmazonSageMaker-ExecutionRole-[UPDATE]"
* `profile_name` (str): for the boto3 session
* `alias` (str): used to add identifying label to experiments
* `s3bucket` (str): name of the s3 bucket to use. The bucket should have a non-epty folder datasets, though we don't use that data.


### Locally
Alternatively, the experiments can be run locally, but this will be slow if all the experiments are rerun. To run the experiments locally, run `python launch_collect_results.py --local`.

As a rough estimate, expect each combination of seed, benchmark and optimiser to take about 10 minutes. Then, for 50 seeds it should take about 8h20min for each benchmark and optimiser combination. We ran three of the benchmarks with 10 optimisers and the remaining seven with 5 optimisers, so 65 combinations total. So we'd expect about 541 hours or 22.5 days for the full results.


## Collect XGBoost evaluations
To reproduce collecting the XGBoost evaluations, the `launch_xgboost.py` file should be used. Like for `launch_collect_results.py`, it can be run either remotely or locally.


## Preparing to plot
The experiment results need to be downloaded (if run remotely), and specified in `experiment_master_file.py` in order to enable the plotting.
The folder `optimisation_results` is used to store experiment results. For each experiment, we give the list of experiment result files to use.


## Source files
* `bo_warm_transfer.py` contains the code for StudentBO.
* `blackbox_helper.py` contains most of the source code, used by `collect_results.py`.
* `backend_definitions_dict.py` contains some backend constants.
* `collect_results.py` code to run an experiment.
* `launch_collect_results.py` launch experiments, locally or on AWS.
* `preprocess_results.py`: file to process results from sagemaker runs in preparation of plotting.
* `calculate_downstream_performance.py` used to calculate how well SimpleOrdered does compared to CTS.


### simopt
This folder contains the files for the NewsVendor benchmark
* `generate_simopt_fixed_factors.py`: script for generating fixed settings, which are stored in `generated_files/default_fixed_factors.p`
* `generated_files/default_fixed_factors.p`: settings that stay the same across tasks
* `generate_simopt_context.py`: script for generating varying settings, which are stored in `generated_files/opt-price-random-walk-utility-context-2022-11-08.p`
* `generated_files/opt-price-random-walk-utility-context-2022-11-08.p`: settings that change across tasks
* `simopt_helpers.py`: helper functions for the NewsVendor benchmark
* `SimOptNewsPrice.py`: script called by Syne Tune to evaluate the NewsVendor benchmark


### xgboost
We use the files in the xgboost folder to collect the 1000 hyperparameter evaluations which we use as the basis for our XGBoost benchmark. The folder consists of the following files:
* `XGBoost_helper.py`: helper functions
* `generate_hyperparameter_files.py`: generate a file of the hyperparameters we're interested in
* `hyperparameters_file_random_num-1000_seed-13.p`: the generated hyperparameters
* `XGBoost_script.py`: script for evaluating the hyperparameters
* `requirements.txt`: requirements for AWS Sagemaker for running XGBoost experiments
* `launch_xgboost.py`: launcher script for running `XGBoost_script.py`
* `aggregate_experiment.py`: aggregate the evaluations collected

Run `launch_xgboost.py`from within the xgboost folder.

Note that the XGBoost version used is different for collecting the results and for the experiments (where it is used within the ZeroShot baseline).


## Plotting
The following files are used to generate the plots in the paper.
To generate all the files, please run `bash generate_paper_plots.sh`

* `plotting/plotting_helper.py`: shared plotting functionality.
* `experiment_master_file.py`: grouping of result files and settings needed for plotting.
* `plotting/collect_yahpo_evaluations_for_plotting.py` collect YAHPO evaluations to plot hyperparameter landscapes.


### Paper figures
The paper figures are generated by the following commands:
* Figure 1: `plotting/plot_xgboost_landscapes.py`
* Figures 2-3: drawn without data
* Figure 4: `plotting/plot_yahpo_landscapes.py`, `plotting/plot_simopt_landscapes.py`
* Figure 5: `plotting/plot_normalised_score_bars.py`
* Figure 6: `plotting/plot_compare_simpleordered_cts.py`
* Figure 7: `plotting/plot_different_scenarios.py`

Appendix:
* Figure 8: `plotting/plot_xgboost_landscapes.py`
* Figure 9: `plotting/plot_yahpo_landscapes.py`
* Figure 10: `plotting/plot_yahpo_landscapes.py`
* Figure 11: `plotting/plot_different_scenarios.py`
* Figure 12: `plotting/plot_different_scenarios.py`
* Figure 13: `plotting/plot_rankings.py`
* Figure 14: `plotting/plot_normalised_score_bars.py`
* Figure 15: `plotting/plot_normalised_score_bars.py`
* Figure 16: `plotting/plot_iteration_curves.py`
* Figure 17: `plotting/plot_sampling_locations.py`


## Tests
`test_file.py` contains tests.
