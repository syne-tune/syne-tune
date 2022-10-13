
# Change Log
All notable changes to this project will be documented in this file.
 
The format is based on [Keep a Changelog](http://keepachangelog.com/).


## [0.3.1] - 2022-10-??

We release 0.3.1 version which you can install with `pip install syne-tune[extra]`.

Thanks to all contributors (sorted by chronological commit order):
@mseeger, @geoalgo, @aaronkl, @wistuba, @talesa, @hfurkanbozkurt, @rsnirwan,
@duck105, @ondrejbohdal, @iaroslav-ai, @austinmw, @banyikun, @jjaeyeon

### Added
* New tutorial: How to Contribute a New Scheduler
* New tutorial: Multi-Fidelity Hyperparameter Optimization
* YAHPO benchmarks integrated into blackbox repository
* PD1 benchmark integrated into blackbox repository
* New HPO algorithm: Hyper-Tune
* New HPO algorithm: Differential Evolution Hyperband (DEHB)
* New experimental HPO algorithm: Neuralband
* New HPO algorithm: Grid search (categorical variables only)
* BOTorch searcher
* MOBSTER algorithm supports independent GPs at each rung level
* Support for launching benchmarks in benchmarking/commons
* New benchmark: Fine-tuning Hugging Face transformers
* Add IPython util function to display results as parallel categories plot
* New hyperparameter types `ordinal`, `logordinal`
* Support no checkpointing in BlackboxRepositoryBackend
* Plateau rule as StoppingCriterion
* Automate PyPI releases: python-publish.yml
* Add license hook

### Changed
* Replace PyTorch MLP by sklearn in BORE (better performance)
* AWS dependencies moved out of core into `aws`
* New dependencies `yahpo`
 
### Fixed
* In SageMaker back-end, trials with low IDs received reports several times
* Fixing issue with checkpoint_s3_uri usage
* Fix mode in BOTorch searcher when maximizing
* Avoid experiment abort due to throttling of SageMaker job launching
* Surrogate model for lcbench defaults to 1-NN now
* Fix conditional imports, so Syne Tune can be run with reduced dependencies
* Fix lcbench blackbox (ignore first and last fidelity)
* Fix bug in BlackboxSimulatorBackend for pause/resume scheduling (issue #304)
* Revert wait_trial_completion_when_stopping to False
* Terminate with error when tuning sees an exception
* Docker Building Fixed by Adding Line Breaks At End of Requirements Files 
* Control Decision for Running Trials When Stopping Criterion is Met
* Fix mode MSR and HB+BB


## [0.3.0] - 2022-05-07

We release 0.3.0 version which you can install with `pip install syne-tune[extra]==0.3.0`, thanks to all contributors!
(sorted by chronological commit order :-))
@mseeger, @geoalgo, @iaroslav-ai, @aaronkl, @wistuba, @rsnirwan, @hfurkanbozkurt, @ondrejbohdal, @ltiao, @lostella, @jjaeyeon
 
### Added
* Tensorboard visualization 
* Option to change the path where experiments results are written
* Revamp of README.md, addition of an FAQ
* Code formatting with Black
* Scripts for launching and analysing benchmark results of Syne Tune AutoML paper
* Option to avoid suffixing the tuner name with a random hash
* Tutorial for PASHA
* Adds support to run BORE with Hyperband
* Updates python version to 3.8 in remote launcher as 3.6 is EOL
* Support custom images in remote launcher
* Add ZS/SMFO scheduler
* Add support to tune python functions directly rather than file scripts (experimental)  
* Now display where results are written at the end of the tuning in addition to the beginning 
* Log config space with tuning metadata
* Add option to delete checkpoints of finished trials and improve support of checkpointing for SageMaker backend
* Add support for adding surrogate on top of blackboxes evaluations
 
### Changed
* Removed deprecated `search_space` which has been renamed `config_space` in all the code base
* Removed some unused dependencies
 
### Fixed
* fix conditional imports
* avoid sampling twice the same candidate in REA
* print tuning statistics only for numerics
* allow to load blackboxes without s3 credentials
* fix sagemaker root path in remote launcher
* fix a typo in product kernel implementation
