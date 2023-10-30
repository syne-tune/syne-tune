
# Change Log
All notable changes to this project will be documented in this file.
 
The format is based on [Keep a Changelog](http://keepachangelog.com/).


<a name="v0.10.0"></a>
## [v0.10.0] - 2023-11-01
### New Features
- Add example to resume tuning from previous experiment and update conf… ([#780](https://github.com/awslabs/syne-tune/issues/780))
- Add method to get the best configuration directly from Tuner, add com… ([#767](https://github.com/awslabs/syne-tune/issues/767))
- Add util to plot trials over time from ExperimentResult ([#768](https://github.com/awslabs/syne-tune/issues/768))
- Add SMAC wrapper and examples ([#765](https://github.com/awslabs/syne-tune/issues/765))
- Automatic streamlining of configuration space ([#741](https://github.com/awslabs/syne-tune/issues/741))

### CI
- automatically test all notebooks under `examples/notebooks` and render them in docs page ([#756](https://github.com/awslabs/syne-tune/issues/756))

### Bug Fixes
- Fix malformatted table in docs ([#757](https://github.com/awslabs/syne-tune/issues/757))
- pass arguments to scheduler ([#755](https://github.com/awslabs/syne-tune/issues/755))
- RSBO ([#748](https://github.com/awslabs/syne-tune/issues/748))
- Fix for issue [#749](https://github.com/awslabs/syne-tune/issues/749) ([#750](https://github.com/awslabs/syne-tune/issues/750))

### Maintenance
- Add smac to possible install tag, fix comment in example ([#766](https://github.com/awslabs/syne-tune/issues/766))
- Update sphinx requirement from <7.0.0 to <8.0.0 ([#745](https://github.com/awslabs/syne-tune/issues/745))
- Update numpy requirement from <1.24.0,>=1.16.0 to >=1.16.0,<1.27.0 ([#761](https://github.com/awslabs/syne-tune/issues/761))
- Bump release-drafter/release-drafter from 5.23.0 to 5.24.0 ([#735](https://github.com/awslabs/syne-tune/issues/735))
- Bump aws-actions/configure-aws-credentials from 3 to 4 ([#760](https://github.com/awslabs/syne-tune/issues/760))
- Bump actions/checkout from 3 to 4 ([#763](https://github.com/awslabs/syne-tune/issues/763))
- Bump actions/setup-python from 2 to 4 ([#762](https://github.com/awslabs/syne-tune/issues/762))
- Bump actions/checkout from 3 to 4 ([#759](https://github.com/awslabs/syne-tune/issues/759))
- Bump actions/setup-python from 2 to 4 ([#747](https://github.com/awslabs/syne-tune/issues/747))
- Bump aws-actions/configure-aws-credentials from 2 to 3 ([#752](https://github.com/awslabs/syne-tune/issues/752))

[v0.10.0]: https://github.com/awslabs/syne-tune/compare/v0.9.1...v0.10.0

 <a name="v0.9.1"></a>
## [v0.9.1] - 2023-07-19
### New Features
- New group tag 'basic' for dependencies of reasonable size ([#738](https://github.com/awslabs/syne-tune/issues/738))

### Documentation Updates
- Small fix and update of README.md ([#740](https://github.com/awslabs/syne-tune/issues/740))

### Maintenance
- Bump zgosalvez/github-actions-ensure-sha-pinned-actions from 2.1.3 to 2.1.4 ([#742](https://github.com/awslabs/syne-tune/issues/742))
- update README ([#739](https://github.com/awslabs/syne-tune/issues/739))


[v0.9.1]: https://github.com/awslabs/syne-tune/compare/v0.9.0...v0.9.1


<a name="v0.9.0"></a>
## [v0.9.0] - 2023-07-04

### Bug Fixes
- Clean up imports so that code runs with core dependencies. Update test workflows ([#734](https://github.com/awslabs/syne-tune/issues/734))
- Make sure RemoteLauncher works with requirements.txt not ending on \n ([#731](https://github.com/awslabs/syne-tune/issues/731))
- ExperimentResult.best_config() always used max mode ([#728](https://github.com/awslabs/syne-tune/issues/728))

### Code Refactoring
- Move code from benchmarking/commons to syne_tune/experiments ([#719](https://github.com/awslabs/syne-tune/issues/719))
- Wrappers for multi-objective methods ([#727](https://github.com/awslabs/syne-tune/issues/727))

### Documentation Updates
- Experimentation framework without Syne Tune installed from source ([#733](https://github.com/awslabs/syne-tune/issues/733))
- New tutorial on how to implement Bayesian optimization ([#709](https://github.com/awslabs/syne-tune/issues/709))
- Add CQR in list of method supported in readme ([#736](https://github.com/awslabs/syne-tune/issues/736))
- Updated documentation ([#732](https://github.com/awslabs/syne-tune/issues/732))

### Maintenance
- Fix yahpo dependency version of configspace ([#725](https://github.com/awslabs/syne-tune/issues/725))

[v0.9.0]: https://github.com/awslabs/syne-tune/compare/v0.8.0...v0.9.0


<a name="0.8.0"></a>
## [v0.8.0] - 2023-06-20

### New Features
- Code for OTHPO paper ([#710](https://github.com/awslabs/syne-tune/issues/710))
- Multi Surrogate Multi Objective Searcher ([#711](https://github.com/awslabs/syne-tune/issues/711))
- Allow to transform result dataframe before aggregation in plotting ([#696](https://github.com/awslabs/syne-tune/issues/696))
- Plotting for single seed results works as well ([#706](https://github.com/awslabs/syne-tune/issues/706))
- Add Conformal Quantile Regression method from ICML paper. ([#689](https://github.com/awslabs/syne-tune/issues/689))
- Some extra plotting features ([#694](https://github.com/awslabs/syne-tune/issues/694))
- Allow base kernel to be selected ([#698](https://github.com/awslabs/syne-tune/issues/698))
- Adding new searcher that uses contributed surrogate models.  ([#684](https://github.com/awslabs/syne-tune/issues/684))
- Try again in remote launching when ResourceLimitExceeded is caught ([#676](https://github.com/awslabs/syne-tune/issues/676))

### Bug Fixes
- Avoid sphinx 7.0.0, since the readthedocs build fails with that ([#720](https://github.com/awslabs/syne-tune/issues/720))
- Removed the broken example from github actions ([#718](https://github.com/awslabs/syne-tune/issues/718))
- Fix bug in RemoteLauncher with SageMakerBackend ([#717](https://github.com/awslabs/syne-tune/issues/717))
- Some fixes for multi-objective benchmarking ([#708](https://github.com/awslabs/syne-tune/issues/708))
- Command line arguments of benchmarking ([#702](https://github.com/awslabs/syne-tune/issues/702))
- Avoid MOO warning message ([#700](https://github.com/awslabs/syne-tune/issues/700))
- New benchmarks need **kwargs, so that defaults can be overwritte… ([#693](https://github.com/awslabs/syne-tune/issues/693))
- Remove (CITATION?). Currently, these results are not published a… ([#692](https://github.com/awslabs/syne-tune/issues/692))

### Code Refactoring
- Remove tuple option for Type[AcquisitionFunction], as this … ([#715](https://github.com/awslabs/syne-tune/issues/715))
- Simplification in scikit-learn based estimators ([#704](https://github.com/awslabs/syne-tune/issues/704))
- Clean up transformer_wikitext2 code ([#701](https://github.com/awslabs/syne-tune/issues/701))
- Refactoring of scikit-learn based surrogate models ([#699](https://github.com/awslabs/syne-tune/issues/699))

### Documentation Updates
- Tutorial on experimentation ([#705](https://github.com/awslabs/syne-tune/issues/705))
- Tutorial for ODSC presentation ([#703](https://github.com/awslabs/syne-tune/issues/703))
- **examples:** Typo fix ([#713](https://github.com/awslabs/syne-tune/issues/713))

### Maintenance
- update README ([#723](https://github.com/awslabs/syne-tune/issues/723))
- Remove documentation link from README.md ([#722](https://github.com/awslabs/syne-tune/issues/722))
- Update pytest-cov requirement from ~=4.0.0 to ~=4.1.0 ([#697](https://github.com/awslabs/syne-tune/issues/697))
- enable caching for unit tests, fix README typo ([#695](https://github.com/awslabs/syne-tune/issues/695))

[v0.8.0]: https://github.com/awslabs/syne-tune/compare/v0.7.0...v0.8.0


<a name="v0.7.0"></a>
## [v0.7.0] - 2023-05-23

### New Features
- Plotting code for fine_tuning_transformer_glue benchmark ([#686](https://github.com/awslabs/syne-tune/issues/686))
- Backoff decorator for jobs scheduling ([#673](https://github.com/awslabs/syne-tune/issues/673))
- Plotting tools for result of comparative study ([#652](https://github.com/awslabs/syne-tune/issues/652))
- New surrogate models API ([#669](https://github.com/awslabs/syne-tune/issues/669))
- implementation of NSGA-2 ([#660](https://github.com/awslabs/syne-tune/issues/660))

### Bug Fixes
- typos in "Getting started" code example ([#687](https://github.com/awslabs/syne-tune/issues/687))
- Made scoring function compatible with acquisition function interface ([#682](https://github.com/awslabs/syne-tune/issues/682))
- Sklearn estimator now fits the surrogate model without considering pending evaluations. ([#681](https://github.com/awslabs/syne-tune/issues/681))
- Domain.len returns 0 for infinite domain (e.g., Float) ([#672](https://github.com/awslabs/syne-tune/issues/672))
- Fix bug in benchmarking where default values were int instead of… ([#670](https://github.com/awslabs/syne-tune/issues/670))
- Bugfix in benchmarking ([#667](https://github.com/awslabs/syne-tune/issues/667))
- Fix import bug, and also new workflow to install Syne Tune with … ([#663](https://github.com/awslabs/syne-tune/issues/663))

### Code Refactoring
- Cleanup of training scripts ([#668](https://github.com/awslabs/syne-tune/issues/668))
- Allow benchmark to be multi-objective ([#666](https://github.com/awslabs/syne-tune/issues/666))
- Graduate some content in benchmarking/nursery and create ne… ([#659](https://github.com/awslabs/syne-tune/issues/659))
- SurrogateModel -> Predictor; ModelFactory -> Estimator; Simplify bayesopt code ([#651](https://github.com/awslabs/syne-tune/issues/651))

### Documentation Updates
- Tutorial for plotting ([#675](https://github.com/awslabs/syne-tune/issues/675))
- Update README.md ([#661](https://github.com/awslabs/syne-tune/issues/661))

### Maintenance
- Bump codecov/codecov-action from 3.1.3 to 3.1.4 ([#685](https://github.com/awslabs/syne-tune/issues/685))
- fix unit test ([#678](https://github.com/awslabs/syne-tune/issues/678))
- add initial multiobjective benchmarking logic ([#674](https://github.com/awslabs/syne-tune/issues/674))
- re-introduce check-standalone-bayesian-optimization.yml ([#664](https://github.com/awslabs/syne-tune/issues/664))
- Increase time limits on flakey tests ([#658](https://github.com/awslabs/syne-tune/issues/658))
- Bump zgosalvez/github-actions-ensure-sha-pinned-actions from 2.1.2 to 2.1.3 ([#655](https://github.com/awslabs/syne-tune/issues/655))

[v0.7.0]: https://github.com/awslabs/syne-tune/compare/v0.6.0...v0.7.0


<a name="v0.6.0"></a>
## [v0.6.0] - 2023-05-08

### New Features
- Add example to optimize hyperparams of HF models on SWAG ([#641](https://github.com/awslabs/syne-tune/issues/641))
- LocalBackend allows for >1 GPU per trial ([#635](https://github.com/awslabs/syne-tune/issues/635))

### Bug Fixes
- Fixing limits for random seed ([#645](https://github.com/awslabs/syne-tune/issues/645))
- Pin fastparquet version ([#647](https://github.com/awslabs/syne-tune/issues/647))
- Fix `ListTrainingJobs` throttling for E2E tests ([#634](https://github.com/awslabs/syne-tune/issues/634))
- fix: Random seed range in benchmarking ([#656](https://github.com/awslabs/syne-tune/issues/656))

### Code Refactoring
- Towards introduction of plotting code ([#625](https://github.com/awslabs/syne-tune/issues/625))
- Simplified Bayesian optimization code ([#640](https://github.com/awslabs/syne-tune/issues/640))
- Changes to benchmarking formalism ([#637](https://github.com/awslabs/syne-tune/issues/637))

### Documentation Updates
- Link to D2L chapter in our docs ([#653](https://github.com/awslabs/syne-tune/issues/653))
- Alternative of 1-NN surrogate to restrict_configurations ([#650](https://github.com/awslabs/syne-tune/issues/650))

### Maintenance
- Bump gpy from 1.9.9 to 1.12.0 ([#643](https://github.com/awslabs/syne-tune/issues/643))
- Move long-running test ([#654](https://github.com/awslabs/syne-tune/issues/654))
- Run two additional examples as tests ([#648](https://github.com/awslabs/syne-tune/issues/648))
- Only validate PR titles; specify list of valid prefixes ([#646](https://github.com/awslabs/syne-tune/issues/646))
- Bump codecov/codecov-action from 3.1.2 to 3.1.3 ([#639](https://github.com/awslabs/syne-tune/issues/639))
- Create workflow to draft CHANGELOG.md; auto-tag PRs on push ([#631](https://github.com/awslabs/syne-tune/issues/631))

[v0.6.0]: https://github.com/awslabs/syne-tune/compare/v0.5.0...v0.6.0


<a name="v0.5.0"></a>
## [v0.5.0] - 2023-04-20

### New Features

- Speculative early checkpoint removal for async multi-fidelity ([#628](https://github.com/awslabs/syne-tune/issues/628))
- Simple linear scalarization scheduler ([#619](https://github.com/awslabs/syne-tune/issues/619))
- Automatic termination criterion ([#605](https://github.com/awslabs/syne-tune/issues/605))
- All schedulers have a is_multiobjective_scheduler function ([#618](https://github.com/awslabs/syne-tune/issues/618))
- Allow for customized extra results to be written to results.csv.zip ([#612](https://github.com/awslabs/syne-tune/issues/612))
- Plotting functions to analyse multi-objective experiments. ([#611](https://github.com/awslabs/syne-tune/issues/611))
- Downsampling of observed data for single-fidelity Bayesian optimization ([#607](https://github.com/awslabs/syne-tune/issues/607))

### Bug Fixes
- CI failing with `ModuleNotFoundError: No module named 'examples'` error ([#626](https://github.com/awslabs/syne-tune/issues/626))
- Call init of class ([#584](https://github.com/awslabs/syne-tune/issues/584))
- Random seed initialisation limited to int32 ([#608](https://github.com/awslabs/syne-tune/issues/608))
- Make sure that checkpoints in PBT are removed once they are no longer needed ([#600](https://github.com/awslabs/syne-tune/pull/600))

### Code Refactoring
- Move early checkpoint removal into mixin ([#621](https://github.com/awslabs/syne-tune/issues/621))
- Keep rung levels sorted in HyperbandScheduler ([#604](https://github.com/awslabs/syne-tune/issues/604))
- Move utils from benchmarking to syne_tune ([#606](https://github.com/awslabs/syne-tune/pull/606))

### Documentation Updates
- Update instructions for how to install from source ([#629](https://github.com/awslabs/syne-tune/issues/629))
- Update README.md ([#615](https://github.com/awslabs/syne-tune/pull/615))

### Maintenance
- Bump codecov/codecov-action from 3.1.1 to 3.1.2 ([#623](https://github.com/awslabs/syne-tune/issues/623))
- Bump zgosalvez/github-actions-ensure-sha-pinned-actions from 2.1.0 to 2.1.2 ([#624](https://github.com/awslabs/syne-tune/issues/624))
- Add release drafter automation ([#568](https://github.com/awslabs/syne-tune/issues/568))
- Moved scheduler metadata generation ([#617](https://github.com/awslabs/syne-tune/issues/617))
- Bump tensorflow from 2.11.0 to 2.11.1 in /examples/training_scripts/rl_cartpole ([#609](https://github.com/awslabs/syne-tune/issues/609))

[v0.5.0]: https://github.com/awslabs/syne-tune/compare/v0.4.1...v0.5.0


## [0.4.1] - 2023-03-16

We release version 0.4.1 which you can install with `pip install syne-tune[extra]`.

Thanks to all contributors:
@mseeger, @wesk, @sighellan, @aaronkl, @wistuba, @jgolebiowski, @610v4nn1, @geoalgo


### Added
* New tutorial: Using Syne Tune for Transfer Learning
* Multi-objective regularized evolution (MO-REA)
* Gaussian-process based methods support Box-Cox target transform and input
  warping
* Configuration can be passed as JSON file to evaluation script
* Remote tuning: Metrics are published to CloudWatch console

### Changed
* Refactoring and cleanup of `benchmarking`
* FIFOScheduler supports multi-objective schedulers
* Remote tuning with local backend: Checkpoints are not synced to S3
* Code in documentation is now tested automatically
* Benchmarking: Use longer descriptive names for result directories

### Fixed
* Specific state converter for DyHPO (can work poorly with the state converter
  for MOBSTER or Hyper-Tune)
* Fix of BOHB and SyncBOHB
* _objective_function parsing for YAHPO blackbox


## [0.4.0] - 2023-02-21

We release version 0.4.0 which you can install with `pip install syne-tune[extra]`.

Thanks to all contributors:
@mseeger, @wesk, @sighellan, @ondrejbohdal, @aaronkl, @wistuba, @jacekgo, @geoalgo


### Added
* New HPO algorithm: DyHPO
* New tutorial: Progressive ASHA
* Extended developer tutorial: Wrapping external scheduler code
* Schedulers can be restricted to set of configurations as subset of the
  full configuration space (option `restrict_configurations`)
* Data filtering for model-based multi-fidelity schedulers to avoid slowdown
  (option `max_size_data_for_model`)
* Input warping for Gaussian process covariance kernels
* Allow searchers to return the same configuration more than once (option 
  `allow_duplicates`). Unify duplicate filtering across all searchers
* Support `Ordinal` and `OrdinalNearestNeighbor` in `active_config_space` and
  warmstarting

### Changed
* Major simplifications in `benchmarking`
* Examples and documentation are encouraging to use `max_resource_attr`
  instead of `max_t`
* Major refactoring and extensions in testing and CI pipeline
* `RemoteLauncher` does not require custom container anymore

### Fixed
* `TensorboardCallback`: Logging of hyperparameters made optional, and updated
  example
* Refactoring of `BoTorchSearcher`
* Default value for DEHB baseline fixed
* Number of GPUs is now determined correctly


## [0.3.4] - 2023-01-11

We release version 0.3.4 which you can install with `pip install syne-tune[extra]`.

Thanks to all contributors!

### Changed
* Different searchers suggest the same initial random configurations if run
  with the same random seed
* Streamlined how SageMaker backend uses pre-built containers
* Improvements in documentation
* Improved random seed management in benchmarking
* New baseline wrappers: BOHB, ASHABORE, BOTorch, KDE

### Fixed
* Switch off hash checking of blackbox repository by default, since hash
  computation seem system-dependent. Hash computation will be fixed in a
  forthcoming release, and will be switched on again
* Fixed defaults of Hyper-Tune in benchmarking
* Bug fix of SyncMOBSTER (along with refactoring)
* Bug fix in REA (RegularizedEvolution)
* Bug fix in examples for constrained and cost-aware BO


## [0.3.3] - 2022-12-19

We release version 0.3.3 which you can install with `pip install syne-tune[extra]`.

Thanks to all contributors (sorted by chronological commit order):
@mseeger, @mina-ghashami, @aaronkl, @jgolebiowski, @Valavanca, @TrellixVulnTeam,
@geoalgo, @wistuba, @mlblack

### Added
* Revamped documentation hosted at https://syne-tune.readthedocs.io
* New tutorial: Benchmarking in Syne Tune
* Added section on backends in Basics of Syne Tune tutorial
* Control of re-creating of blackboxes by checking and storing hash codes
* New benchmark: Transformer on WikiText-2
* Support SageMaker managed warm pools in SageMaker backend
* Improvements for benchmarking with YAHPO blackboxes
* Support points_to_evaluate in BORE
* SageMaker backend supports delete_checkpoints=True

### Changed
* GridSearch supports all domain types now
* BlackboxSurrogate of blackbox repository supports different modes
* Add timeout to unit tests
* New unit tests which run schedulers for longer, using simulator backend

### Fixed
* HyperbandScheduler: does_pause_resume for all types
* ASHA with type="promotion" did not work when checkpointing not implemented
* Fixed preprocessing of PD1 blackbox
* SageMaker backend reports on true number of busy workers (fixes issue #250)
* Fix issue with uploading/syncing to S3 of YAHPO blackbox
* Fix YAHPO surrogate evaluations in the presence of inactive hyperparameters
* Fix treatment of Status.paused in TuningStatus and Tuner


## [0.3.2] - 2022-10-14

We release 0.3.2 version which you can install with `pip install syne-tune[extra]`.

Thanks to all contributors (sorted by chronological commit order):
@mseeger, @geoalgo, @aaronkl, @wistuba, @talesa, @hfurkanbozkurt, @rsnirwan,
@duck105, @ondrejbohdal, @iaroslav-ai, @austinmw, @banyikun, @jjaeyeon, @ltiao

### Added
* New tutorial: How to Contribute a New Scheduler
* New tutorial: Multi-Fidelity Hyperparameter Optimization
* YAHPO benchmarks integrated into blackbox repository
* PD1 benchmarks integrated into blackbox repository
* New HPO algorithm: Hyper-Tune
* New HPO algorithm: Differential Evolution Hyperband (DEHB)
* New experimental HPO algorithm: Neuralband
* New HPO algorithm: Grid search (categorical variables only)
* BOTorch searcher
* MOBSTER algorithm supports independent GPs at each rung level
* Support for launching experiments in benchmarking/commons, for local,
  SageMaker, and simulator backend
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
* In SageMaker backend, trials with low IDs received reports several times. This
  is fixed
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
