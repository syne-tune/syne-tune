# YAHPO: Overview of Scenarios and Their Configuration Spaces


## nb301

```python
instances = ['CIFAR10']

metrics = ['val_accuracy', 'runtime']

op_types = ['max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5']
config_space = {
   'NetworkSelectorDatasetInfo_COLON_darts_COLON_edge_normal_0': choice(op_types),
   'NetworkSelectorDatasetInfo_COLON_darts_COLON_edge_normal_1': choice(op_types),
   'NetworkSelectorDatasetInfo_COLON_darts_COLON_edge_reduce_0': choice(op_types),
   'NetworkSelectorDatasetInfo_COLON_darts_COLON_edge_reduce_1': choice(op_types),
   'NetworkSelectorDatasetInfo_COLON_darts_COLON_inputs_node_normal_3': choice(['0_1', '0_2', '1_2']),
   'NetworkSelectorDatasetInfo_COLON_darts_COLON_inputs_node_normal_4': choice(['0_1', '0_2', '0_3', '1_2', '1_3', '2_3']),
   'NetworkSelectorDatasetInfo_COLON_darts_COLON_inputs_node_normal_5': choice(['0_1', '0_2', '0_3', '0_4', '1_2', '1_3', '1_4', '2_3', '2_4', '3_4']),
   'NetworkSelectorDatasetInfo_COLON_darts_COLON_inputs_node_reduce_3': choice(['0_1', '0_2', '1_2']),
   'NetworkSelectorDatasetInfo_COLON_darts_COLON_inputs_node_reduce_4': choice(['0_1', '0_2', '0_3', '1_2', '1_3', '2_3']),
   'NetworkSelectorDatasetInfo_COLON_darts_COLON_inputs_node_reduce_5': choice(['0_1', '0_2', '0_3', '0_4', '1_2', '1_3', '1_4', '2_3', '2_4', '3_4']),
   'NetworkSelectorDatasetInfo_COLON_darts_COLON_edge_normal_10': choice(op_types),
   'NetworkSelectorDatasetInfo_COLON_darts_COLON_edge_normal_11': choice(op_types),
   'NetworkSelectorDatasetInfo_COLON_darts_COLON_edge_normal_12': choice(op_types),
   'NetworkSelectorDatasetInfo_COLON_darts_COLON_edge_normal_13': choice(op_types),
   'NetworkSelectorDatasetInfo_COLON_darts_COLON_edge_normal_2': choice(op_types),
   'NetworkSelectorDatasetInfo_COLON_darts_COLON_edge_normal_3': choice(op_types),
   'NetworkSelectorDatasetInfo_COLON_darts_COLON_edge_normal_4': choice(op_types),
   'NetworkSelectorDatasetInfo_COLON_darts_COLON_edge_normal_5': choice(op_types),
   'NetworkSelectorDatasetInfo_COLON_darts_COLON_edge_normal_6': choice(op_types),
   'NetworkSelectorDatasetInfo_COLON_darts_COLON_edge_normal_7': choice(op_types),
   'NetworkSelectorDatasetInfo_COLON_darts_COLON_edge_normal_8': choice(op_types),
   'NetworkSelectorDatasetInfo_COLON_darts_COLON_edge_normal_9': choice(op_types),
   'NetworkSelectorDatasetInfo_COLON_darts_COLON_edge_reduce_10': choice(op_types),
   'NetworkSelectorDatasetInfo_COLON_darts_COLON_edge_reduce_11': choice(op_types),
   'NetworkSelectorDatasetInfo_COLON_darts_COLON_edge_reduce_12': choice(op_types),
   'NetworkSelectorDatasetInfo_COLON_darts_COLON_edge_reduce_13': choice(op_types),
   'NetworkSelectorDatasetInfo_COLON_darts_COLON_edge_reduce_2': choice(op_types),
   'NetworkSelectorDatasetInfo_COLON_darts_COLON_edge_reduce_3': choice(op_types),
   'NetworkSelectorDatasetInfo_COLON_darts_COLON_edge_reduce_4': choice(op_types),
   'NetworkSelectorDatasetInfo_COLON_darts_COLON_edge_reduce_5': choice(op_types),
   'NetworkSelectorDatasetInfo_COLON_darts_COLON_edge_reduce_6': choice(op_types),
   'NetworkSelectorDatasetInfo_COLON_darts_COLON_edge_reduce_7': choice(op_types),
   'NetworkSelectorDatasetInfo_COLON_darts_COLON_edge_reduce_8': choice(op_types),
   'NetworkSelectorDatasetInfo_COLON_darts_COLON_edge_reduce_9': choice(op_types),
}
fidelity_space = {
   'epoch': randint(1, 98),
}
```

## lcbench

```python
instances = ['3945', '7593', '34539', '126025', '126026', '126029', '146212', '167104', '167149', '167152', '167161', '167168', '167181', '167184', '167185', '167190', '167200', '167201', '168329', '168330', '168331', '168335', '168868', '168908', '168910', '189354', '189862', '189865', '189866', '189873', '189905', '189906', '189908', '189909']

metrics = ['time', 'val_accuracy', 'val_cross_entropy', 'val_balanced_accuracy', 'test_cross_entropy', 'test_balanced_accuracy']

config_space = {
   'batch_size': lograndint(16, 512),
   'learning_rate': loguniform(0.0001, 0.1),
   'max_dropout': uniform(0.0, 1.0),
   'max_units': lograndint(64, 1024),
   'momentum': uniform(0.1, 0.99),
   'num_layers': randint(1, 5),
   'weight_decay': uniform(1e-05, 0.1),
}
fidelity_space = {
   'epoch': randint(1, 52),
}
```

## iaml: Interpretable AutoML

Text from YAHPO paper (Appendix F):

All scenarios prefixed with iaml_ rely on data that were newly collected by us.
Different mlr3 [41] learners (“classif.glmnet”, “classif.rpart”, “classif.ranger”,
“classif.xgboost”) were incorporated into an ML pipeline with minimal preprocessing
(removing constant features, fixing unseen factor levels during prediction and missing
value imputation for factor variables by sampling from non- missing training levels) 
via mlr3pipelines [8]. Hyperparameters of the learners were sampled uniformly at
random (for the search spaces, see Table 12) and the ML pipeline performance
(classification error - mmce, F1 score - f1, AUC - auc, logloss - logloss) was
evaluated via 5- fold cross-validation on the following OpenML [71] datasets (dataset
id): 40981, 41146, 1489, 1067. Each pipeline was then refitted and used for prediction
on the whole data to estimate training and predict time (timetrain, timepredict) and
RAM usage (during training and prediction, ramtrain and rampredict as well as model
size, rammodel). Moreover, interpretability measures as described in [52] were computed
for all models: number of features used (nf), interaction strength of features (ias)
and main effect complexity of features (mec). To our best knowledge, this is the first
publicly available benchmark that combines performance, resource usage and
interpretability of models allowing for the construction of interesting
multi-objective benchmarks. Hyperparameter configurations were evaluated at different
fidelity steps (training sizes of the following fractions:
0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1) achieved via incorporating resampling in the ML
pipeline. The super learner scenario was constructed by using the data of all four
base learners introducing conditional hyperparameters in the form of branching. In
total, 5451872 different configurations were evaluated. Data collection was performed
on the moran partition of the ARCC Teton HPC cluster of the University of Wyoming
using batchtools [42] for job scheduling and took around 9.8 CPU years. Surrogate
models were then fitted on the available data as described in Supplement D.1. Table 12
lists all hyperparameters of the search spaces of the iaml_ scenarios. Instance ID’s
correspond to OpenML [71] dataset ids through which dataset properties can be
queried.

### iaml_super

```python
instances = ['40981', '41146', '1489', '1067']

metrics = ['mmce', 'f1', 'auc', 'logloss', 'ramtrain', 'rammodel', 'rampredict', 'timetrain', 'timepredict', 'mec', 'ias', 'nf']

config_space = {
   'learner': choice(['ranger', 'glmnet', 'xgboost', 'rpart']),
   'glmnet.alpha': uniform(0.0, 1.0),
   'glmnet.s': loguniform(0.0001, 1000),
   'ranger.min.node.size': randint(1, 100),
   'ranger.mtry.ratio': uniform(0.0, 1.0),
   'ranger.num.trees': randint(1, 2000),
   'ranger.replace': choice(['TRUE', 'FALSE']),
   'ranger.respect.unordered.factors': choice(['ignore', 'order', 'partition']),
   'ranger.sample.fraction': uniform(0.1, 1.0),
   'ranger.splitrule': choice(['gini', 'extratrees']),
   'rpart.cp': loguniform(0.0001, 1.0),
   'rpart.maxdepth': randint(1, 30),
   'rpart.minbucket': randint(1, 100),
   'rpart.minsplit': randint(1, 100),
   'xgboost.alpha': loguniform(0.0001, 1000),
   'xgboost.booster': choice(['gblinear', 'gbtree', 'dart']),
   'xgboost.lambda': loguniform(0.0001, 1000),
   'xgboost.nrounds': lograndint(3, 2000),
   'xgboost.subsample': uniform(0.1, 1.0),
   'ranger.num.random.splits': randint(1, 100),
   'xgboost.colsample_bylevel': uniform(0.01, 1.0),
   'xgboost.colsample_bytree': uniform(0.01, 1.0),
   'xgboost.eta': loguniform(0.0001, 1.0),
   'xgboost.gamma': loguniform(0.0001, 7),
   'xgboost.max_depth': randint(1, 15),
   'xgboost.min_child_weight': loguniform(2.718281828459045, 150),
   'xgboost.rate_drop': uniform(0.0, 1.0),
   'xgboost.skip_drop': uniform(0.0, 1.0),
}
fidelity_space = {
   'trainsize': uniform(0.03, 1.0),
}
```

### iaml_rpart

```python
instances = ['40981', '41146', '1489', '1067']

metrics = ['mmce', 'f1', 'auc', 'logloss', 'ramtrain', 'rammodel', 'rampredict', 'timetrain', 'timepredict', 'mec', 'ias', 'nf']

config_space = {
   'cp': loguniform(0.0001, 1.0),
   'maxdepth': randint(1, 30),
   'minbucket': randint(1, 100),
   'minsplit': randint(1, 100),
}
fidelity_space = {
   'trainsize': uniform(0.03, 1.0),
}
```

### iaml_glmnet

```python
instances = ['40981', '41146', '1489', '1067']

metrics = ['mmce', 'f1', 'auc', 'logloss', 'ramtrain', 'rammodel', 'rampredict', 'timetrain', 'timepredict', 'mec', 'ias', 'nf']

config_space = {
   'alpha': uniform(0.0, 1.0),
   's': loguniform(0.0001, 1000),
}
fidelity_space = {
   'trainsize': uniform(0.03, 1.0),
}
```

### iaml_ranger

```python
instances = ['40981', '41146', '1489', '1067']

metrics = ['mmce', 'f1', 'auc', 'logloss', 'ramtrain', 'rammodel', 'rampredict', 'timetrain', 'timepredict', 'mec', 'ias', 'nf']

config_space = {
   'min.node.size': randint(1, 100),
   'mtry.ratio': uniform(0.0, 1.0),
   'num.trees': randint(1, 2000),
   'replace': choice(['TRUE', 'FALSE']),
   'respect.unordered.factors': choice(['ignore', 'order', 'partition']),
   'sample.fraction': uniform(0.1, 1.0),
   'splitrule': choice(['gini', 'extratrees']),
   'num.random.splits': randint(1, 100),
}
fidelity_space = {
   'trainsize': uniform(0.03, 1.0),
}
```

### iaml_xgboost

```python
instances = ['40981', '41146', '1489', '1067']

metrics = ['mmce', 'f1', 'auc', 'logloss', 'ramtrain', 'rammodel', 'rampredict', 'timetrain', 'timepredict', 'mec', 'ias', 'nf']

config_space = {
   'alpha': loguniform(0.0001, 1000),
   'booster': choice(['gblinear', 'gbtree', 'dart']),
   'lambda': loguniform(0.0001, 1000),
   'nrounds': lograndint(3, 2000),
   'subsample': uniform(0.1, 1.0),
   'colsample_bylevel': uniform(0.01, 1.0),
   'colsample_bytree': uniform(0.01, 1.0),
   'eta': loguniform(0.0001, 1.0),
   'gamma': loguniform(0.0001, 7),
   'max_depth': randint(1, 15),
   'min_child_weight': loguniform(2.718281828459045, 150),
   'rate_drop': uniform(0.0, 1.0),
   'skip_drop': uniform(0.0, 1.0),
}
fidelity_space = {
   'trainsize': uniform(0.03, 1.0),
}
```


## rbv2: Random Bot V2

Text from YAHPO paper (Appendix F):

All scenarios prefixed with rbv2_ use data described in [9]. Data contains results
from several ML algorithms trained across up to 119 datasets evaluated for a large
amount of random evaluations. Table 9 lists all hyperparameters of the search space
of the rbv2_ scenarios. Targets are given by accuracy (acc), balanced accuracy (bac),
AUC (auc), Brier Score (brier), F1 (f1), log loss (logloss), time for training the
model (timetrain), and memory usage (memory). Surrogates are fitted on subsets of the
full data available from [9], such that a minimum of 1500 and a maximum of 200000 
(depending on the scenario) evaluations are available for each instance in each
scenario. All scenarios consist of a pre-processing step (missing data imputation) and
a subsequently fitted ML algorithm. Instance ID’s correspond to OpenML [71] dataset
ids through which dataset properties can be queried . OpenML tasks corresponding to
each dataset can be obtained from [9]. We abbreviate the num.impute.selected.cpo
hyperparameter with imputation throughout the tables. We fix the repl parameter to 10
for experiments.

From reference [9]:

We use subsampling with the following factions of the training data:
0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9.

From communication with YAHPO authors:
* F1 is problematic for multi-class classification, since it is NaN often in the
  raw data. It should only be used with binary classification setups.
* AUC can often be very close to 1, since the OpenML tasks are often quite simple.
  For example, for `svm`, about 3% of configs give AUC larger than 0.9999999.

### rbv2_super

```python
instances = ['41138', '40981', '4134', '1220', '4154', '41163', '4538', '40978', '375', '1111', '40496', '40966', '4534', '40900', '40536', '41156', '1590', '1457', '458', '469', '41157', '11', '1461', '1462', '1464', '15', '40975', '41142', '40701', '40994', '23', '1468', '40668', '29', '31', '6332', '37', '40670', '23381', '151', '188', '41164', '1475', '1476', '1478', '41169', '1479', '41212', '1480', '300', '41143', '1053', '41027', '1067', '1063', '41162', '3', '6', '1485', '1056', '12', '14', '16', '18', '40979', '22', '1515', '334', '24', '1486', '1493', '28', '1487', '1068', '1050', '1049', '32', '1489', '470', '1494', '182', '312', '40984', '1501', '40685', '38', '42', '44', '46', '40982', '1040', '41146', '377', '40499', '50', '54', '307', '1497', '60', '1510', '40983', '40498', '181']

metrics = ['acc', 'bac', 'auc', 'brier', 'f1', 'logloss', 'timetrain', 'timepredict', 'memory']

config_space = {
   'learner_id': choice(['aknn', 'glmnet', 'ranger', 'rpart', 'svm', 'xgboost']),
   'num.impute.selected.cpo': choice(['impute.mean', 'impute.median', 'impute.hist']),
   'aknn.M': randint(18, 50),
   'aknn.distance': choice(['l2', 'cosine', 'ip']),
   'aknn.ef': lograndint(7, 403),
   'aknn.ef_construction': lograndint(7, 1097),
   'aknn.k': randint(1, 50),
   'glmnet.alpha': uniform(0.0, 1.0),
   'glmnet.s': loguniform(0.0009118819655545162, 1096.6331584284585),
   'ranger.min.node.size': randint(1, 100),
   'ranger.mtry.power': randint(0, 1),
   'ranger.num.trees': randint(1, 2000),
   'ranger.respect.unordered.factors': choice(['ignore', 'order', 'partition']),
   'ranger.sample.fraction': uniform(0.1, 1.0),
   'ranger.splitrule': choice(['gini', 'extratrees']),
   'rpart.cp': loguniform(0.0009118819655545162, 1.0),
   'rpart.maxdepth': randint(1, 30),
   'rpart.minbucket': randint(1, 100),
   'rpart.minsplit': randint(1, 100),
   'svm.cost': loguniform(4.5399929762484854e-05, 22026.465794806718),
   'svm.kernel': choice(['linear', 'polynomial', 'radial']),
   'svm.tolerance': loguniform(4.5399929762484854e-05, 2.0),
   'xgboost.alpha': loguniform(0.0009118819655545162, 1096.6331584284585),
   'xgboost.booster': choice(['gblinear', 'gbtree', 'dart']),
   'xgboost.lambda': loguniform(0.0009118819655545162, 1096.6331584284585),
   'xgboost.nrounds': lograndint(7, 2981),
   'xgboost.subsample': uniform(0.1, 1.0),
   'ranger.num.random.splits': randint(1, 100),
   'svm.degree': randint(2, 5),
   'svm.gamma': loguniform(4.5399929762484854e-05, 22026.465794806718),
   'xgboost.colsample_bylevel': uniform(0.01, 1.0),
   'xgboost.colsample_bytree': uniform(0.01, 1.0),
   'xgboost.eta': loguniform(0.0009118819655545162, 1.0),
   'xgboost.gamma': loguniform(4.5399929762484854e-05, 7.38905609893065),
   'xgboost.max_depth': randint(1, 15),
   'xgboost.min_child_weight': loguniform(2.718281828459045, 148.4131591025766),
   'xgboost.rate_drop': uniform(0.0, 1.0),
   'xgboost.skip_drop': uniform(0.0, 1.0),
}
fidelity_space = {
   'repl': randint(1, 10),
   'trainsize': uniform(0.03, 1.0),
}
```

### rbv2_rpart

```python
instances = ['41138', '4135', '40981', '4134', '40927', '1220', '4154', '40923', '41163', '40996', '4538', '40978', '375', '1111', '40496', '40966', '41150', '4534', '40900', '40536', '41156', '1590', '1457', '458', '469', '41157', '11', '1461', '1462', '1464', '15', '40975', '41142', '40701', '40994', '23', '1468', '40668', '29', '31', '6332', '37', '4541', '40670', '23381', '151', '188', '41164', '1475', '1476', '41159', '1478', '41169', '23512', '1479', '41212', '1480', '300', '41168', '41143', '1053', '41027', '1067', '1063', '41162', '3', '6', '1485', '1056', '12', '14', '16', '18', '40979', '22', '1515', '554', '334', '24', '1486', '23517', '1493', '28', '1487', '1068', '1050', '1049', '32', '1489', '470', '1494', '41161', '41165', '182', '312', '40984', '1501', '40685', '38', '42', '44', '46', '40982', '1040', '41146', '377', '40499', '50', '54', '41166', '307', '1497', '60', '1510', '40983', '40498', '181']

metrics = ['acc', 'bac', 'auc', 'brier', 'f1', 'logloss', 'timetrain', 'timepredict', 'memory']

config_space = {
   'cp': loguniform(0.0009118819655545162, 1.0),
   'maxdepth': randint(1, 30),
   'minbucket': randint(1, 100),
   'minsplit': randint(1, 100),
   'num.impute.selected.cpo': choice(['impute.mean', 'impute.median', 'impute.hist']),
}
fidelity_space = {
   'repl': randint(1, 10),
   'trainsize': uniform(0.03, 1.0),
}
```

### rbv2_glmnet

```python
instances = ['41138', '4135', '40981', '4134', '1220', '4154', '41163', '4538', '40978', '375', '1111', '40496', '40966', '41150', '4534', '40900', '40536', '41156', '1590', '1457', '458', '469', '41157', '11', '1461', '1462', '1464', '15', '40975', '41142', '40701', '40994', '23', '1468', '40668', '29', '31', '6332', '37', '4541', '40670', '23381', '151', '188', '41164', '1475', '1476', '41159', '1478', '41169', '23512', '1479', '41212', '1480', '300', '41168', '41143', '1053', '41027', '1067', '1063', '41162', '3', '6', '1485', '1056', '12', '14', '16', '18', '40979', '22', '1515', '334', '24', '1486', '23517', '41278', '1493', '28', '1487', '1068', '1050', '1049', '32', '1489', '470', '1494', '41161', '182', '312', '40984', '1501', '40685', '38', '42', '44', '46', '40982', '1040', '41146', '377', '40499', '50', '54', '41216', '41166', '307', '1497', '60', '1510', '40983', '40498', '181', '554']

metrics = ['acc', 'bac', 'auc', 'brier', 'f1', 'logloss', 'timetrain', 'timepredict', 'memory']

config_space = {
   'alpha': uniform(0.0, 1.0),
   'num.impute.selected.cpo': choice(['impute.mean', 'impute.median', 'impute.hist']),
   's': loguniform(0.0009118819655545162, 1096.6331584284585),
}
fidelity_space = {
   'repl': randint(1, 10),
   'trainsize': uniform(0.03, 1.0),
}
```

### rbv2_ranger

```python
instances = ['4135', '40981', '4134', '1220', '4154', '4538', '40978', '375', '40496', '40966', '4534', '40900', '40536', '41156', '1590', '1457', '458', '469', '41157', '11', '1461', '1462', '1464', '15', '40975', '41142', '40701', '40994', '23', '1468', '40668', '29', '31', '6332', '37', '40670', '23381', '151', '188', '41164', '1475', '1476', '1478', '1479', '41212', '1480', '41143', '1053', '41027', '1067', '1063', '3', '6', '1485', '1056', '12', '14', '16', '18', '40979', '22', '1515', '334', '24', '1486', '41278', '28', '1487', '1068', '1050', '1049', '32', '1489', '470', '1494', '182', '312', '40984', '1501', '40685', '38', '42', '44', '46', '40982', '1040', '41146', '377', '40499', '50', '54', '41216', '307', '1497', '60', '1510', '40983', '40498', '181', '41138', '41163', '1111', '41159', '300', '41162', '23517', '41165', '4541', '41161', '41166', '40927', '41150', '23512', '41168', '1493', '40996', '554', '40923', '41169']

metrics = ['acc', 'bac', 'auc', 'brier', 'f1', 'logloss', 'timetrain', 'timepredict', 'memory']

config_space = {
   'min.node.size': randint(1, 100),
   'mtry.power': randint(0, 1),
   'num.impute.selected.cpo': choice(['impute.mean', 'impute.median', 'impute.hist']),
   'num.trees': randint(1, 2000),
   'respect.unordered.factors': choice(['ignore', 'order', 'partition']),
   'sample.fraction': uniform(0.1, 1.0),
   'splitrule': choice(['gini', 'extratrees']),
   'num.random.splits': randint(1, 100),
}
fidelity_space = {
   'repl': randint(1, 10),
   'trainsize': uniform(0.03, 1.0),
}
```

### rbv2_xgboost

```python
instances = ['16', '40923', '41143', '470', '1487', '40499', '40966', '41164', '1497', '40975', '1461', '41278', '11', '54', '300', '40984', '31', '1067', '1590', '40983', '41163', '41165', '182', '1220', '41159', '41169', '42', '188', '1457', '1480', '6332', '181', '1479', '40670', '40536', '41138', '41166', '6', '14', '29', '458', '1056', '1462', '1494', '40701', '12', '1493', '44', '307', '334', '40982', '41142', '38', '1050', '469', '23381', '41157', '15', '4541', '23', '4134', '40927', '40981', '41156', '3', '1049', '40900', '1063', '23512', '40979', '1040', '1068', '41161', '22', '1489', '41027', '24', '4135', '23517', '1053', '1468', '312', '377', '1515', '18', '1476', '1510', '41162', '28', '375', '1464', '40685', '40996', '41146', '41216', '40668', '41212', '32', '60', '4538', '40496', '41150', '37', '46', '554', '1475', '1485', '1501', '1111', '4534', '41168', '151', '4154', '40978', '40994', '50', '1478', '1486', '40498']

metrics = ['acc', 'bac', 'auc', 'brier', 'f1', 'logloss', 'timetrain', 'timepredict', 'memory']

config_space = {
   'alpha': loguniform(0.0009118819655545162, 1096.6331584284585),
   'booster': choice(['gblinear', 'gbtree', 'dart']),
   'lambda': loguniform(0.0009118819655545162, 1096.6331584284585),
   'nrounds': lograndint(7, 2981),
   'num.impute.selected.cpo': choice(['impute.mean', 'impute.median', 'impute.hist']),
   'subsample': uniform(0.1, 1.0),
   'colsample_bylevel': uniform(0.01, 1.0),
   'colsample_bytree': uniform(0.01, 1.0),
   'eta': loguniform(0.0009118819655545162, 1.0),
   'gamma': loguniform(4.5399929762484854e-05, 7.38905609893065),
   'max_depth': randint(1, 15),
   'min_child_weight': loguniform(2.718281828459045, 148.4131591025766),
   'rate_drop': uniform(0.0, 1.0),
   'skip_drop': uniform(0.0, 1.0),
}
fidelity_space = {
   'repl': randint(1, 10),
   'trainsize': uniform(0.03, 1.0),
}
```

### rbv2_svm

```python
instances = ['40981', '4134', '1220', '40978', '40966', '40536', '41156', '458', '41157', '40975', '40994', '1468', '6332', '40670', '151', '1475', '1476', '1478', '1479', '41212', '1480', '1053', '1067', '1056', '12', '1487', '1068', '32', '470', '312', '38', '40982', '50', '41216', '307', '40498', '181', '1464', '41164', '16', '1461', '41162', '6', '14', '1494', '54', '375', '1590', '23', '41163', '1111', '41027', '40668', '41138', '4135', '4538', '40496', '4534', '40900', '1457', '11', '1462', '41142', '40701', '29', '37', '23381', '188', '41143', '1063', '3', '18', '40979', '22', '1515', '334', '24', '1493', '28', '1050', '1049', '40984', '40685', '42', '44', '46', '1040', '41146', '377', '40499', '1497', '60', '40983', '4154', '469', '31', '41278', '1489', '1501', '15', '300', '1485', '1486', '1510', '182', '41169']

metrics = ['acc', 'bac', 'auc', 'brier', 'f1', 'logloss', 'timetrain', 'timepredict', 'memory']

config_space = {
   'cost': loguniform(4.5399929762484854e-05, 22026.465794806718),
   'kernel': choice(['linear', 'polynomial', 'radial']),
   'num.impute.selected.cpo': choice(['impute.mean', 'impute.median', 'impute.hist']),
   'tolerance': loguniform(4.5399929762484854e-05, 2.0),
   'degree': randint(2, 5),
   'gamma': loguniform(4.5399929762484854e-05, 22026.465794806718),
}
fidelity_space = {
   'repl': randint(1, 10),
   'trainsize': uniform(0.03, 1.0),
}
```

### rbv2_aknn

```python
instances = ['41138', '40981', '4134', '40927', '1220', '4154', '41163', '40996', '4538', '40978', '375', '1111', '40496', '40966', '41150', '4534', '40900', '40536', '41156', '1590', '1457', '458', '469', '41157', '11', '1461', '1462', '1464', '15', '40975', '41142', '40701', '40994', '23', '1468', '40668', '29', '31', '6332', '37', '4541', '40670', '23381', '151', '188', '41164', '1475', '1476', '41159', '1478', '41169', '23512', '1479', '41212', '1480', '300', '41168', '41143', '1053', '41027', '1067', '1063', '41162', '3', '6', '1485', '1056', '12', '14', '16', '18', '40979', '22', '1515', '554', '334', '24', '1486', '23517', '41278', '1493', '28', '1487', '1068', '1050', '1049', '32', '1489', '470', '1494', '41161', '41165', '182', '312', '40984', '1501', '40685', '38', '42', '44', '46', '40982', '1040', '41146', '377', '40499', '50', '54', '41216', '41166', '307', '1497', '60', '1510', '40983', '40498', '181', '40923']

metrics = ['acc', 'bac', 'auc', 'brier', 'f1', 'logloss', 'timetrain', 'timepredict', 'memory']

config_space = {
   'M': randint(18, 50),
   'distance': choice(['l2', 'cosine', 'ip']),
   'ef': lograndint(7, 403),
   'ef_construction': lograndint(7, 1097),
   'k': randint(1, 50),
   'num.impute.selected.cpo': choice(['impute.mean', 'impute.median', 'impute.hist']),
}
fidelity_space = {
   'repl': randint(1, 10),
   'trainsize': uniform(0.03, 1.0),
}
```
