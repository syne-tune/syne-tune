# Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.
"""
DistilBERT fine-tuned on IMDB sentiment classification task
"""
import argparse
import logging
import time

from sagemaker_tune.report import Reporter
from sagemaker_tune.search_space import loguniform, add_to_argparse


_config_space = {
    'learning_rate': loguniform(1e-6, 1e-4),
    'weight_decay': loguniform(1e-6, 1e-4)
}


def distilbert_imdb_default_params(params=None):
    return {
        'max_resource_level': 15,
        'instance_type': 'ml.g4dn.xlarge',
        'num_workers': 4,
        'framework': 'HuggingFace',
        'framework_version': '4.4',
        'pytorch_version': '1.6',
        'dataset_path': './'
    }


def distilbert_imdb_benchmark(params):
    config_space = dict(
        _config_space,
        dataset_path=params['dataset_path'],
        max_steps=params['max_resource_level'])
    return {
        'script': __file__,
        'metric': 'accuracy',
        'mode': 'max',
        'resource_attr': 'step',
        'config_space': config_space,
    }


def download_data(config):
    train_dataset, eval_dataset = load_dataset(
        'imdb', split=['train', 'test'], cache_dir=config['dataset_path'])
    return train_dataset, eval_dataset


def prepare_data(config, train_dataset, eval_dataset, seed=42):
    # Subsample data
    train_dataset = train_dataset.shuffle(seed=seed).select(range(config['n_train_data']))
    eval_dataset = eval_dataset.shuffle(seed=seed).select(range(config['n_eval_data']))

    # Tokenize data
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

    def tokenize(batch):
        return tokenizer(batch['text'], padding='max_length', truncation=True)

    train_dataset = train_dataset.map(tokenize, batched=True)
    eval_dataset = eval_dataset.map(tokenize, batched=True)

    return train_dataset, eval_dataset


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    metric = load_metric("accuracy")
    return metric.compute(predictions=predictions, references=labels)


def objective(config):
    trial_id = config.get('trial_id')
    debug_log = trial_id is not None

    # Download and prepare data
    train_dataset, eval_dataset = download_data(config)
    train_dataset, eval_dataset = prepare_data(config, train_dataset, eval_dataset)

    report = Reporter()

    # Do not want to count the time to download the dataset, which can be
    # substantial the first time
    ts_start = time.time()

    # Download model from Hugging Face model hub
    model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

    # Define training args
    training_args = TrainingArguments(
        output_dir='./',
        num_train_epochs=config['epochs'],
        per_device_train_batch_size=config['train_batch_size'],
        per_device_eval_batch_size=config['eval_batch_size'],
        evaluation_strategy='steps',
        eval_steps=config['eval_interval'] // config['train_batch_size'],
        learning_rate=float(config['learning_rate']),
        weight_decay=float(config['weight_decay']),
        load_best_model_at_end=True,
        metric_for_best_model='eval_accuracy',
        greater_is_better=True,
        # avoid filling disk
        save_strategy="no",
    )

    # Create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # add a callback so that accuracy is sent to Sagemaker Tune whenever it is computed
    class Callback(TrainerCallback):
        def __init__(self):
            self.step = 1

        def on_evaluate(self, args, state, control, metrics, **kwargs):
            # Feed the validation accuracy back to Tune
            report(step=self.step, accuracy=metrics['eval_accuracy'])
            self.step += 1

    trainer.add_callback(Callback())

    # Train model
    trainer.train()

    # Evaluate model
    eval_result = trainer.evaluate(eval_dataset=eval_dataset)
    eval_accuracy = eval_result['eval_accuracy']

    elapsed_time = time.time() - ts_start

    if debug_log:
        print("Trial {}: accuracy = {:.3f}, elapsed_time = {:.2f}".format(
            trial_id, eval_accuracy, elapsed_time), flush=True)


if __name__ == '__main__':
    # Benchmark-specific imports are done here, in order to avoid import
    # errors if the dependencies are not installed (such errors should happen
    # only when the code is really called)
    from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer
    from transformers import TrainerCallback
    from datasets import load_dataset, load_metric

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--eval_batch_size', type=int, default=32)
    parser.add_argument('--n_train_data', type=int, default=25000)
    parser.add_argument('--n_eval_data', type=int, default=5000)
    parser.add_argument('--eval_interval', type=int, default=5000)
    parser.add_argument('--trial_id', type=str)
    add_to_argparse(parser, _config_space)
    
    args, _ = parser.parse_known_args()

    objective(config=vars(args))
