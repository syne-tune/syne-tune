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
BERT-like models for text classification
"""

from typing import Optional
import os
import time

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from tqdm.auto import tqdm

from utils import compute_metrics, clip_grad_norm, cross_entropy, pooling, \
    reinit_parameters

from sagemaker_tune.report import Reporter


class BERTClassificationModel(nn.Module):
    """
    A class for BERT-like text classification models
    """

    def __init__(self,
            model_name: str,
            n_classes: int,
            reinit_weights: bool = False):
        """
        Initialize a pre-trained BERT-like text classification model

        Args:
            model_name: The name of the model to load
            n_classes: The number of classes (i.e. the number of outputs for the classification head)
            reinit_weights: Should we randomly re-initialize the BERT weights?
        """

        super().__init__()

        # Check validity of parameters
        assert 'bert' in model_name, "Currently only BERT-like models are supported!"

        # Set attributes
        self.model_name = model_name
        self.n_classes = n_classes
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # Instantiate pre-trained BERT model and corresponding tokenizer
        self.bert_model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.n_layers = self.bert_model.config.num_hidden_layers

        if reinit_weights:
            # Re-initialize BERT model using the original initialization distribution
            for module in self.bert_model.modules():
                self.bert_model._init_weights(module)

        # Instantiate classification head
        self.n_outputs = self.n_classes 
        self.classifier = nn.Linear(self.bert_model.config.hidden_size, self.n_outputs).to(self.device)

    def prepare(
            self, reinit_pooler: bool = False,
            n_top_layers_to_reinit: int = 0):
        """
        Prepare the BERT model for fine-tuning.

        Args:
            reinit_pooler: Should we re-initialize the pooler?
            n_top_layers_to_reinit: The number of top layers to re-initialize randomly
        """
        if reinit_pooler or n_top_layers_to_reinit > 0:
            # Randomly re-initialize the parameters of the number of top layers specified
            print(f"Randomly re-initializing the top {n_top_layers_to_reinit} BERT layers.")
            reinit_parameters(self, reinit_pooler, n_top_layers_to_reinit)

    def fit(self,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler._LRScheduler,
            train_loader: torch.utils.data.DataLoader,
            eval_loader: torch.utils.data.DataLoader,
            n_epochs: int = 3,
            n_evals_per_epoch: int = 8,
            max_num_evaluations: Optional[int] = None,
            report: Optional[Reporter] = None,
            log_steps: int = 100,
            reinit_pooler: bool = False,
            n_top_layers_to_reinit: int = 0,
            eval_metric: str = 'acc',
            max_grad_norm: int = 1,
            checkpoint_config: Optional[dict] = None,
            temp: float = 1.0):
        """
        Custom training loop for fine-tuning BERT model. Supports advanced
        features like checkpoint and resume or re-initialization of top layers.

        Args:
            optimizer: The PyTorch optimizer to use
            scheduler: The PyTorch learning rate scheduler to use
            train_loader: The data loader for training
            eval_loader: The data loader for evaluation
            n_epochs: The number of epochs to train
            n_evals_per_epoch: The number of evaluations to make per training epoch
            max_num_evaluations: The training loop is stopped once this number
                of evaluations are done. If not given, this defaults to
                `n_epochs * n_evals_per_epoch`.
            report: The SageMaker Tune report method to call after every evaluation
            log_steps: The number of training steps in-between logging
            reinit_pooler: Should we re-initialize the pooler?
            n_top_layers_to_reinit: The number of top layers to re-initialize randomly
            eval_metric: The main metric to use for evaluation
            max_grad_norm: The maximum gradient norm to use for gradient clipping
            checkpoint_config: The configuration dictionary used for checkpointing
            temp: The temperature parameter for distillation
        """

        # Check the validity of the arguments passed
        assert not (n_top_layers_to_reinit > 0 and not reinit_pooler),\
                'Must also re-initialize pooler if top layers are re-initialized!'
        assert eval_metric in ['acc', 'f1'], f'Invalid evaluation metric: {eval_metric}!'
        if checkpoint_config is None:
            checkpoint_config = dict()
        if max_num_evaluations is None:
            max_num_evaluations = n_epochs * n_evals_per_epoch
        else:
            assert max_num_evaluations <= n_epochs * n_evals_per_epoch, \
                f"max_num_evaluations = {max_num_evaluations} > {n_epochs * n_evals_per_epoch} = n_epochs * n_evals_per_epoch"

        # Prepare the model for fine-tuning (e.g. freeze/drop/re-initalize certain layers)
        self.prepare(reinit_pooler, n_top_layers_to_reinit)

        # Define mutable state for checkpointing (for promotion-based HPO schedulers)
        mutable_state = {'best_eval_score': 0.0}
    
        resume_from = 0
        save_model_fn = None
        if report is not None:
            from benchmarks.checkpoint import pytorch_load_save_functions, \
                resume_from_checkpointed_model

            # Define load and save functions for checkpointing (for
            # promotion-based HPO schedulers)
            model_state = {
                'bert_model': self.bert_model,
                'classifier': self.classifier,
                'optimizer': optimizer,
                'scheduler': scheduler}
            load_model_fn, save_model_fn = pytorch_load_save_functions(
                model_state, mutable_state=mutable_state)
            # Resume from a previously saved checkpoint (if applicable)
            resume_from = resume_from_checkpointed_model(
                checkpoint_config, load_model_fn)

        # Define / initialize some variables
        self.train()
        optimizer.zero_grad()
        n_batches_total = len(train_loader) * n_epochs
        n_batches_between_evals = len(train_loader) // n_evals_per_epoch
        n_batches_seen = 0
        n_evaluations = 0
        results = []

        # Run the training/fine-tuning for the specified number of epochs
        start_time = time.time()
        print(f"Step\tTime\tTrain loss\tTrain acc\tEval loss\tEval acc\tBest")
        stop_training = False
        for epoch in range(n_epochs):
            for batch in tqdm(train_loader, total=len(train_loader)):
                # Increase the counter for the number of batches we have seen/trained on so far
                n_batches_seen += 1

                # Determine if we would evaluate our model after this batch update:
                # we evaluate the model every `n_batches_between_evals` training steps/batches
                # OR when training is complete
                evaluate_model = False
                if n_batches_seen % n_batches_between_evals == 0 or n_batches_seen == n_batches_total:
                    evaluate_model = True
                    n_evaluations += 1

                # If we use a promotion-based HPO scheduler, we might want to
                # resume training from a previous checkpoint and therefore skip
                # all the training steps we have already done to obtain that
                # checkpoint; we need the second condition to skip the evaluation
                # we left off at (i.e to not do it twice)
                if n_evaluations < resume_from or (n_evaluations == resume_from and evaluate_model):
                    continue

                # Put the data from the current batch (i.e. the input ids, attention mask and labels) on the GPU
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                # Do forward pass, compute loss and do backward pass
                outputs = self(input_ids, attention_mask, labels, temp=temp)
                loss = outputs['loss']
                loss.backward()

                # Take an optimizer step; if we use branch rotation for TreeNet,
                # we use the accumulated gradients across all branches for the update
                clip_grad_norm(optimizer, max_grad_norm)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()

                # Compute train accuracy and store metrics
                if len(labels.shape) > 1:
                    # Convert soft to hard labels when doing distillation (to be able to compute the accuracy)
                    labels = labels.argmax(axis=-1)
                train_metrics = compute_metrics(outputs['preds'], labels, ['accuracy'])
                results.append({
                    'step': n_batches_seen,
                    'n_evaluations': n_evaluations,
                    'time': time.time() - start_time,
                    'train_loss': loss.item(),
                    'train_acc': train_metrics['accuracy'],
                    'eval_loss': 0.0,
                    'eval_acc': 0.0,
                    'eval_f1': 0.0,
                })

                if evaluate_model:
                    # Evaluate the model on the validation data
                    cuda_str = f" (GPU {os.environ['CUDA_VISIBLE_DEVICES']})" if 'CUDA_VISIBLE_DEVICES' in os.environ else ""
                    print(f"Evaluating model on validation data{cuda_str}...")
                    results[-1] = self.evaluate(eval_loader, results[-1], return_predictions=False)
                    self.train()
                    eval_score = results[-1][f'eval_{eval_metric}']

                    if report is not None:
                        from benchmarks.checkpoint import checkpoint_model_at_rung_level

                        # Report evaluation score to SageMaker Tune
                        report(**{eval_metric: eval_score, 'n_evaluations': n_evaluations})
                        # Write checkpoint (if applicable)
                        checkpoint_model_at_rung_level(
                            checkpoint_config, save_model_fn, n_evaluations)

                    if eval_score > mutable_state['best_eval_score']:
                        mutable_state['best_eval_score'] = eval_score

                # Log results every `log_steps` training steps/batches
                if n_batches_seen % log_steps == 0:
                    r = results[-1]
                    print(f"{r['step']}\t{r['time']:.1f}\t{r['train_loss']:.3f}\t\t{r['train_acc']:.3f}\t\t{r['eval_loss']:.3f}\t\t"\
                            f"{r['eval_acc']:.3f}\t\t{r['eval_f1']:.3f}\t\t{mutable_state['best_eval_score']:.3f}")

                if n_evaluations >= max_num_evaluations:
                    stop_training = True
                    break  # Step out of `max_num_evaluations` done

            if stop_training:
                break  # Step out of `max_num_evaluations` done

    def evaluate(self,
            eval_loader: torch.utils.data.DataLoader,
            results: dict = None,
            return_predictions: bool = False) -> dict:
        """
        Evaluate the BERT model (e.g. on a validation set)

        Args:
            eval_loader: The data loader for evaluation
            results: The dictionary for storing the evaluation metrics (i.e. loss, accuracy and F1-score)
            return_predictions: Should we also return the model's predictive distributions?

        Returns:
            The dictionary with the evaluation metrics
        """

        if results is None:
            results = {}

        self.eval()
        all_preds = []
        all_labels = []
        all_pred_distributions = []
        all_logits = []
        loss = 0
        with torch.no_grad():
            for batch in tqdm(eval_loader, total=len(eval_loader)):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                # Compute the model predictions
                outputs = self(input_ids, attention_mask, labels, return_predictions=return_predictions)
                loss += outputs['loss'].item()
                all_preds.append(outputs['preds'])
                all_labels.append(labels)
                if return_predictions:
                    all_pred_distributions.append(outputs['pred_distributions'])
                    all_logits.append(outputs['logits'])

        # Compute evaluation metrics (i.e. loss, accuracy and F1-score)
        results['eval_loss'] = loss / len(eval_loader)
        all_labels = torch.cat(all_labels)
        if len(all_labels.shape) > 1:
            # Convert soft to hard labels when doing distillation (to be able to compute the accuracy)
            all_labels = all_labels.argmax(axis=-1)
        eval_metrics = compute_metrics(torch.cat(all_preds), all_labels)
        results['eval_acc'] = eval_metrics['accuracy']
        results['eval_f1'] = eval_metrics['f1']
        if return_predictions:
            results['eval_predictions'] = torch.cat(all_pred_distributions)
            results['eval_logits'] = torch.cat(all_logits)
            results['eval_labels'] = all_labels.detach().cpu()

        return results

    def forward(self, input_ids, attention_mask, labels=None, use_bert_pooler=False, return_predictions=False, temp=1.0):
        """ Do a forward pass through a single (i.e non-ensembled) model """

        # Compute the input embeddings
        hidden_state = self.bert_model.embeddings(input_ids=input_ids)

        # Apply the transformer layers
        extended_attention_mask = self.bert_model.get_extended_attention_mask(attention_mask, input_ids.size(), input_ids.device)
        for layer in self.bert_model.encoder.layer:
            hidden_state = layer(hidden_state, extended_attention_mask)[0]

        # Apply pooling
        if use_bert_pooler and self.bert_model.pooler is not None:
            pooled_output = self.bert_model.pooler(hidden_state)
        else:
            pooled_output = pooling(hidden_state)

        if labels is None:
            return {'pooled_output': pooled_output}

        # Apply the classifier
        logits = self.classifier(pooled_output)

        # Compute the model predictions and the resulting loss value
        if self.n_outputs == 1:
            preds = (nn.Sigmoid()(logits.flatten()) > 0.5).float()
            loss = nn.BCEWithLogitsLoss()(logits.flatten(), labels.float())
        else:
            preds = logits.argmax(axis=-1)
            loss = cross_entropy(logits, labels, temp=temp)

        # Return the logits, loss, predictions and model outputs after pooling
        output = {'logits': logits, 'loss': loss, 'preds': preds, 'pooled_output': pooled_output}

        if return_predictions:
            # Also return the predictive distributions of the model on the data
            if self.n_outputs == 1:
                output['pred_distributions'] = nn.Sigmoid()(logits.flatten()).detach().cpu()
            else:
                output['pred_distributions'] = nn.Softmax(dim=-1)(logits).detach().cpu()

        return output

