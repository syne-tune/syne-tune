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
import torch
import argparse
import numpy as np

from dataclasses import dataclass
from typing import Optional, Union

from datasets import load_dataset

from transformers import (
    AutoModelForMultipleChoice,
    TrainingArguments,
    Trainer,
    AutoTokenizer,
)
from transformers.tokenization_utils_base import (
    PreTrainedTokenizerBase,
    PaddingStrategy,
)
from transformers.trainer_callback import TrainerCallback

from syne_tune.report import Reporter


class ReportBackMetrics(TrainerCallback):
    """
    This callback is used in order to report metrics back to Syne Tune, using a
    ``Reporter`` object.

    If ``test_dataset`` is given, we also compute and report test set metrics here.
    These are just for final evaluations. HPO must use validation metrics (in
    ``metrics`` passed to ``on_evaluate``).

    If ``additional_info`` is given, it is a static dict reported with each call.
    """

    def __init__(self, trainer):
        self.trainer = trainer
        self.report = Reporter()

    def on_evaluate(self, args, state, control, **kwargs):
        # Metrics on train and validation set:
        results = kwargs["metrics"].copy()
        results["step"] = state.global_step
        results["epoch"] = int(state.epoch)
        # Report results back to Syne Tune
        self.report(**results)


@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)]
            for feature in features
        ]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Un-flatten
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        # Add back labels
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch


def compute_metrics(eval_predictions):
    predictions, label_ids = eval_predictions
    preds = np.argmax(predictions, axis=1)
    return {"accuracy": (preds == label_ids).astype(np.float32).mean().item()}


def preprocess_function(examples):
    # Repeat each first sentence four times to go with the four possibilities of second sentences.
    first_sentences = [[context] * 4 for context in examples["sent1"]]
    # Grab all second sentences possible for each context.
    question_headers = examples["sent2"]
    ending_names = ["ending0", "ending1", "ending2", "ending3"]
    second_sentences = [
        [f"{header} {examples[end][i]}" for end in ending_names]
        for i, header in enumerate(question_headers)
    ]

    # Flatten everything
    first_sentences = sum(first_sentences, [])
    second_sentences = sum(second_sentences, [])

    # Tokenize
    tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=True)
    # Un-flatten
    return {
        k: [v[i : i + 4] for i in range(0, len(v), 4)]
        for k, v in tokenized_examples.items()
    }


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument("--lr_schedule", type=str, default="linear")
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    args, _ = parser.parse_known_args()

    model_checkpoint = "bert-base-uncased"

    datasets = load_dataset("swag", "regular")

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

    encoded_datasets = datasets.map(preprocess_function, batched=True)

    model = AutoModelForMultipleChoice.from_pretrained(model_checkpoint)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(model_checkpoint, n_params)
    model_name = model_checkpoint.split("/")[-1]

    training_args = TrainingArguments(
        f"{model_name}-finetuned-swag",
        evaluation_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_schedule,
        warmup_ratio=args.warmup_ratio,
        fp16=True,
        save_strategy="no",
    )

    trainer = Trainer(
        model,
        training_args,
        train_dataset=encoded_datasets["train"],
        eval_dataset=encoded_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForMultipleChoice(tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.add_callback(ReportBackMetrics(trainer=trainer))

    trainer.train()
