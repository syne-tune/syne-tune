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
Example that reproduces the LSTM on WikiText2 benchmark from AutoGluonExperiments repo
"""
import os
import argparse
import logging
import time
import math

from syne_tune import Reporter
from syne_tune.config_space import randint, uniform, loguniform, add_to_argparse
from benchmarking.utils import (
    resume_from_checkpointed_model,
    checkpoint_model_at_rung_level,
    add_checkpointing_to_argparse,
    pytorch_load_save_functions,
    parse_bool,
)


BATCH_SIZE_LOWER = 8

BATCH_SIZE_UPPER = 256

BATCH_SIZE_KEY = "batch_size"

METRIC_NAME = "objective"

RESOURCE_ATTR = "epoch"

ELAPSED_TIME_ATTR = "elapsed_time"


_config_space = {
    "lr": loguniform(1, 50),
    "dropout": uniform(0, 0.99),
    BATCH_SIZE_KEY: randint(BATCH_SIZE_LOWER, BATCH_SIZE_UPPER),
    "clip": uniform(0.1, 2),
    "lr_factor": loguniform(1, 100),
}


DATASET_PATH = "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/"


def download_data(root):
    import urllib

    path = os.path.join(root, "wikitext-2")
    for fname in ("train.txt", "valid.txt"):
        fh = os.path.join(path, fname)
        if not os.path.exists(fh):
            os.makedirs(path, exist_ok=True)
            urllib.request.urlretrieve(DATASET_PATH + fname, fh)


class Dictionary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus:
    def __init__(self, root):
        self.dictionary = Dictionary()
        # Make sure files are present locally
        download_data(root)
        path = os.path.join(root, "wikitext-2")
        self.train = self.tokenize(path, "train.txt")
        self.valid = self.tokenize(path, "valid.txt")
        # self.test = self.tokenize(path, 'test.txt')

    def tokenize(self, path, fname):
        """Tokenizes a text file."""
        assert fname in {"train.txt", "valid.txt", "test.txt"}
        fh = os.path.join(path, fname)
        # Add words to the dictionary
        with open(fh, "r", encoding="utf8") as f:
            for line in f:
                words = line.split() + ["<eos>"]
                for word in words:
                    self.dictionary.add_word(word)
        # Tokenize file content
        with open(fh, "r", encoding="utf8") as f:
            idss = []
            for line in f:
                words = line.split() + ["<eos>"]
                ids = []
                for word in words:
                    ids.append(self.dictionary.word2idx[word])
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)
        return ids


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def objective(config):
    # print(args)
    model_type = "rnn"
    emsize = 200
    nhid = emsize
    nlayers = 2
    eval_batch_size = 10
    bptt = 35
    tied = True
    seed = np.random.randint(10000)
    # log_interval = 200
    # save = "./model.pt"
    nhead = 2
    dropout = config["dropout"]
    batch_size = config["batch_size"]
    clip = config["clip"]
    lr_factor = config["lr_factor"]
    report_current_best = parse_bool(config["report_current_best"])
    trial_id = config.get("trial_id")
    debug_log = trial_id is not None
    if debug_log:
        print("Trial {}: Starting evaluation".format(trial_id), flush=True)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    #######################################################################
    # Load data
    #######################################################################
    path = config["dataset_path"]
    os.makedirs(path, exist_ok=True)
    # Lock protection is needed for backends which run multiple worker
    # processes on the same instance
    lock_path = os.path.join(path, "lock")
    lock = SoftFileLock(lock_path)
    try:
        with lock.acquire(timeout=120, poll_intervall=1):
            corpus = Corpus(config["dataset_path"])
    except Timeout:
        print(
            "WARNING: Could not obtain lock for dataset files. Trying anyway...",
            flush=True,
        )
        corpus = Corpus(config["dataset_path"])

    # Do not want to count the time to download the dataset, which can be
    # substantial the first time
    ts_start = time.time()
    report = Reporter()

    def batchify(data, bsz):
        # Work out how cleanly we can divide the dataset into bsz parts.
        nbatch = data.size(0) // bsz
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, nbatch * bsz)
        # Evenly divide the data across the bsz batches.
        data = data.view(bsz, -1).t().contiguous()
        return data.to(device)

    train_data = batchify(corpus.train, batch_size)
    val_data = batchify(corpus.valid, eval_batch_size)
    # test_data = batchify(corpus.test, eval_batch_size)

    #######################################################################
    # Build the model
    #######################################################################
    ntokens = len(corpus.dictionary)
    if model_type == "transformer":
        model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(
            device
        )
    else:
        model = RNNModel("LSTM", ntokens, emsize, nhid, nlayers, dropout, tied).to(
            device
        )
    criterion = nn.CrossEntropyLoss()

    def get_batch(source, i):
        seq_len = min(bptt, len(source) - 1 - i)
        data = source[i : i + seq_len]
        target = source[i + 1 : i + 1 + seq_len].view(-1)
        return data, target

    def evaluate(model, corpus, criterion, data_source):
        # Turn on evaluation mode which disables dropout.
        model.eval()
        total_loss = 0.0
        ntokens = len(corpus.dictionary)
        if model_type != "transformer":
            hidden = model.init_hidden(eval_batch_size)
        with torch.no_grad():
            for i in range(0, data_source.size(0) - 1, bptt):
                data, targets = get_batch(data_source, i)
                if model_type == "transformer":
                    output = model(data)
                else:
                    output, hidden = model(data, hidden)
                    hidden = repackage_hidden(hidden)
                output_flat = output.view(-1, ntokens)
                total_loss += len(data) * criterion(output_flat, targets).item()
        return total_loss / (len(data_source) - 1)

    def train(model, corpus, criterion, train_data, lr, batch_size, clip):
        # Turn on training mode which enables dropout.
        model.train()
        # total_loss = 0.
        # start_time = time.time()
        ntokens = len(corpus.dictionary)
        if model_type != "transformer":
            hidden = model.init_hidden(batch_size)
        for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
            data, targets = get_batch(train_data, i)
            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            model.zero_grad()
            if model_type == "transformer":
                output = model(data)
            else:
                hidden = repackage_hidden(hidden)
                output, hidden = model(data, hidden)
            loss = criterion(output.view(-1, ntokens), targets)
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            for p in model.parameters():
                p.data.add_(p.grad.data, alpha=-lr)

            # total_loss += loss.item()
            # if batch % log_interval == 0 and batch > 0:
            #    cur_loss = total_loss / log_interval
            #    elapsed = time.time() - start_time
            #    print('| {:5d}/{:5d} batches | lr {:02.5f} | ms/batch {:5.2f} | '
            #          'loss {:5.4f} | ppl {:8.2f}'.format(
            #        batch, len(train_data) // bptt, lr,
            #        elapsed * 1000 / log_interval, cur_loss, math.exp(cur_loss)))
            #    total_loss = 0
            #    start_time = time.time()

    # Checkpointing
    # Note that `lr` and `best_val_loss` are also part of the state to be
    # checkpointed. In order for things to work out, we keep them in a
    # dict (otherwise, they'd not be mutable in `load_model_fn`,
    # `save_model_fn`.
    mutable_state = {"lr": config["lr"], "best_val_loss": None}

    load_model_fn, save_model_fn = pytorch_load_save_functions(
        {"model": model}, mutable_state
    )

    # Resume from checkpoint (optional)
    resume_from = resume_from_checkpointed_model(config, load_model_fn)

    # Loop over epochs.
    for epoch in range(resume_from + 1, config["epochs"] + 1):
        train(
            model, corpus, criterion, train_data, mutable_state["lr"], batch_size, clip
        )
        val_loss = evaluate(model, corpus, criterion, val_data)

        val_loss = np.clip(val_loss, 1e-10, 10)
        # print('-' * 89)
        # print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
        #      'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
        #                                 val_loss, math.exp(val_loss)))
        # print('-' * 89)
        elapsed_time = time.time() - ts_start

        if not np.isfinite(val_loss):
            val_loss = 7

        best_val_loss = mutable_state["best_val_loss"]
        if not best_val_loss or val_loss < best_val_loss:
            best_val_loss = val_loss
            mutable_state["best_val_loss"] = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            mutable_state["lr"] /= lr_factor

        # Feed the score back back to Tune.
        _loss = best_val_loss if report_current_best else val_loss
        objective = -math.exp(_loss)
        report_kwargs = {
            RESOURCE_ATTR: epoch,
            METRIC_NAME: objective,
            ELAPSED_TIME_ATTR: elapsed_time,
        }
        report(**report_kwargs)

        # Write checkpoint (optional)
        checkpoint_model_at_rung_level(config, save_model_fn, epoch)

        if debug_log:
            print(
                "Trial {}: epoch = {}, objective = {:.3f}, elapsed_time = {:.2f}".format(
                    trial_id, epoch, objective, elapsed_time
                ),
                flush=True,
            )


if __name__ == "__main__":
    # Benchmark-specific imports are done here, in order to avoid import
    # errors if the dependencies are not installed (such errors should happen
    # only when the code is really called)
    from io import open
    import numpy as np
    from filelock import SoftFileLock, Timeout
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    # References to superclasses require torch and torch.nn to be defined here

    # Temporarily leave PositionalEncoding module here. Will be moved somewhere else.
    class PositionalEncoding(nn.Module):
        r"""Inject some information about the relative or absolute position of the tokens
            in the sequence. The positional encodings have the same dimension as
            the embeddings, so that the two can be summed. Here, we use sine and cosine
            functions of different frequencies.
        .. math::
            \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
            \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
            \text{where pos is the word position and i is the embed idx)
        Args:
            d_model: the embed dim (required).
            dropout: the dropout value (default=0.1).
            max_len: the max. length of the incoming sequence (default=5000).
        Examples:
            >>> pos_encoder = PositionalEncoding(d_model)
        """

        def __init__(self, d_model, dropout=0.1, max_len=5000):
            super(PositionalEncoding, self).__init__()
            self.dropout = nn.Dropout(p=dropout)
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0).transpose(0, 1)
            self.register_buffer("pe", pe)

        def forward(self, x):
            r"""Inputs of forward function
            Args:
                x: the sequence fed to the positional encoder model (required).
            Shape:
                x: [sequence length, batch size, embed dim]
                output: [sequence length, batch size, embed dim]
            Examples:
                >>> output = pos_encoder(x)
            """
            x = x + self.pe[: x.size(0), :]
            return self.dropout(x)

    class TransformerModel(nn.Module):
        """Container module with an encoder, a recurrent or transformer module, and a decoder."""

        def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
            super(TransformerModel, self).__init__()
            try:
                from torch.nn import TransformerEncoder, TransformerEncoderLayer
            except:
                raise ImportError(
                    "TransformerEncoder module does not exist in PyTorch 1.1 or lower."
                )
            self.model_type = "Transformer"
            self.src_mask = None
            self.pos_encoder = PositionalEncoding(ninp, dropout)
            encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
            self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
            self.encoder = nn.Embedding(ntoken, ninp)
            self.ninp = ninp
            self.decoder = nn.Linear(ninp, ntoken)
            self.init_weights()

        def _generate_square_subsequent_mask(self, sz):
            mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
            mask = (
                mask.float()
                .masked_fill(mask == 0, float("-inf"))
                .masked_fill(mask == 1, float(0.0))
            )
            return mask

        def init_weights(self):
            initrange = 0.1
            self.encoder.weight.data.uniform_(-initrange, initrange)
            self.decoder.bias.data.zero_()
            self.decoder.weight.data.uniform_(-initrange, initrange)

        def forward(self, src, has_mask=True):
            if has_mask:
                device = src.device
                if self.src_mask is None or self.src_mask.size(0) != len(src):
                    mask = self._generate_square_subsequent_mask(len(src)).to(device)
                    self.src_mask = mask
            else:
                self.src_mask = None
            src = self.encoder(src) * math.sqrt(self.ninp)
            src = self.pos_encoder(src)
            output = self.transformer_encoder(src, self.src_mask)
            output = self.decoder(output)
            return F.log_softmax(output, dim=-1)

    class RNNModel(nn.Module):
        """Container module with an encoder, a recurrent module, and a decoder."""

        def __init__(
            self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False
        ):
            super(RNNModel, self).__init__()
            self.drop = nn.Dropout(dropout)
            self.encoder = nn.Embedding(ntoken, ninp)
            if rnn_type in ["LSTM", "GRU"]:
                self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
            else:
                try:
                    nonlinearity = {"RNN_TANH": "tanh", "RNN_RELU": "relu"}[rnn_type]
                except KeyError:
                    raise ValueError(
                        """An invalid option for `--model` was supplied,
                                     options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']"""
                    )
                self.rnn = nn.RNN(
                    ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout
                )
            self.decoder = nn.Linear(nhid, ntoken)

            # Optionally tie weights as in:
            # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
            # https://arxiv.org/abs/1608.05859
            # and
            # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
            # https://arxiv.org/abs/1611.01462
            if tie_weights:
                if nhid != ninp:
                    raise ValueError(
                        "When using the tied flag, nhid must be equal to emsize"
                    )
                self.decoder.weight = self.encoder.weight
            self.init_weights()
            self.rnn_type = rnn_type
            self.nhid = nhid
            self.nlayers = nlayers

        def init_weights(self):
            initrange = 0.1
            self.encoder.weight.data.uniform_(-initrange, initrange)
            self.decoder.bias.data.zero_()
            self.decoder.weight.data.uniform_(-initrange, initrange)

        def forward(self, input, hidden):
            emb = self.drop(self.encoder(input))
            output, hidden = self.rnn(emb, hidden)
            output = self.drop(output)
            decoded = self.decoder(output)
            return decoded, hidden

        def init_hidden(self, bsz):
            weight = next(self.parameters())
            if self.rnn_type == "LSTM":
                return (
                    weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid),
                )
            else:
                return weight.new_zeros(self.nlayers, bsz, self.nhid)

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--report_current_best", type=str, default="False")
    parser.add_argument("--trial_id", type=str)
    add_to_argparse(parser, _config_space)
    add_checkpointing_to_argparse(parser)

    args, _ = parser.parse_known_args()

    objective(config=vars(args))
