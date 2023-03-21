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
import argparse
import os
import time
import logging
import math
from pathlib import Path

try:
    # Benchmark-specific imports are done here, in order to avoid import
    # errors if the dependencies are not installed (such errors should happen
    # only when the code is really called)
    import numpy as np
    from filelock import SoftFileLock, Timeout
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    logging.info(
        f"Please install benchmark-specific dependencies ({Path(__file__).parent / 'requirements.txt'})"
    )
try:
    from apex import amp
except ImportError:
    print("Failed to import apex. You can still train with --precision {float|double}.")

from syne_tune.report import Reporter
from syne_tune.config_space import randint, uniform, loguniform, add_to_argparse
from syne_tune.utils import (
    resume_from_checkpointed_model,
    checkpoint_model_at_rung_level,
    add_checkpointing_to_argparse,
    pytorch_load_save_functions,
)


BATCH_SIZE_LOWER = 16

BATCH_SIZE_UPPER = 48

BATCH_SIZE_KEY = "batch_size"

METRIC_NAME = "val_loss"

RESOURCE_ATTR = "epoch"

MAX_RESOURCE_ATTR = "epochs"

ELAPSED_TIME_ATTR = "elapsed_time"


_config_space = {
    "lr": loguniform(1e-6, 1e-3),
    "dropout": uniform(0, 0.99),
    BATCH_SIZE_KEY: randint(BATCH_SIZE_LOWER, BATCH_SIZE_UPPER),
    "momentum": uniform(0, 0.99),
    "clip": uniform(0, 1),
}


DATASET_PATH = "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/"


def download_data(root):
    import urllib

    path = os.path.join(root, "wikitext-2")
    for fname in ("train.txt", "valid.txt", "test.txt"):
        fh = os.path.join(path, fname)
        if not os.path.exists(fh):
            os.makedirs(path, exist_ok=True)
            urllib.request.urlretrieve(DATASET_PATH + fname, fh)


class Dictionary(object):
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


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = None
        self.valid = None
        self.test = None
        if not self.load_cache(path):
            self.train = self.tokenize(os.path.join(path, "train.txt"))
            self.valid = self.tokenize(os.path.join(path, "valid.txt"))
            self.test = self.tokenize(os.path.join(path, "test.txt"))
            self.save_cache(path)

    def load_cache(self, path):
        for cache in ["dict.pt", "train.pt", "valid.pt", "test.pt"]:
            cache_path = os.path.join(path, cache)
            if not os.path.exists(cache_path):
                return False
        self.dictionary = torch.load(os.path.join(path, "dict.pt"))
        self.train = torch.load(os.path.join(path, "train.pt"))
        self.valid = torch.load(os.path.join(path, "valid.pt"))
        self.test = torch.load(os.path.join(path, "test.pt"))
        return True

    def save_cache(self, path):
        torch.save(self.dictionary, os.path.join(path, "dict.pt"))
        torch.save(self.train, os.path.join(path, "train.pt"))
        torch.save(self.valid, os.path.join(path, "valid.pt"))
        torch.save(self.test, os.path.join(path, "test.pt"))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, "r", encoding="utf8") as f:
            for line in f:
                words = line.split() + ["<eos>"]
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, "r", encoding="utf8") as f:
            idss = []
            for line in f:
                words = line.split() + ["<eos>"]
                ids = []
                for word in words:
                    ids.append(self.dictionary.word2idx[word])
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)

        return ids


def get_batch(source, i, bptt):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i : i + seq_len]
    target = source[i + 1 : i + 1 + seq_len].view(-1)
    return data, target


def batchloader(train_data, bptt):
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        yield get_batch(train_data, i, bptt)


def batchify(data, bsz, device):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)


def setprec(t, precision):
    if precision == "half":
        # do nothing since this is handled by AMP
        return t
    elif precision == "float":
        return t.float()
    elif precision == "double":
        return t.double()
    else:
        raise ValueError(f"invalid precision string {precision}")


def _download_data(config):
    path = config["input_data_dir"]
    os.makedirs(path, exist_ok=True)
    # Lock protection is needed for backends which run multiple worker
    # processes on the same instance
    lock_path = os.path.join(path, "lock")
    lock = SoftFileLock(lock_path)
    try:
        with lock.acquire(timeout=120, poll_intervall=1):
            # Make sure files are present locally
            download_data(path)
            corpus = Corpus(os.path.join(path, "wikitext-2"))
    except Timeout:
        print(
            "WARNING: Could not obtain lock for dataset files. Trying anyway...",
            flush=True,
        )
        # Make sure files are present locally
        download_data(path)
        corpus = Corpus(os.path.join(path, "wikitext-2"))
    return corpus


def objective(config):
    eval_batch_size = 10

    # Set the random seed manually for reproducibility.
    torch.manual_seed(config["seed"])
    use_cuda = config["use_cuda"]
    if torch.cuda.is_available() and not use_cuda:
        print("WARNING: You have a CUDA device, so you should run with --use-cuda 1")
    device = torch.device("cuda" if use_cuda else "cpu")

    #######################################################################
    # Load data
    #######################################################################
    corpus = _download_data(config)
    train_data = batchify(corpus.train, config["batch_size"], device)
    val_data = batchify(corpus.valid, eval_batch_size, device)

    # Do not want to count the time to download the dataset, which can be
    # substantial the first time
    ts_start = time.time()
    report = Reporter()

    #######################################################################
    # Build the model
    #######################################################################
    ntokens = len(corpus.dictionary)
    bptt = config["bptt"]
    precision = config["precision"]

    def evaluate(data_source):
        # Turn on evaluation mode which disables dropout.
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for i in range(0, data_source.size(0) - 1, bptt):
                data, targets = get_batch(data_source, i, bptt)
                output = model(data)
                output = output.view(-1, ntokens)
                total_loss += len(data) * criterion(output, targets).item()
        return total_loss / (len(data_source) - 1)

    def train(optimizer, epoch):
        # Turn on training mode which enables dropout.
        model.train()
        total_loss = 0.0
        epoch_loss = 0.0
        start_time = time.time()
        first_loss = None
        for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
            data, targets = get_batch(train_data, i, bptt)
            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.

            optimizer.zero_grad()
            output = model(data)
            output = output.view(-1, ntokens)
            loss = criterion(output, targets)
            if torch.isnan(loss):
                exit(0)
            if precision == "half":
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            clip = config["clip"]
            if clip > 0:
                # ``clip_grad_norm`` helps prevent the exploding gradient problem in RNNs / LSTMs.
                if precision == "half":
                    params = amp.master_params(optimizer)
                else:
                    params = model.parameters()
                torch.nn.utils.clip_grad_norm_(params, clip)

            optimizer.step()

            total_loss += loss.item()
            epoch_loss += len(data) * loss.item()

            log_interval = config["log_interval"]
            if batch % log_interval == 0 and batch > 0:
                cur_loss = total_loss / log_interval
                elapsed = time.time() - start_time
                print(
                    "| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.5f} | ms/batch {:5.2f} | "
                    "loss {:5.2f} | ppl {:8.2f}".format(
                        epoch,
                        batch,
                        len(train_data) // bptt,
                        config["lr"],
                        elapsed * 1000 / log_interval,
                        cur_loss,
                        np.exp(cur_loss),
                    )
                )
                total_loss = 0
                start_time = time.time()
                if first_loss is None:
                    first_loss = cur_loss

        return epoch_loss / (len(train_data) - 1), first_loss

    d_model = config["d_model"]
    model = TransformerModel(
        ntokens,
        ninp=d_model,
        nhead=config["nhead"],
        nhid=d_model * config["ffn_ratio"],
        nlayers=config["nlayers"],
        dropout=config["dropout"],
    )

    model = model.to(device)
    model = setprec(model, precision)
    criterion = nn.NLLLoss()

    if config["optimizer_name"] == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config["lr"],
            momentum=config["momentum"],
        )
    elif config["optimizer_name"] == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["lr"],
            betas=(config["momentum"], 0.999),
        )
    else:
        raise ValueError()

    # half-precision black magic
    if precision == "half":
        model, optimizer = amp.initialize(
            model, optimizer, opt_level="O1", min_loss_scale=0.0001, verbosity=0
        )

    # Checkpointing
    # Note that ``best_val_loss`` and ``logs`` are also part of the state to be
    # checkpointed. In order for things to work out, we keep them in a
    # dict (otherwise, they'd not be mutable in ``load_model_fn``,
    # ``save_model_fn``).
    mutable_state = {"best_val_loss": None, "logs": []}
    state_dict_objects = {
        "model": model,
        "optimizer": optimizer,
    }
    if precision == "half":
        state_dict_objects["amp"] = amp

    load_model_fn, save_model_fn = pytorch_load_save_functions(
        state_dict_objects=state_dict_objects,
        mutable_state=mutable_state,
    )

    # Resume from checkpoint (optional)
    resume_from = resume_from_checkpointed_model(config, load_model_fn)

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for epoch in range(resume_from + 1, config[MAX_RESOURCE_ATTR] + 1):
            epoch_start_time = time.time()
            train_loss, first_loss = train(optimizer, epoch)
            val_loss = evaluate(val_data)

            curr_ts = time.time()
            elapsed_time = curr_ts - ts_start
            duration = curr_ts - epoch_start_time
            print("-" * 89)
            print(
                "| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | "
                "valid ppl {:8.2f}".format(epoch, duration, val_loss, np.exp(val_loss))
            )
            print("-" * 89)
            mutable_state["logs"].append(
                dict(
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    first_loss=first_loss,
                    duration=duration,
                )
            )
            best_val_loss = mutable_state["best_val_loss"]
            if not best_val_loss or val_loss < best_val_loss:
                mutable_state["best_val_loss"] = val_loss

            # Write checkpoint (optional)
            checkpoint_model_at_rung_level(config, save_model_fn, epoch)

            # report validation loss back to Syne Tune
            report_kwargs = {
                RESOURCE_ATTR: epoch,
                METRIC_NAME: val_loss,
                ELAPSED_TIME_ATTR: elapsed_time,
            }
            report(**report_kwargs)

    except KeyboardInterrupt:
        print("-" * 89)
        print("Exiting from training early")


if __name__ == "__main__":
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
            except ImportError:
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

        @staticmethod
        def _generate_square_subsequent_mask(sz):
            mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
            mask = (
                mask.float()
                .masked_fill(mask == 0, float("-inf"))
                .masked_fill(mask == 1, float(0.0))
            )
            return mask

        def init_weights(self):
            initrange = 0.1
            nn.init.uniform_(self.encoder.weight, -initrange, initrange)
            nn.init.zeros_(self.decoder.bias)
            nn.init.uniform_(self.decoder.weight, -initrange, initrange)

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

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    parser = argparse.ArgumentParser(
        description="PyTorch Wikitext-2 Transformer Language Model",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--" + MAX_RESOURCE_ATTR, type=int, default=40, help="upper epoch limit"
    )
    parser.add_argument("--use_cuda", type=int, default=1)
    parser.add_argument(
        "--input_data_dir",
        type=str,
        default="./",
        help="location of the data corpus",
    )
    parser.add_argument(
        "--optimizer_name", type=str, default="sgd", choices=["sgd", "adam"]
    )
    parser.add_argument("--bptt", type=int, default=35, help="sequence length")
    parser.add_argument("--seed", type=int, default=1111, help="random seed")
    parser.add_argument(
        "--precision", type=str, default="float", help="float | double | half"
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=200,
        help="report interval",
    )
    # These could become hyperparameters as well (more like NAS)
    parser.add_argument("--d_model", type=int, default=256, help="width of the model")
    parser.add_argument(
        "--ffn_ratio", type=int, default=1, help="the ratio of d_ffn to d_model"
    )
    parser.add_argument("--nlayers", type=int, default=2, help="number of layers")
    parser.add_argument(
        "--nhead",
        type=int,
        default=2,
        help="the number of heads in the encoder/decoder of the transformer model",
    )
    add_to_argparse(parser, _config_space)
    add_checkpointing_to_argparse(parser)

    args, _ = parser.parse_known_args()
    args.use_cuda = bool(args.use_cuda)

    objective(config=vars(args))
