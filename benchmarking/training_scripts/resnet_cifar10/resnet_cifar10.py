import os
import argparse
import logging
import time
from pathlib import Path

try:
    # Benchmark-specific imports are done here, in order to avoid import
    # errors if the dependencies are not installed (such errors should happen
    # only when the code is really called)
    from filelock import SoftFileLock, Timeout
    import numpy as np
    from tqdm import tqdm
    import torch
    import torch.nn.functional as F
    from torch.utils.data.sampler import SubsetRandomSampler
    from torchvision import datasets, transforms
    from torchvision.models import resnet18
except ImportError:
    logging.info(
        f"Please install benchmark-specific dependencies ({Path(__file__).parent / 'requirements.txt'})"
    )

from syne_tune import Reporter
from syne_tune.config_space import randint, uniform, loguniform, add_to_argparse
from syne_tune.utils import (
    resume_from_checkpointed_model,
    checkpoint_model_at_rung_level,
    add_checkpointing_to_argparse,
    pytorch_load_save_functions,
)


BATCH_SIZE_LOWER = 8

BATCH_SIZE_UPPER = 256

BATCH_SIZE_KEY = "batch_size"

METRIC_NAME = "objective"

RESOURCE_ATTR = "epoch"

MAX_RESOURCE_ATTR = "epochs"

ELAPSED_TIME_ATTR = "elapsed_time"


_config_space = {
    BATCH_SIZE_KEY: randint(BATCH_SIZE_LOWER, BATCH_SIZE_UPPER),
    "momentum": uniform(0, 0.99),
    "weight_decay": loguniform(1e-5, 1e-3),
    "lr": loguniform(1e-3, 0.1),
}


# ATTENTION: train_dataset, valid_dataset are both based on the CIFAR10
# training set, but train_dataset uses data augmentation. Make sure to
# only use disjoint parts for training and validation further down.
def get_CIFAR10(root):
    input_size = 32
    num_classes = 10
    normalize = [(0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)]
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(*normalize),
        ]
    )
    local_path = os.path.join(root, "CIFAR10")
    train_dataset = datasets.CIFAR10(
        local_path, train=True, transform=train_transform, download=True
    )

    valid_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(*normalize),
        ]
    )
    valid_dataset = datasets.CIFAR10(
        local_path, train=True, transform=valid_transform, download=True
    )

    return input_size, num_classes, train_dataset, valid_dataset


def train(model, train_loader, optimizer):
    model.train()
    total_loss = []
    for data, target in tqdm(train_loader):
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()
        optimizer.zero_grad()
        prediction = model(data)
        loss = F.nll_loss(prediction, target)
        loss.backward()
        optimizer.step()
        total_loss.append(loss.item())
    avg_loss = sum(total_loss) / len(total_loss)


def valid(model, valid_loader):
    model.eval()
    loss = 0
    correct = 0
    for data, target in valid_loader:
        with torch.no_grad():
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
            prediction = model(data)
            loss += F.nll_loss(prediction, target, reduction="sum")
            prediction = prediction.max(1)[1]
            correct += prediction.eq(target.view_as(prediction)).sum().item()
    n_valid = len(valid_loader.sampler)
    loss /= n_valid
    valid_error = correct / n_valid
    return loss, valid_error


def _download_data(config):
    path = config["dataset_path"]
    os.makedirs(path, exist_ok=True)
    # Lock protection is needed for backends which run multiple worker
    # processes on the same instance
    lock_path = os.path.join(path, "lock")
    lock = SoftFileLock(lock_path)
    try:
        with lock.acquire(timeout=120, poll_intervall=1):
            input_size, num_classes, train_dataset, valid_dataset = get_CIFAR10(
                root=path
            )
    except Timeout:
        print(
            "WARNING: Could not obtain lock for dataset files. Trying anyway...",
            flush=True,
        )
        input_size, num_classes, train_dataset, valid_dataset = get_CIFAR10(root=path)
    return input_size, num_classes, train_dataset, valid_dataset


def _create_data_loaders(config, train_dataset, valid_dataset):
    indices = list(range(train_dataset.data.shape[0]))
    train_idx, valid_idx = indices[:40000], indices[40000:]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config[BATCH_SIZE_KEY],
        # shuffle=True,
        num_workers=0,
        sampler=train_sampler,
        pin_memory=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=128,
        # shuffle=False,
        num_workers=0,
        sampler=valid_sampler,
        pin_memory=True,
    )
    return train_loader, valid_loader


def _create_training_objects(config):
    model = Model()
    if torch.cuda.is_available():
        model = model.cuda()
        device = torch.device("cuda")
        model = torch.nn.DataParallel(
            model, device_ids=[i for i in range(config["num_gpus"])]
        ).to(device)
    milestones = [25, 40]
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config["lr"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=0.1
    )
    return model, optimizer, scheduler


def objective(config):
    torch.manual_seed(np.random.randint(10000))
    # Download data, setup data loaders
    input_size, num_classes, train_dataset, valid_dataset = _download_data(config)
    train_loader, valid_loader = _create_data_loaders(
        config, train_dataset, valid_dataset
    )
    # Do not want to count the time to download the dataset, which can be
    # substantial the first time
    ts_start = time.time()
    report = Reporter()
    # Create model, optimizer, LR scheduler
    model, optimizer, scheduler = _create_training_objects(config)
    # Checkpointing for PyTorch model
    load_model_fn, save_model_fn = pytorch_load_save_functions(
        {"model": model, "optimizer": optimizer, "lr_scheduler": scheduler}
    )
    # Resume from checkpoint (optional)
    resume_from = resume_from_checkpointed_model(config, load_model_fn)

    for epoch in range(resume_from + 1, config[MAX_RESOURCE_ATTR] + 1):
        train(model, train_loader, optimizer)
        scheduler.step()
        elapsed_time = time.time() - ts_start
        # Write checkpoint (optional)
        checkpoint_model_at_rung_level(config, save_model_fn, epoch)
        # Evaluate and send metrics back to Syne Tune
        _, valid_error = valid(model, valid_loader)
        report(
            **{
                RESOURCE_ATTR: epoch,
                METRIC_NAME: valid_error,
                ELAPSED_TIME_ATTR: elapsed_time,
            }
        )


if __name__ == "__main__":
    # Superclass reference torch.nn.Module requires torch to be defined
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.resnet = resnet18(pretrained=False, num_classes=10)
            self.resnet.conv1 = torch.nn.Conv2d(
                3, 64, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.resnet.maxpool = torch.nn.Identity()

        def forward(self, x):
            x = self.resnet(x)
            x = F.log_softmax(x, dim=1)
            return x

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(f"--{MAX_RESOURCE_ATTR}", type=int, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--num_gpus", type=int, default=1)
    add_to_argparse(parser, _config_space)
    add_checkpointing_to_argparse(parser)

    args, _ = parser.parse_known_args()

    objective(config=vars(args))
