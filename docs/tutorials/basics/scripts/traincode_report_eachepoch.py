import os
import argparse
import logging

from syne_tune.report import Reporter


# Boilerplate for objective

def download_data(config):
    path = os.path.join(config['dataset_path'], 'FashionMNIST')
    os.makedirs(path, exist_ok=True)
    # Lock protection is needed for backends which run multiple worker
    # processes on the same instance
    lock_path = os.path.join(path, 'lock')
    lock = SoftFileLock(lock_path)
    try:
        with lock.acquire(timeout=120, poll_intervall=1):
            data_train = datasets.FashionMNIST(
                root=path, train=True, download=True,
                transform=transforms.ToTensor())
    except Timeout:
        print(
            "WARNING: Could not obtain lock for dataset files. Trying anyway...",
            flush=True)
        data_train = datasets.FashionMNIST(
            root=path, train=True, download=True,
            transform=transforms.ToTensor())
    return data_train


def split_data(config, data_train):
    # We use 50000 samples for training and 10000 samples for validation
    indices = list(range(data_train.data.shape[0]))
    train_idx, valid_idx = indices[:50000], indices[50000:]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    batch_size = config['batch_size']
    train_loader = torch.utils.data.DataLoader(
        data_train, batch_size=batch_size, sampler=train_sampler,
        drop_last=True)
    valid_loader = torch.utils.data.DataLoader(
        data_train, batch_size=batch_size, sampler=valid_sampler,
        drop_last=True)
    return train_loader, valid_loader


def model_and_optimizer(config):
    n_units_1 = config['n_units_1']
    n_units_2 = config['n_units_2']
    dropout_1 = config['dropout_1']
    dropout_2 = config['dropout_2']
    learning_rate = config['learning_rate']
    weight_decay = config['weight_decay']
    # Define the network architecture
    comp_list = [
        nn.Linear(28 * 28, n_units_1),
        nn.Dropout(p=dropout_1),
        nn.ReLU(),
        nn.Linear(n_units_1, n_units_2),
        nn.Dropout(p=dropout_2),
        nn.ReLU(),
        nn.Linear(n_units_2, 10)]
    model = nn.Sequential(*comp_list)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    return {
        'model': model,
        'optimizer': optimizer,
        'criterion': criterion}


def train_model(config, state, train_loader):
    model = state['model']
    optimizer = state['optimizer']
    criterion = state['criterion']
    batch_size = config['batch_size']
    model.train()
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data.view(batch_size, -1))
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()


def validate_model(config, state, valid_loader):
    batch_size = config['batch_size']
    model = state['model']
    model.eval()
    correct = 0
    total = 0
    for data, target in valid_loader:
        output = model(data.view(batch_size, -1))
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
    return correct / total  # Validation accuracy


def objective(config):
    # Download data
    data_train = download_data(config)

    # Report results to Syne Tune
    report = Reporter()

    # Split into training and validation set
    train_loader, valid_loader = split_data(config, data_train)

    # Create model and optimizer
    state = model_and_optimizer(config)

    # Training loop
    for epoch in range(1, config['epochs'] + 1):
        train_model(config, state, train_loader)
        accuracy = validate_model(config, state, valid_loader)
        # Report validation accuracy to Syne Tune
        # [1]
        report(
            epoch=epoch,
            accuracy=accuracy)


if __name__ == '__main__':
    # Benchmark-specific imports are done here, in order to avoid import
    # errors if the dependencies are not installed (such errors should happen
    # only when the code is really called)
    from filelock import SoftFileLock, Timeout
    import torch
    import torch.nn as nn
    from torch.utils.data.sampler import SubsetRandomSampler
    from torchvision import datasets
    from torchvision import transforms

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--dataset_path', type=str, required=True)
    # Hyperparameters
    parser.add_argument('--n_units_1', type=int, required=True)
    parser.add_argument('--n_units_2', type=int, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--dropout_1', type=float, required=True)
    parser.add_argument('--dropout_2', type=float, required=True)
    parser.add_argument('--learning_rate', type=float, required=True)
    parser.add_argument('--weight_decay', type=float, required=True)

    args, _ = parser.parse_known_args()

    objective(config=vars(args))
