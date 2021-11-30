import bz2
import pickle
import pandas as pd
import numpy as np

from blackbox_repository.blackbox_tabular import serialize, BlackboxTabular
from blackbox_repository.conversion_scripts.utils import repository_path

from syne_tune import search_space
from syne_tune.util import catchtime


def str_to_list(arch_str):
    node_strs = arch_str.split('+')
    config = []
    for i, node_str in enumerate(node_strs):
        inputs = [x for x in node_str.split('|') if x != '']
        for xinput in inputs:
            assert len(xinput.split('~')) == 2, 'invalid input length : {:}'.format(xinput)
        inputs = (xi.split('~') for xi in inputs)

        config.extend(op for (op, idx) in inputs)

    return config


def convert_dataset(data, dataset):
    hp_cols = [f"hp_x{i}" for i in range(6)]

    hps = dict()

    for h in hp_cols:
        hps[h] = []

    n_hps = data['total_archs']

    for i in range(n_hps):
        config = str_to_list(data['arch2infos'][i]['200']['arch_str'])

        for j, hp in enumerate(config):
            hps[f"hp_x{j}"].append(hp)

    hyperparameters = pd.DataFrame(
        data=hps,
        columns=hp_cols
    )

    objective_names = [
        'valid_error',
        'train_error',
        'runtime',
        'latency',
        'flops',
        'params',

    ]

    fidelity_values = np.arange(1, 201)
    n_fidelities = len(fidelity_values)
    n_objectives = len(objective_names)
    n_seeds = 3

    objective_evaluations = np.empty((n_hps, n_seeds, n_fidelities, n_objectives)).astype('float32')
    name_index = {name: i for i, name in enumerate(objective_names)}

    def save_objective_values_helper(name, values):
        assert values.shape == (n_hps, n_seeds, n_fidelities)

        objective_evaluations[..., name_index[name]] = values

    ve = np.empty((n_hps, n_seeds, n_fidelities)).astype('float32')
    te = np.empty((n_hps, n_seeds, n_fidelities)).astype('float32')
    rt = np.empty((n_hps, n_seeds, n_fidelities)).astype('float32')

    for ai in range(n_hps):
        for si, seed in enumerate([777, 888, 999]):

            try:
                entry = data['arch2infos'][ai]['200']['all_results'][(dataset, seed)]
                validation_error = [1 - entry['eval_acc1es']['ori-test@%d' % ei] / 100 for ei in range(n_fidelities)]
                train_error = [1 - entry['train_acc1es'][ei] / 100 for ei in range(n_fidelities)]
                # runtime measure the time for a single epoch
                runtime = [entry['train_times'][ei] + entry['eval_times']['ori-test@%d' % ei]
                           for ei in range(n_fidelities)]

            except KeyError:
                validation_error = [np.nan] * n_fidelities
                train_error = [np.nan] * n_fidelities
                runtime = [np.nan] * n_fidelities
            ve[ai, si, :] = validation_error
            te[ai, si, :] = train_error
            rt[ai, si, :] = runtime

    def impute(values):
        idx = np.isnan(values)
        a, s, e = np.where(idx == True)
        for ai, si, ei in zip(a, s, e):
            l = values[ai, :, ei]
            m = np.mean(np.delete(l, si))
            values[ai, si, ei] = m
        return values

    # The original data contains missing values, since not all architectures were evaluated for all three seeds
    # We impute these missing values by taking the average of the available datapoints for the corresponding
    # architecture and time step

    save_objective_values_helper('valid_error', impute(ve))
    save_objective_values_helper('train_error', impute(te))
    save_objective_values_helper('runtime', impute(rt))

    latency = np.array(
        [data['arch2infos'][ai]['200']['all_results'][(dataset, 777)]['latency'][0] for ai in range(n_hps)])
    latency = np.repeat(np.expand_dims(latency, axis=-1), n_seeds, axis=-1)
    latency = np.repeat(np.expand_dims(latency, axis=-1), n_fidelities, axis=-1)
    save_objective_values_helper('latency', latency)

    flops = np.array([data['arch2infos'][ai]['200']['all_results'][(dataset, 777)]['flop'] for ai in range(n_hps)])
    flops = np.repeat(np.expand_dims(flops, axis=-1), n_seeds, axis=-1)
    flops = np.repeat(np.expand_dims(flops, axis=-1), n_fidelities, axis=-1)
    save_objective_values_helper('flops', flops)

    params = np.array([data['arch2infos'][ai]['200']['all_results'][(dataset, 777)]['params'] for ai in range(n_hps)])
    params = np.repeat(np.expand_dims(params, axis=-1), n_seeds, axis=-1)
    params = np.repeat(np.expand_dims(params, axis=-1), n_fidelities, axis=-1)
    save_objective_values_helper('params', params)

    configuration_space = {
        node: search_space.choice(['avg_pool_3x3', 'nor_conv_3x3', 'skip_connect', 'nor_conv_1x1', 'none'])
        for node in hp_cols
    }

    fidelity_space = {
        "hp_epoch": search_space.randint(lower=1, upper=201)
    }

    return BlackboxTabular(
        hyperparameters=hyperparameters,
        configuration_space=configuration_space,
        fidelity_space=fidelity_space,
        objectives_evaluations=objective_evaluations,
        fidelity_values=fidelity_values,
        objectives_names=[f"metric_{m}" for m in objective_names],
    )


def generate_nasbench201():
    blackbox_name = "nasbench201"
    file_name = repository_path / "NATS-tss-v1_0-3ffb9.pickle.pbz2"
    if not file_name.exists():
        print(f"did not find {file_name}, downloading")
        import requests

        def download_file_from_google_drive(id, destination):
            URL = "https://docs.google.com/uc?export=download"

            session = requests.Session()

            response = session.get(URL, params={'id': id}, stream=True)
            token = get_confirm_token(response)

            if token:
                params = {'id': id, 'confirm': token}
                response = session.get(URL, params=params, stream=True)

            save_response_content(response, destination)

        def get_confirm_token(response):
            for key, value in response.cookies.items():
                if key.startswith('download_warning'):
                    return value

            return None

        def save_response_content(response, destination):
            CHUNK_SIZE = 32768

            with open(destination, "wb") as f:
                for chunk in response.iter_content(CHUNK_SIZE):
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)

        download_file_from_google_drive('1vzyK0UVH2D3fTpa1_dSWnp1gvGpAxRul', file_name)

    f = bz2.BZ2File(file_name, 'rb')
    data = pickle.load(f)

    with catchtime("converting"):
        bb_dict = {}
        for dataset in ['cifar10', 'cifar100', 'ImageNet16-120']:
            print(f"converting {dataset}")
            bb_dict[dataset] = convert_dataset(data, dataset)

    with catchtime("saving to disk"):
        serialize(bb_dict=bb_dict, path=repository_path / blackbox_name)

    with catchtime("uploading to s3"):
        from blackbox_repository.conversion_scripts.utils import upload
        upload(blackbox_name)


if __name__ == '__main__':
    generate_nasbench201()

    # plot one learning-curve for sanity-check
    from blackbox_repository import load

    bb_dict = load("nasbench201")

    b = bb_dict['cifar10']
    configuration = {k: v.sample() for k, v in b.configuration_space.items()}
    errors = []
    runtime = []

    import matplotlib.pyplot as plt

    for i in range(1, 201):
        res = b.objective_function(configuration=configuration, fidelity={'epochs': i})
        errors.append(res['metric_valid_error'])
        runtime.append(res['metric_runtime'])

    plt.plot(np.cumsum(runtime), errors)
    plt.show()
