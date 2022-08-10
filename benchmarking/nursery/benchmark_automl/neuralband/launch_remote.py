from argparse import ArgumentParser
from pathlib import Path

from coolname import generate_slug
from sagemaker.pytorch import PyTorch

from benchmarking.nursery.benchmark_automl.neuralband.baselines import methods, Methods
from syne_tune.backend.sagemaker_backend.sagemaker_utils import get_execution_role
import syne_tune
import benchmarking
from syne_tune.util import s3_experiment_path, random_string


benchmark_names =  ['fcnet-protein', 'fcnet-naval', 'fcnet-parkinsons', 'fcnet-slice', 'nas201-cifar10', 'nas201-cifar100', 'nas201-ImageNet16-120', 'lcbench-APSFailure', 'lcbench-Amazon-employee-access', 'lcbench-Australian', 'lcbench-Fashion-MNIST', 'lcbench-KDDCup09-appetency', 'lcbench-MiniBooNE', 'lcbench-adult', 'lcbench-airlines', 'lcbench-albert', 'lcbench-bank-marketing', 'lcbench-car', 'lcbench-christine', 'lcbench-cnae-9', 'lcbench-connect-4', 'lcbench-covertype', 'lcbench-credit-g', 'lcbench-dionis', 'lcbench-fabert', 'lcbench-helena', 'lcbench-higgs', 'lcbench-jannis', 'lcbench-jasmine', 'lcbench-kc1', 'lcbench-kr-vs-kp', 'lcbench-mfeat-factors', 'lcbench-nomao', 'lcbench-numerai286', 'lcbench-phoneme', 'lcbench-segment', 'lcbench-shuttle', 'lcbench-sylvine', 'lcbench-vehicle', 'lcbench-volkert']
#benchmark_names =  ['fcnet-naval']
#benchmark_names =  ['fcnet-protein', 'fcnet-naval', 'fcnet-parkinsons', 'fcnet-slice', 'nas201-cifar10', 'nas201-cifar100', 'nas201-ImageNet16-120']

#benchmark_names = ['fcnet-protein']
#benchmark_names =  ['fcnet-protein', 'fcnet-naval', 'fcnet-parkinsons', 'fcnet-slice']
benchmark_fcnas =  ['fcnet-protein', 'fcnet-naval', 'fcnet-parkinsons', 'fcnet-slice', 'nas201-cifar10', 'nas201-cifar100', 'nas201-ImageNet16-120']


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--experiment_tag", type=str, required=False, default=generate_slug(2)
    )
    args, _ = parser.parse_known_args()
    experiment_tag = args.experiment_tag
    #experiment_tag = "tttNtest5"
    print("experiment_tag", experiment_tag)
    hash = random_string(4)
    #methods  = ["RS", "ASHA", "GP", "BOHB", "MOB", "TPE", "NeuralBand"]
    #methods  = {}
    #methods  = {"NeuralBandHP":1}
    print(methods.keys())
    for method in methods.keys():
        sm_args = dict(
            entry_point="benchmark_main.py",
            source_dir=str(Path(__file__).parent),
            # instance_type="local",
            checkpoint_s3_uri=s3_experiment_path(
                tuner_name=method, experiment_name=experiment_tag
            ),
            instance_type="ml.c5.4xlarge",
            instance_count=1,
            py_version="py38",
            framework_version="1.10.0",
            max_run=3600 * 72,
            role= 'arn:aws:iam::036649622372:role/service-role/AmazonSageMakerServiceCatalogProductsUseRole',
            dependencies=syne_tune.__path__ + benchmarking.__path__,
            disable_profiler=True,
        )
        
        #if method == Methods.NeuralBand or method == Methods.NeuralBandHP or method ==  Methods.GP:
        if method == Methods.NeuralBandSH or method == Methods.NeuralBandHP or method == Methods.MOBSTER:
        # For mobster, we schedule one job per seed as the method takes much longer
            for seed in range(3):
                for benchm in benchmark_names:
                    print(f"{experiment_tag}-{method}-{benchm}-{seed}")
                    sm_args["hyperparameters"] = {
                        "experiment_tag": experiment_tag,
                        "num_seeds": seed,
                        "run_all_seed": 0,
                        "method": method,
                        "benchmark": benchm,
                    }
                    est = PyTorch(**sm_args)
                    #print(experiment_tag)
                    est.fit(job_name=f"{experiment_tag}-{method}-{benchm}-{seed}-{hash}", wait=False)

        elif method ==  Methods.RS:
            print(f"{experiment_tag}-{method}")
            sm_args["hyperparameters"] = {
                "experiment_tag": experiment_tag,
                "num_seeds": 3,
                "run_all_seed": 1,
                "method": method,
            }
            est = PyTorch(**sm_args)
            #print(experiment_tag)
            est.fit(job_name=f"{experiment_tag}-{method}-{hash}", wait=False)
        else:
            for seed in range(3):
                print(f"{experiment_tag}-{method}-{seed}")
                sm_args["hyperparameters"] = {
                    "experiment_tag": experiment_tag,
                    "num_seeds": seed,
                    "run_all_seed": 0,
                    "method": method,
                }
                est = PyTorch(**sm_args)
                #print(experiment_tag)
                est.fit(job_name=f"{experiment_tag}-{method}-{seed}-{hash}", wait=False)
        
            
            
'''


     for seed in range(5):
            for benchm in benchmark_names:
                print(f"{experiment_tag}-{method}-{benchm}-{seed}")
                sm_args["hyperparameters"] = {
                    "experiment_tag": experiment_tag,
                    "num_seeds": seed,
                    "run_all_seed": 0,
                    "method": method,
                    "benchmark": benchm,
                }
                est = PyTorch(**sm_args)
                #print(experiment_tag)
                est.fit(job_name=f"{experiment_tag}-{method}-{benchm}-{seed}-{hash}", wait=False)

        


  elif method ==  Methods.GP:
             for seed in range(3):
                print(f"{experiment_tag}-{method}-{seed}")
                sm_args["hyperparameters"] = {
                    "experiment_tag": experiment_tag,
                    "num_seeds": seed,
                    "run_all_seed": 0,
                    "method": method,
                }
                est = PyTorch(**sm_args)
                #print(experiment_tag)
                est.fit(job_name=f"{experiment_tag}-{method}-{seed}-{hash}", wait=False)

'''
            
           
