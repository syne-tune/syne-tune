from setuptools import setup, find_packages
from pathlib import Path


def load_requirements(filename):
    with open(filename) as f:
        return f.read().splitlines()


required_core = load_requirements('requirements.txt')
required_ray = load_requirements('requirements-ray.txt')
required_gpsearchers = load_requirements('requirements-gpsearchers.txt')
required_bore = load_requirements('requirements-bore.txt')
required_blackbox_repository = load_requirements('requirements-blackbox-repository.txt')

# the requirements of benchmarks are placed into the same directory as the examples script, this allows to install
# them on the fly when using Sagemaker frameworks
required_benchmarks = set()
for fname in Path(__file__).parent.glob("examples/training_scripts/*/requirements.txt"):
    required_benchmarks.update(load_requirements(fname))
# Fix:
k = 'git+https://github.com/awslabs/gluon-ts.git'
if k in required_benchmarks:
    required_benchmarks.remove(k)
required_benchmarks = list(required_benchmarks)

setup(
    name='syne_tune',
    version='0.1',
    description='Distributed Hyperparameter Optimization on SageMaker',
    author='',
    packages=find_packages(include=[
        'syne_tune',
        'syne_tune.*',
    ]),
    extras_require={
        'raytune': required_ray,
        'bore': required_bore,
        'gpsearchers': required_gpsearchers,
        'benchmarks': required_benchmarks,
        'blackbox-repository': required_blackbox_repository,
        'extra': required_ray + required_gpsearchers + required_benchmarks + required_blackbox_repository,
    },
    install_requires=required_core,
    include_package_data=True,
)
