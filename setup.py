from syne_tune import read_version
from setuptools import setup, find_packages
from pathlib import Path


def load_requirements(filename):
    with open(filename) as f:
        return f.read().splitlines()


def load_benchmark_requirements():
    # the requirements of benchmarks are placed into the same directory as the examples script
    res = set()
    for fname in Path(__file__).parent.glob("benchmarking/training_scripts/*/requirements.txt"):
        res.update(load_requirements(fname))
    # gluon-ts is not added as the git dependency does not work with setup.py
    k = 'git+https://github.com/awslabs/gluon-ts.git'
    if k in res:
        res.remove(k)
    return list(res)


required_core = load_requirements('requirements.txt')
required_ray = load_requirements('requirements-ray.txt')
required_gpsearchers = load_requirements('requirements-gpsearchers.txt')
required_bore = load_requirements('requirements-bore.txt')
required_kde = load_requirements('requirements-kde.txt')
required_blackbox_repository = load_requirements('benchmarking/blackbox_repository/requirements.txt')
required_benchmarks = load_benchmark_requirements()

long_description = (Path(__file__).parent / "README.md").read_text()

setup(
    name='syne_tune',
    version=read_version(),
    description='Distributed Hyperparameter Optimization on SageMaker',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='AWS',
    packages=find_packages(include=[
        'syne_tune',
        'syne_tune.*',
    ]),
    extras_require={
        'raytune': required_ray,
        'bore': required_bore,
        'kde': required_kde,
        'gpsearchers': required_gpsearchers,
        'benchmarks': required_benchmarks,
        'blackbox-repository': required_blackbox_repository,
        'extra': required_ray + required_gpsearchers + required_benchmarks + required_blackbox_repository + required_kde,
    },
    install_requires=required_core,
    include_package_data=True,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.8',
    ],
)
