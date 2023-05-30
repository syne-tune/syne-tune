from syne_tune import read_version
from setuptools import setup, find_packages
from pathlib import Path
import sys


def load_requirements(filename):
    with open(filename) as f:
        return f.read().splitlines()


def load_benchmark_requirements():
    # the requirements of benchmarks are placed into the same directory as the examples script
    res = set()
    for fname in Path(__file__).parent.glob(
        "benchmarking/training_scripts/*/requirements.txt"
    ):
        res.update(load_requirements(fname))
    # gluon-ts is not added as the git dependency does not work with setup.py
    k = "git+https://github.com/awslabs/gluon-ts.git"
    if k in res:
        res.remove(k)
    return list(res)


required_core = load_requirements("requirements.txt")
required_ray = load_requirements("requirements-ray.txt")
required_gpsearchers = load_requirements("requirements-gpsearchers.txt")
required_bore = load_requirements("requirements-bore.txt")
required_botorch = load_requirements("requirements-botorch.txt")
required_kde = load_requirements("requirements-kde.txt")
required_blackbox_repository = load_requirements(
    "syne_tune/blackbox_repository/requirements.txt"
)
required_yahpo = load_requirements(
    "syne_tune/blackbox_repository/conversion_scripts/scripts/requirements-yahpo.txt"
)
required_benchmarks = load_benchmark_requirements()
required_dev = load_requirements("requirements-dev.txt")
required_aws = load_requirements("requirements-aws.txt")
required_moo = load_requirements("requirements-moo.txt")
required_visual = load_requirements("requirements-visual.txt")
required_sklearn = load_requirements("requirements-sklearn.txt")

long_description = (Path(__file__).parent / "README.md").read_text()
required_extra = (
    required_gpsearchers
    + required_kde
    + required_dev
    + required_aws
    + required_moo
    + required_visual
    + required_sklearn
    + required_blackbox_repository
    + required_benchmarks
    + required_yahpo
    + required_ray
)

# Botorch only supports python version >= 3.8
if sys.version_info >= (3, 8):
    required_extra += required_botorch

setup(
    name="syne_tune",
    version=read_version(),
    description="Distributed Hyperparameter Optimization on SageMaker",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="AWS",
    packages=find_packages(
        include=[
            "syne_tune",
            "syne_tune.*",
        ]
    ),
    extras_require={
        "gpsearchers": required_gpsearchers,
        "kde": required_kde,
        "dev": required_dev,
        "aws": required_aws,
        "moo": required_moo,
        "visual": required_visual,
        "sklearn": required_sklearn,
        "blackbox-repository": required_blackbox_repository,
        "benchmarks": required_benchmarks,
        "yahpo": required_yahpo,
        "raytune": required_ray,
        "botorch": required_botorch,
        "bore": required_bore,
        "extra": required_extra,
    },
    install_requires=required_core,
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.8",
    ],
)
