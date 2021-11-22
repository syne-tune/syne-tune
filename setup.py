from setuptools import setup, find_packages
from pathlib import Path


def load_requirements(filename):
    with open(filename) as f:
        return f.read().splitlines()


def load_benchmark_requirements():
    # the requirements of benchmarks are placed into the same directory as the examples script
    res = set()
    for fname in Path(__file__).parent.glob("examples/training_scripts/*/requirements.txt"):
        res.update(load_requirements(fname))
    # gluon-ts is not added as the git dependency does not work with setup.py
    k = 'git+https://github.com/awslabs/gluon-ts.git'
    if k in res:
        res.remove(k)
    return list(res)


def read_version():
    with open(Path(__file__).parent / "version.py", "r") as f:
        return f.readline()


required_core = load_requirements('requirements.txt')
required_ray = load_requirements('requirements-ray.txt')
required_gpsearchers = load_requirements('requirements-gpsearchers.txt')
required_bore = load_requirements('requirements-bore.txt')
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
        'gpsearchers': required_gpsearchers,
        'benchmarks': required_benchmarks,
        'extra': required_ray + required_gpsearchers + required_benchmarks,
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
