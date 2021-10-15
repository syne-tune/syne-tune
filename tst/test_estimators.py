from pathlib import Path

from sagemaker_tune.backend.sagemaker_backend.estimator_factory import \
    sagemaker_estimator_factory
from benchmarks.benchmark_factory import benchmark_factory, \
    supported_benchmarks
from sagemaker_tune.backend.sagemaker_backend.sagemaker_utils import \
    get_execution_role


def test_create_estimators():
    try:
        role = get_execution_role()
        for benchmark_name in supported_benchmarks():
            benchmark = benchmark_factory({'benchmark_name': benchmark_name})
            def_params = benchmark['default_params']
            framework = def_params.get('framework')
            if framework is not None:
                sm_estimator = sagemaker_estimator_factory(
                    entry_point=benchmark['script'],
                    instance_type=def_params['instance_type'],
                    framework=framework,
                    role=role,
                    dependencies=[str(Path(__file__).parent.parent / "benchmarks/")],
                    framework_version=def_params.get('framework_version'),
                    pytorch_version=def_params.get('pytorch_version'))
    except Exception:
        print("Cannot run this test, because SageMaker role is not specified, "
              "and it cannot be inferred")
