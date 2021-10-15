from sagemaker_tune.backend.sagemaker_backend.instance_info import select_instance_type, InstanceInfos


def test_instance_info():
    instance_infos = InstanceInfos()
    for instance in select_instance_type(max_gpu=0):
        assert instance_infos(instance).num_gpu == 0

    for instance in select_instance_type(min_gpu=1):
        assert instance_infos(instance).num_gpu >= 1

    for instance in select_instance_type(min_cost_per_hour=0.5, max_cost_per_hour=4.0):
        cost = instance_infos(instance).cost_per_hour
        assert 0.5 <= cost <= 4.0
