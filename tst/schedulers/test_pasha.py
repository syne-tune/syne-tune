from syne_tune.optimizer.schedulers.hyperband_pasha import PASHARungSystem


def create_pasha_rung_system(mode="max"):
    """
    Function to set-up the rung system for testing. It is possible to pass the relevant arguments for testing.
    """
    rung_levels = [1, 3, 9, 27, 81]
    promote_quantiles = [
        0.3333333333333333,
        0.3333333333333333,
        0.3333333333333333,
        0.3333333333333333,
        0.405,
    ]
    metric = "objective"
    resource_attr = "epoch"
    max_t = 200

    pasha_rung_rystem = PASHARungSystem(
        rung_levels, promote_quantiles, metric, mode, resource_attr, max_t
    )
    return pasha_rung_rystem


def test_resources_increase():
    prs = create_pasha_rung_system(mode="max")

    # define the current value of epsilon
    prs.epsilon = 0.1

    oneranking = [[("0", 0, 10.0), ("1", 3, 19.6), ("2", 2, 14.3), ("3", 1, 11.6)]]
    tworankings = [
        [("0", 0, 10.0), ("1", 3, 19.6), ("2", 2, 14.3), ("3", 1, 11.6)],
        [("0", 0, 10.1), ("1", 3, 19.7), ("2", 2, 14.2), ("3", 1, 11.5)],
    ]

    assert prs._decide_resource_increase(oneranking) == False

    assert prs._decide_resource_increase(tworankings) == False

    tworankings = [
        [("0", 0, 10.0), ("1", 1, 11.55), ("2", 3, 14.3), ("3", 2, 11.6)],
        [("0", 0, 10.1), ("1", 3, 19.7), ("2", 2, 14.2), ("3", 1, 11.5)],
    ]
    assert prs._decide_resource_increase(tworankings) == True


def test_soft_ranking_criterion():
    """
    Test for verifying the correctness of the soft ranking criterion.

    There are several variations of the soft ranking, but they are all
    fundamentally the same and the variations only use simple methods
    for estimating the value of epsilon - so it is enough to have one set of tests.
    """
    # test if the ranking is the same
    pasha_rung_system = create_pasha_rung_system(mode="min")
    pasha_rung_system.epsilon = 0.3
    rankings = [
        [("0", 0, 10.0), ("1", 3, 19.6), ("2", 2, 14.3), ("3", 1, 11.6)],
        [("0", 0, 10.1), ("1", 3, 19.7), ("2", 2, 14.2), ("3", 1, 11.5)],
    ]
    sorted_top_rung, sorted_previous_rung = pasha_rung_system._get_sorted_top_rungs(
        rankings
    )
    keep_current_budget = pasha_rung_system._evaluate_soft_ranking(
        sorted_top_rung, sorted_previous_rung
    )

    assert keep_current_budget == True

    # test if the change is only within the group
    rankings = [
        [("0", 0, 10.0), ("1", 3, 14.6), ("2", 2, 14.3), ("3", 1, 11.6)],
        [("0", 0, 10.0), ("1", 2, 14.2), ("2", 3, 14.3), ("3", 1, 11.6)],
    ]
    sorted_top_rung, sorted_previous_rung = pasha_rung_system._get_sorted_top_rungs(
        rankings
    )
    keep_current_budget = pasha_rung_system._evaluate_soft_ranking(
        sorted_top_rung, sorted_previous_rung
    )

    assert keep_current_budget == True

    # test if the change is outside the group
    pasha_rung_system = create_pasha_rung_system(mode="min")
    pasha_rung_system.epsilon = 0.03
    rankings = [
        [("0", 0, 10.0), ("1", 3, 14.6), ("2", 2, 14.3), ("3", 1, 11.6)],
        [("0", 0, 10.0), ("1", 2, 14.2), ("2", 3, 14.3), ("3", 1, 11.6)],
    ]
    sorted_top_rung, sorted_previous_rung = pasha_rung_system._get_sorted_top_rungs(
        rankings
    )
    keep_current_budget = pasha_rung_system._evaluate_soft_ranking(
        sorted_top_rung, sorted_previous_rung
    )

    assert keep_current_budget == False

    # test if the change is within the group while maximizing the objective
    pasha_rung_system = create_pasha_rung_system(mode="max")
    pasha_rung_system.epsilon = 0.3
    rankings = [
        [("0", 0, 10.0), ("1", 3, 14.6), ("2", 2, 14.3), ("3", 1, 11.6)],
        [("0", 0, 10.0), ("1", 2, 14.2), ("2", 3, 14.3), ("3", 1, 11.6)],
    ]
    sorted_top_rung, sorted_previous_rung = pasha_rung_system._get_sorted_top_rungs(
        rankings
    )
    keep_current_budget = pasha_rung_system._evaluate_soft_ranking(
        sorted_top_rung, sorted_previous_rung
    )

    assert keep_current_budget == True
