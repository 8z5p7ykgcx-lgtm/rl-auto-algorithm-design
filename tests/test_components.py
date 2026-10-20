"""构件库与执行器测试：注册、按编码筛选、算法组装与运行。"""

import numpy as np

from auto_alg_design import FlowShop, Knapsack, TSP
from auto_alg_design.components.registry import AlgorithmSpec, describe, list_for
from auto_alg_design.components.runner import run_algorithm


def test_registry_filter_by_encoding():
    perm_inits = {c.cid for c in list_for("init", "permutation")}
    assert {"random_init", "greedy_tsp", "neh_init"} <= perm_inits
    bin_inits = {c.cid for c in list_for("init", "binary")}
    assert "greedy_kp" in bin_inits and "greedy_tsp" not in bin_inits
    accepts = {c.cid for c in list_for("accept", "any")}
    assert {"greedy_accept", "elite_accept", "sa_accept"} <= accepts


def test_run_algorithm_on_tsp():
    p = TSP(n=20, seed=1)
    inst = p.random_instance(np.random.default_rng(1))
    spec = AlgorithmSpec(
        init_id="greedy_tsp", init_param=0,
        search_steps=[("two_opt_ls", 1), ("swap_mutation", 0)] * 2,
        accept_id="elite_accept", accept_param=0, iterations=40,
    )
    cost, traj, sol = run_algorithm(p, inst, spec, np.random.default_rng(2), record_trajectory=True)
    assert traj is not None and len(traj) == 40
    assert traj[-1] <= traj[0] + 1e-9         # 精英保留：best-so-far 单调不增
    assert np.isclose(p.cost(inst, sol), cost)
    # 描述可读
    assert "init=" in describe(spec)


def test_run_algorithm_on_knapsack():
    p = Knapsack(n=40, seed=2)
    inst = p.random_instance(np.random.default_rng(2))
    spec = AlgorithmSpec(
        init_id="greedy_kp", init_param=0,
        search_steps=[("bitflip_ls", 1), ("bitflip_mutation", 1)] * 2,
        accept_id="greedy_accept", accept_param=0, iterations=30,
    )
    cost, _, sol = run_algorithm(p, inst, spec, np.random.default_rng(3))
    assert sol @ inst["weights"] <= inst["capacity"] + 1e-9
    assert cost < 0                             # 存在可行正价值解


def test_run_algorithm_on_flowshop():
    p = FlowShop(n_jobs=8, n_machines=4, seed=3)
    inst = p.random_instance(np.random.default_rng(3))
    spec = AlgorithmSpec(
        init_id="neh_init", init_param=0,
        search_steps=[("first_improve_ls", 1), ("insert_mutation", 0)] * 2,
        accept_id="sa_accept", accept_param=1, iterations=30,
    )
    cost, _, sol = run_algorithm(p, inst, spec, np.random.default_rng(4))
    assert len(np.unique(sol)) == 8
    assert cost > 0


def pytest_approx(x):
    return float(np.round(x, 9))
