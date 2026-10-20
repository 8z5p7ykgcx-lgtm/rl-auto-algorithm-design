"""问题模块测试：实例生成、解的可行性、cost 计算、修复逻辑。"""

import numpy as np
import pytest

from auto_alg_design import FlowShop, Knapsack, TSP


def test_tsp_instance_and_cost():
    p = TSP(n=20, seed=1)
    inst = p.random_instance(np.random.default_rng(1))
    assert inst["dist"].shape == (20, 20)
    sol = p.random_solution(inst, np.random.default_rng(2))
    assert len(np.unique(sol)) == 20          # 排列：每个城市恰好一次
    c = p.cost(inst, sol)
    assert c > 0
    # 最近邻贪心应优于随机解（期望上）
    nn = TSP.nearest_neighbor_tour(inst, np.random.default_rng(3))
    assert p.cost(inst, nn) < p.cost(inst, np.arange(20, dtype=int))


def test_knapsack_repair_always_feasible():
    p = Knapsack(n=50, seed=1)
    rng = np.random.default_rng(4)
    inst = p.random_instance(rng)
    for _ in range(10):
        sol = p.random_solution(inst, rng)
        assert sol @ inst["weights"] <= inst["capacity"] + 1e-9
    # 全选会超容量 → 修复后必须可行
    full = np.ones(50, dtype=int)
    repaired = p.repair(inst, full)
    assert repaired @ inst["weights"] <= inst["capacity"] + 1e-9


def test_flowshop_makespan_and_neh():
    p = FlowShop(n_jobs=10, n_machines=5, seed=1)
    rng = np.random.default_rng(5)
    inst = p.random_instance(rng)
    sol = p.random_solution(inst, rng)
    ms = p.cost(inst, sol)
    assert ms > 0
    neh = FlowShop.neh_init(inst)
    assert p.cost(inst, neh) <= ms + 1e-9     # NEH 应不劣于随机


def test_features_shapes():
    for p, exp in [(TSP(20), 6), (Knapsack(30), 10), (FlowShop(8, 4), 8)]:
        inst = p.random_instance(np.random.default_rng(0))
        f = p.features(inst)
        assert f.shape == (exp,)
