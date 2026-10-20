"""基线与端到端训练测试（小规模、快速）。"""

import numpy as np

from auto_alg_design import FlowShop, Knapsack, TSP
from auto_alg_design.baselines.genetic_algorithm import GeneticAlgorithm
from auto_alg_design.baselines.simulated_annealing import SimulatedAnnealing
from auto_alg_design.rl.trainer import TrainConfig, train


def test_ga_and_sa_runnable():
    p = TSP(n=20, seed=1)
    inst = p.random_instance(np.random.default_rng(1))
    ga = GeneticAlgorithm(population=12, generations=10, seed=0)
    sol, cost = ga.solve(p, inst)
    assert len(np.unique(sol)) == 20 and cost > 0
    sa = SimulatedAnnealing(iterations=50, seed=0)
    sol2, cost2 = sa.solve(p, inst)
    assert cost2 > 0


def test_short_training_loop():
    """短训练：确保策略梯度闭环可跑通（验证型，非效果型）。"""
    p = TSP(n=15, seed=2)
    res = train(p, TrainConfig(episodes=12, eval_every=6, log=True), iterations=10, n_search_steps=2)
    assert len(res.rewards) == 12
    assert res.best_spec_text != ""
    assert len(res.eval_means) == 2           # 6、12 两个评估点
