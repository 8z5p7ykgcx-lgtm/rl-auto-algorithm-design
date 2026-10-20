"""环境与策略测试：状态维度、动作表、算法组装、策略采样与梯度。"""

import numpy as np

from auto_alg_design import TSP
from auto_alg_design.rl.env import DesignEnv
from auto_alg_design.rl.features import estimate_feature_stats
from auto_alg_design.rl.policy import SoftmaxPolicy


def make_env():
    p = TSP(n=20, seed=0)
    mean, std = estimate_feature_stats(p, n_samples=10, seed=0)
    inst = p.random_instance(np.random.default_rng(0))
    return p, DesignEnv(problem=p, instance=inst, feats_mean=mean, feats_std=std, n_search_steps=4)


def test_env_action_tables():
    _, env = make_env()
    assert len(env.positions) == 6             # init + 4 search + accept
    # 每个决策位至少有一个候选
    for i in range(6):
        assert env.action_space_size(i) >= 1


def test_env_state_dim_consistent():
    _, env = make_env()
    chosen = []
    d = env.state_dim()
    assert env.state(0, chosen).shape == (d,)
    # 每步状态维度一致
    for step in range(6):
        assert env.state(step, [0] * step).shape == (d,)


def test_build_spec_and_reward():
    p, env = make_env()
    actions = [0] * 6
    spec = env.build_spec(actions)
    assert spec.init_id == env.action_tables[0][0][0]
    assert len(spec.search_steps) == 4
    rng = np.random.default_rng(1)
    reward, cost = env.reward_for(spec, rng)
    assert np.isfinite(reward) and cost > 0


def test_policy_sampling_and_gradient():
    p, env = make_env()
    pol = SoftmaxPolicy(env, seed=0)
    x = env.state(0, [])
    a, prob = pol.sample(0, x, np.random.default_rng(0))
    assert 0 <= a < env.action_space_size(0)
    assert 0 < prob <= 1
    g = pol.grad_log_pi(0, x, a)
    assert g.shape == pol.W[0].shape
    # 概率和为 1
    pv = pol.action_probs(0, x)
    assert abs(pv.sum() - 1.0) < 1e-9
