# RL Auto Algorithm Design

> 智能优化算法**自动设计**（Automated Design of Metaheuristics）验证原型 —— 用强化学习从算法构件库中**自动组合算法并自动调参**，在 TSP / 0-1 背包 / 置换流水车间三个经典组合优化问题上，与人工设计的 GA、SA 进行对比。

本项目是《智能优化算法自动设计综述》中「**方向1：强化学习驱动的一体化自动算法设计**」的技术验证原型：把综述指出的三个割裂模块 —— 自动算法组合（GSF/超启发式）、自动算法选择、自动调参 —— 收敛为「**构件自动组合 + 构件参数自动选择**」的端到端决策问题，由同一策略梯度智能体一次性完成。

---

## 一、核心思想

传统人工设计元启发式算法的流程：研究者凭经验选择 `初始化算子 / 扰动算子 / 局部搜索算子 / 接受准则`，再手动调参（种群规模、温度、迭代次数……）。

本项目把这一过程交给强化学习：

```
问题实例特征 ──►  RL 策略（逐位决策） ──►  自动生成的算法描述
                                              │
     [init #param] -> [search_op #param] ×4 -> [accept #param]
                                              │
                                              ▼
                                    在实例上运行求解 ──► 性能奖励 ──► 更新策略
```

- **构件组合**：对应综述 GSF 框架的构件库思想，智能体从初始化/搜索/接受构件库中挑选并排序；
- **自动调参**：每个被选构件同时带**离散参数档位**（如 2-opt 的扫描轮数、SA 接受准则的初始温度档位），由同一决策步一并输出——「构件 + 参数」联合决策；
- **一体化闭环**：生成算法 → 运行 → 奖励 → 更新策略，端到端，无需人工介入中间环节。

## 二、架构

```
rl-auto-algorithm-design/
├── auto_alg_design/
│   ├── problems/            # 组合优化问题（统一接口：随机实例/解/成本/特征）
│   │   ├── base.py          #   Problem 基类（成本一律最小化）
│   │   ├── tsp.py           #   对称欧氏 TSP（含最近邻贪心）
│   │   ├── knapsack.py      #   0-1 背包（含密度贪心修复）
│   │   └── flowshop.py      #   置换流水车间（含 NEH 启发式）
│   ├── components/          # 算法构件库（自动组合的"零件"）
│   │   ├── registry.py      #   构件注册表 + AlgorithmSpec（自动设计的算法描述）
│   │   ├── init_ops.py      #   初始化构件：random / greedy_tsp / greedy_kp / neh
│   │   ├── search_ops.py    #   搜索构件：swap / inversion / insert / bitflip / 2-opt / first-improve
│   │   ├── accept_rules.py  #   接受构件：greedy / elite / simulated annealing
│   │   └── runner.py        #   算法执行器：按 spec 在实例上迭代求解
│   ├── rl/                  # 强化学习核心（纯 numpy，零重依赖）
│   │   ├── features.py      #   实例特征提取与标准化
│   │   ├── env.py           #   算法设计环境（决策模板 + 奖励）
│   │   ├── policy.py        #   Softmax 线性策略（每位独立权重）
│   │   └── trainer.py       #   REINFORCE + 滑动平均基线 + 动量 SGD
│   ├── baselines/           # 人工设计算法对照
│   │   ├── genetic_algorithm.py
│   │   └── simulated_annealing.py
│   ├── llm/                 # ★ 大模型驱动的自动算法设计（可选功能）
│   │   ├── client.py        #   OpenAI 兼容客户端（DeepSeek/OpenAI/智谱/千问）
│   │   └── designer.py      #   prompt 构建 + 响应解析校验 + 回退默认算法
│   ├── evaluate.py          # 评估器：未见实例上 RL vs GA vs SA vs Random
│   └── visualize.py         # 可视化：训练曲线 / 对比条形图 / TSP 路径 / 构件使用统计
├── scripts/
│   ├── train.py             # 单问题训练入口
│   ├── evaluate.py          # 单问题评估入口
│   ├── demo.py              # 一键：三个问题 训练+评估+可视化
│   └── llm_design.py        # ★ LLM 自动算法设计入口（按模板给数据即可跑）
├── examples/                # ★ LLM 模板示例（tsp / knapsack / flowshop）
└── tests/                   # pytest 测试
```

## 三、快速开始

```bash
# 1) 安装依赖（仅 numpy / matplotlib / scipy；pytest 用于测试）
pip install -r requirements.txt

# 2) 一键 demo：三个问题训练 + 评估 + 可视化（约 2~5 分钟）
python scripts/demo.py --episodes 250

# 3) 单独训练 / 评估某个问题
python scripts/train.py --problem tsp --episodes 300
python scripts/evaluate.py --problem tsp --policy outputs/tsp_policy.pkl

# 4) 跑测试
python -m pytest tests/ -v

# 5) ★ 大模型驱动的自动算法设计（可选）：按模板给数据，LLM 自动选算法去跑
export LLM_API_KEY=sk-xxxx                 # DeepSeek Key
export LLM_BASE_URL=https://api.deepseek.com   # 可选（换 OpenAI/智谱/千问就改这里+模型）
export LLM_MODEL=deepseek-chat             # 可选
python scripts/llm_design.py --template examples/tsp.json --baselines

# 没有 API Key 时，用 --mock 本地演示完整流程（不调用真实 API）
python scripts/llm_design.py --template examples/knapsack.json --mock
```

### LLM 模板格式

模板是一个 JSON 文件（见 `examples/`），**你只需要按模板填入问题类型和实例数据**：

```json
{
  "problem_type": "tsp",                    // tsp | knapsack | flowshop
  "instance": {                              // 你的数据；留空/只给规模则随机生成
    "coords": [[0.51, 0.95], [0.14, 0.95]]   // tsp: 城市坐标
  },
  "iterations": 60,                          // 主循环迭代预算
  "n_search_steps": 4,                       // 算法可包含的搜索构件步数上限
  "runs": 3,                                 // 对设计出的算法重复运行取最优
  "compare_baselines": true,
  "llm": { "model": "deepseek-chat", "temperature": 0.2 }
}
```

- **背包**：`instance` 填 `values` / `weights` / `capacity`；
- **流水车间**：`instance` 填 `p`（n_jobs × n_machines 加工时间矩阵）；
- **TSP**：`instance` 填 `coords`（二维坐标数组）；也可以只填 `"n": 30` 自动生成随机实例。

大模型只做**算法决策**（根据实例统计特征 + 构件库清单输出构件组合与参数档位），
数值求解仍由构件库执行器完成；输出会自动校验（构件 id / 参数档位 / 结构），
非法时回退到默认算法并给出警告。支持 `action: "select"` 直接选现成算法
（`ga` / `sa` / `random`）。兼容任意 OpenAI 兼容端点：换 `base_url` 与 `model`
即可接入 OpenAI、智谱 GLM、通义千问、Kimi 等。

所有运行产物输出到 `outputs/`：

| 文件 | 说明 |
|---|---|
| `{problem}_policy.pkl` | 训练好的策略（可复用） |
| `{problem}_train_curve.png` | 训练奖励曲线 + 未见实例评估点 |
| `{problem}_comparison.png` | 各算法平均成本对比条形图 |
| `{problem}_spec_usage.png` | 自动设计算法中构件使用频率统计 |
| `tsp_routes.png` | TSP 上各算法路径对比 |
| `results.txt` | 评估结果总表 |
| `llm_design_result.txt` | LLM 自动设计算法的求解结果（用 llm_design.py 生成） |

## 四、评估口径

- **训练 / 测试分离**：训练时每个 episode 重新采样实例；评估用全新 seed 生成的「未见实例」，检验泛化能力（避免综述指出的"搜索巨大算子库导致过拟合训练实例"）。
- **对比对象**：RL-Designed（自动设计算法）vs 人工 GA、SA、随机基线。
- **指标**：平均成本、最优成本、相对随机基线的平均改进率（`(random_cost - cost) / |random_cost|`，分母取绝对值以兼容背包等负成本问题）。

## 五、与论文创新点的对应关系（如何写进论文）

| 综述现有不足 | 本项目做法 |
|---|---|
| GSF 只做构件组合，缺参数调优 | 每个构件带参数档位，同一决策步联合输出（组合+调参一体化） |
| 算法选择/组合/调参模块割裂 | 统一建模为端到端序列决策，奖励闭环回传 |
| 超启发式多依赖手工特征 | 状态 = 问题实例统计特征 + 已选构件轨迹 + 决策进度 |
| 易过拟合训练实例 | 训练实例持续重采样 + 未见实例独立评估 |

## 六、局限与扩展方向

- **规模**：当前为中小规模实例验证（TSP 30 城、背包 50 物、流水车间 15×6）；更大规模需引入代理模型降评估成本。
- **奖励**：单标量奖励（解质量改进率）；可扩展为多目标（解质量 + 运行时间 + 鲁棒性）。
- **策略**：线性 softmax 策略；可替换为 PPO/分层 RL（上层选算法范式、下层调参），对应论文方向1 的完整三层架构。
- **构件库**：可继续扩充（路径重连、变邻域搜索 VNS、自适应算子等）。
- **LLM 协同**：本仓库的 `llm/` 模块已提供「LLM 做算法决策、构件库执行求解」的可选入口，可作为 EoH（LLM 驱动的超启发式演化）实验的起点。

## 七、License

MIT License。
