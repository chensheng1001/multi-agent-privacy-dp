# Multi-Agent Privacy with Fact-Level Differential Privacy

研究多智能体协作系统中的组合隐私泄露问题，提出基于Fact-Level Differential Privacy的形式化防御方法。

## 研究背景

多个AI agent各自持有部分数据，看似无害的信息被组合后可能泄露敏感属性（如健康状况、身份等）。现有防御（Patil et al., 2025）依赖启发式LLM判断，缺乏形式化隐私保证。

本项目提出**Fact-Level DP (FDP)**：在"是否说出某个fact"的决策层面引入差分隐私，通过exponential mechanism选择信息项，并利用composition theorem证明组合后的隐私预算上界。

## 快速开始

### 1. 环境准备

```bash
# Python 3.9+
pip install -r requirements.txt
```

### 2. 配置API

编辑 `config.yaml`，填入你的LLM API信息：

```yaml
api:
  base_url: "https://your-api-endpoint/v1"
  api_key: "your-api-key"
  model: "your-model-name"
```

### 3. 运行实验

```bash
python run_experiment.py
```

或者指定配置文件：

```bash
python run_experiment.py config.yaml
```

### 4. 查看结果

实验完成后，结果保存在 `results/` 目录：

- `results/full_results.json` — 完整的逐场景结果
- `results/metrics_summary.json` — 汇总指标

同时终端会打印汇总表格，包含以下指标：

| 指标 | 含义 | 期望方向 |
|------|------|----------|
| Leakage Accuracy | 攻击者成功推断敏感属性的比例 | ↓ 越低越好 |
| Blocking Rate | 防御者拒绝敏感查询的比例 | ↑ 越高越好 |
| Benign Success | 良性查询成功回答的比例 | ↑ 越高越好 |
| Balanced Outcome | 隐私与效用的平衡指标 | ↑ 越高越好 |

## 项目结构

```
multi-agent-privacy-dp/
├── config.yaml              # 实验配置（API、参数、防御方法）
├── requirements.txt         # Python依赖
├── run_experiment.py        # 主入口：运行实验
├── src/
│   ├── __init__.py
│   ├── config.py            # 配置管理
│   ├── llm_client.py        # OpenAI兼容API客户端
│   ├── scenario_gen.py      # 场景生成（医疗/企业/教育领域）
│   ├── defenders.py         # 防御实现（NoDefense/CoT/ToM/FactDP）
│   ├── attacker.py          # 攻击者：多步查询+推断
│   └── evaluator.py         # 评估：泄露率/阻断率/效用
└── results/                 # 实验结果输出目录
```

## 防御方法

### Baseline (复现 Patil et al.)

- **No Defense**: 无防御，所有查询如实回答
- **CoT (Chain-of-Thought)**: 链式推理判断是否敏感
- **ToM (Theory-of-Mind)**: 推理查询者意图后决定

### 本文方法

- **FactDP (Fact-Level Differential Privacy)**:
  - 将agent的知识库分解为离散的信息项 (facts)
  - 对每个fact计算隐私敏感度 Δ(f, s*)
  - 使用exponential mechanism选择要返回的facts
  - 跨查询追踪隐私预算，利用composition theorem保证组合隐私上界

## 配置参数说明

```yaml
factdp:
  epsilon: 2.0              # 每次查询的隐私预算（越大越不私密，效用越高）
  max_total_epsilon: 10.0   # 每个场景的最大总隐私预算
  lambda_tradeoff: 1.0      # 隐私-效用trade-off参数（越大越保守）
  composition: "advanced"    # "basic" 或 "advanced"（advanced更紧）
  delta: 1e-5               # advanced composition的δ参数

experiment:
  num_adversarial_scenarios: 60   # 对抗性场景数量
  num_benign_scenarios: 59        # 良性场景数量
  random_seed: 42                 # 随机种子（可复现）
```

## 调参建议

- **ε 较小 (0.5-1.0)**: 更强隐私保护，但效用下降明显
- **ε 较大 (3.0-5.0)**: 更高效用，但隐私保护减弱
- **λ 较大 (2.0+)**: 更保守，倾向拒绝敏感信息
- **λ 较小 (0.5-)**: 更宽松，倾向回答查询

建议先用默认参数运行，再调整ε观察privacy-utility trade-off曲线。

## 预计运行时间

- 119个场景 × 4种防御方法 ≈ 1-3小时（取决于API响应速度）
- 每个场景约需3-6次LLM调用（defender响应 + attacker推断 + 评估）

## 引用

如果使用本代码，请引用：

```
[待补充论文信息]
```
