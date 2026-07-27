# ROSClaw Phase 7：SelfCore-RL 基础实施与验证报告

日期：2026-07-27

分支：`phase7-selfcore-rl-foundation`

基线：Phase 6 分支提交 `d43c49c2fa40e54a4024e6d40a5625445b462ac8`

证据域：MuJoCo `SIM` 与四卡 `CUDA_SCREENING`；没有真实机器人执行或授权

## 1. 结论先行

本轮不是把 RoboNaldo 整网直接在线微调，而是建立了一个可继续扩展的安全基础：冻结原踢球策略与安全内核，只学习有界 residual；每条经验绑定策略、控制器、身体和工况版本；陈旧经验不能进入 actor；新权重只能在安全动作边界切换；任何稳定性、可塑性、自我核心或因果证据缺失都会阻止晋级。

已经真实完成并验证的内容包括：

- 版本化运动经验、四时间尺度回放和陈旧度路由；
- Constrained Residual SAC 学习内核，包括双 reward critic、独立 fall/constraint cost critic、熵、两个拉格朗日乘子、Anchor 蒸馏和父策略输出 churn 约束；
- 双缓冲、校验和、父子谱系、安全动作边界切换与回滚状态机；
- Stability-Plasticity 机器门，能区分 `REJECT` 与 `NEED_MORE_EVIDENCE`；
- 具身自我 Identity、Self State、Capability、Agency 证据合同；
- permutation-aware 的 SelfCore 候选发现、阈值敏感性分析和严格的因果证据门；
- G1 L1 平衡与 L2 踢球技能的组合“小脑”接口，以及真正连接 MuJoCo policy clock 的 `kick_phase_rate`；
- 80N 相同场景 feedback-off/on 因果窗口验证与严格重放；
- 四张实体 A6000 上并行执行 residual SAC 更新。

当前不能声称的内容：

- 不能声称 ROSClaw 或 G1 具有主观意识；
- 不能声称已经发现功能性 SelfCore；目前只生成“persistent subnetwork candidate”；
- 不能声称四卡筛查已经改善 G1 动作；四卡输入仍是版本合同夹具，不是 MuJoCo rollout；
- 不能声称 L2 相位、瞄准和恢复调制已经晋级；探索性物理测试发现回归，因此默认关闭；
- 不能声称达到 Phase 7 最终验收；多技能、多种子、移动球、身体变化、Agency 校准和真实 rollout 在线训练仍未完成。

## 2. 对参考工作的判断

### 2.1 AReaL 2.0

[AReaL 2.0](https://github.com/areal-project/AReaL/releases/tag/v2.0.0) 值得复用的是 rollout、训练、推理和权重更新解耦，以及数据携带权重版本、限制 off-policyness 的系统思想。它面向 LLM/Agent RL，不能直接照搬到高动态机器人控制。

ROSClaw 采用了更严格的映射：

| AReaL 概念 | ROSClaw 映射 | G1 额外约束 |
|---|---|---|
| token/trajectory version | control segment / episode policy version | 一次高动态动作只允许一个 policy version |
| rollout staleness | `permitted_use()` | lag ≤ 1 才可进入 actor；更旧数据只用于 critic/Self 分析 |
| weight update service | residual double-buffer slot | 摆腿、触球、恢复中禁止切换 |
| async learner | Feedback Plane 外的 SAC learner | 不得写 Registry、DDS 或硬件 transport |

### 2.2 Stability-Plasticity

[Loss of plasticity in deep continual learning](https://www.nature.com/articles/s41586-024-07711-7) 表明长期持续训练会丢失可塑性，普通梯度下降本身不够；低效单元重生是后续要实现的机制，而不是本轮已经完成的能力。

[C-CHAIN](https://arxiv.org/abs/2506.00592) 将持续 RL 的输出 churn 与表示/核秩退化联系起来。本轮据此加入父策略共享参考状态输出约束，并修复了一处“模型与同一时刻的自己比较、梯度恒为零”的实现陷阱。现在训练 penalty 对齐固定父策略参考网络，单步 churn 另行计量。

### 2.3 Self 论文

[Emergent Self in Continual Learning Robots](https://arxiv.org/abs/2603.24350) 的工程启发是：在共同参考状态上，经过持续多行为训练后，寻找跨任务稳定的网络子结构，再用 matched freeze/lesion 验证它是否有功能价值。

这里必须避免循环论证：先人工指定一块网络叫 SelfCore、冻结它，再观察它稳定，不能证明“自我核心涌现”。ROSClaw 因而分成两个不同对象：

1. `Embodied Self Model`：工程上明确设计的身份、状态、预测、Agency 和能力信念；
2. `SelfCore candidate`：训练完成后才做的统计发现，只有通过持续学习对照、阈值敏感性、多种子、freeze/lesion、身体预测和身体变化测试后，才可晋级为 self-like core。

论文讨论的是可测量的 self-like 不变子网络，不是主观意识的证明。ROSClaw 同样只使用“操作性具身自我认知”这一可测试表述。

## 3. 当前闭环架构

```text
MuJoCo / future shadow robot
        │  version-pinned episode
        ▼
Recent / Anchor / Boundary / Self replay
        │  fresh rows ───────────────► Actor
        └─ fresh + stale rows ───────► Reward/Cost Critics + Self analysis
                                      │
                                      ▼
                         Residual SAC candidate artifact
                                      │
                Stability + Plasticity + Self + Safety Gate
                                      │
                         checksum + double-buffer stage
                                      │
                     STAND / PREPARE / COMPLETE atomic swap
                                      │
                         bounded residual + Safety Projector
                                      │
                          frozen RoboNaldo kick prior
```

核心原则是：快速环更新当前状态并救动作，慢速环只改变未来回合的候选参数。一次摆腿不能同时使用两个脑版本。

## 4. 实现清单

### 4.1 Continual RL

- `src/rosclaw/continual/contracts.py`
  - `PolicyVersion` 使用内容寻址 artifact、父版本、控制器、body 和 safety kernel；
  - `ControlSegment` 记录 observation/action/next observation、behavior log-prob、Reward 和 Cost；
  - `VersionedTrajectory` 禁止高动态 episode 跨版本并要求 phase 单调。
- `src/rosclaw/continual/experience.py`
  - Recent 50%、Anchor 25%、Boundary 15%、Self 10%；
  - Anchor 必须绑定 champion；Boundary 必须有安全事件或近失效；Self 必须绑定身体变化；
  - 未来版本拒绝，过旧版本不能进入 actor。
- `src/rosclaw/continual/learner.py`
  - 有界 Gaussian residual actor；
  - twin reward critic、twin fall critic、twin constraint critic；
  - Anchor distillation、父策略 churn、熵和 cost Lagrangian；
  - 只输出候选 artifact，不激活 runtime。
- `src/rosclaw/continual/stability.py`
  - 历史平均下降 ≤ 3%，关键技能下降 ≤ 5%；
  - critical safety regression、stale execution、old-version replay 必须为零；
  - 新技能样本效率、fresh-network gap、dead unit、effective rank、churn 均有门；
  - SelfCore 证据缺失返回 `NEED_MORE_EVIDENCE`，安全失败返回 `REJECT`。
- `src/rosclaw/continual/weight_update.py`
  - artifact checksum、父子版本、body/safety identity；
  - 双缓冲、只在安全 phase 原子切换；
  - 中途切换直接冻结；所有 receipt 的 `hardware_authorized=False`。

### 4.2 具身自我与 SelfCore

- `src/rosclaw/self_model/contracts.py`
  - S0 `SelfIdentity`；
  - S1 `SelfStateSnapshot`，包含关节健康、motor gain/zero bias、时延、摩擦、负载、平衡、能量和传感质量信念；
  - S3 `AgencyAssessment` 的 self/external/sensor-fault 概率合同；
  - S4 `CapabilityBelief`。
- `src/rosclaw/self_model/self_core.py`
  - 共同参考状态 activation；
  - dead-unit 过滤；
  - absolute co-activation graph；
  - Hungarian 神经元排列对齐；
  - activation/connectivity persistence；
  - 多阈值敏感性分析；
  - 输出只能叫 candidate，模块本身禁止把 `causal_validated` 设为真。

尚未完成 S2 forward self model、Agency 估计器、Capability 在线校准器和 S5 autobiographical continuity。现有 Agency 是严格证据合同，不是已经达到 95% 准确率的分类器。

### 4.3 G1 小脑

- `G1CerebellumController` 在一次 Safety Projection 前组合 L1 balance 与 L2 skill residual；
- MuJoCo backend 已让 `skill:kick_phase_rate` 影响真实 policy frame，而不是只写日志；
- policy clock 只能前进或短暂停帧，不能倒放动作；
- 未校准的 phase、lateral aim 和 recovery 三个 L2 通道全部默认关闭；
- 原 Phase 6 L1 balance phase 语义保持不变，避免新 backend 破坏既有冠军行为。

探索性物理 A/B 显示，固定 `expected_contact_phase=0.48` 与本场景真实触球 policy phase（约 0.416）不一致；相位调制在部分 offset 改善目标误差，在 nominal 和另一 offset 反而恶化。lateral/recovery 也出现 nominal 目标误差约增加 4.7cm 的反例。因此这些数据进入 Boundary 思路，而不是作为晋级证据。

## 5. 实际验证结果

### 5.1 80N feedback 因果窗口

正式命令：

```bash
.venv/bin/python -m rosclaw.entrypoint simforge validate g1-goalforge \
  --profile feedback-causal \
  --asset-root /code/rosclaw/phase4_references/RoboNaldo/RoboNaldo_Deploy \
  --output /code/rosclaw/phase7_evidence/causal-feedback-v1.json
```

结果：

| 指标 | Feedback OFF | Feedback ON | 变化 |
|---|---:|---:|---:|
| Episode 状态 | `JOINT_LIMIT_EXCEEDED` | `SUCCESS` | 救援成功 |
| 扰动后姿态误差积分 | 2.78149 rad·s | 2.47131 rad·s | 改善 11.15% |
| 扰动后姿态误差峰值 | 0.65877 rad | 0.50815 rad | 改善 22.86% |
| 最后 0.5s 平均姿态误差 | 0.60136 rad | 0.35207 rad | 改善 41.46% |
| 严格重放 | — | 通过 | trajectory/result 一致 |

控制器在 5.64s 激活、7.82s 结束；激活前 joint/torso/ball 前缀逐值一致，激活后 counterfactual 发生分叉，因此可以把本场景的救援归因到反馈，而不是随机初态差异。

限制同样被保留：激活早期姿态误差积分由 0.47058 上升到 0.50383 rad·s，存在短暂退化；这解释了原 receipt 的 `tracking_improved=false`。该结果只支持一个固定 SIM 场景中的局部因果主张，仍为 `NEED_MORE_EVIDENCE`。

证据 SHA-256：`0e46ba7ac6d3b11e148f867faa250b5754608502f502f3fc9c3a6b81a35ff06c`

### 5.2 四卡 A6000 residual SAC 系统筛查

正式命令：

```bash
.venv/bin/python -m rosclaw.entrypoint simforge validate g1-goalforge \
  --profile continual-four-gpu-smoke \
  --output /code/rosclaw/phase7_evidence/continual-four-gpu-v2
```

结果：4 个不同 GPU UUID 全部参与；每卡完成 3 次 SAC 更新；loss、hidden activation 均为有限值；residual action 全部在边界内；每卡 critic 使用 128 条 transition，actor 只使用 96 条 fresh transition，32 条 stale transition 被排除出 actor；每卡约分配 17.99MB CUDA 内存，未停止或修改机器上的其他 GPU 进程。

这项测试的输入是 G1 命名的合成版本合同 transition，只验证系统路径，不证明动作性能。四个候选均未 stage、未 activate、未写 Registry。

汇总证据 SHA-256：`c1a58d73e2598a8449fc1a18eb7b736db30a64ae79364bfa04f2549740cb8963`

### 5.3 软件验证

阶段性验证已经覆盖：

- continual/self/cerebellum/phase-clock/causal/GPU smoke 定向测试；
- Ruff；
- Mypy；
- Python compileall；
- MuJoCo 真实 physics step 和严格 replay；
- 四张实体 A6000 的 CUDA kernel 与 optimizer update。

最终测试结果：

- 最终 `tests/feedback tests/simforge`：160 passed、1 skipped、3 deselected；
- 全仓首轮：5274 passed、67 skipped、27 deselected，4 个 LeRobot 集成失败；
- 失败定位为测试隔离 home 中没有继承外部 LeRobot 路径；显式使用已安装的 `/code/rosclaw/phase6_runtime/lerobot-0.6.1/bin/python` 后，4 项全部通过；
- 显式设置 `ROSCLAW_TEST_LEROBOT_PYTHON` 后最终全仓复跑：5286 passed、59 skipped、27 deselected、26 warnings，耗时 17m03s，无失败。

## 6. Stability-Plasticity 的实际解决方式

这不是靠一个正则项解决，而是六道不同失效面的防线：

1. 冻结 RoboNaldo prior、安全 projector、hard limits、permit/lease 和 freshness；
2. 只训练低维、有界 residual；
3. 四类回放每个 batch 都出现，旧技能和边界反例不能被新数据挤掉；
4. actor 只吃新鲜数据，critic/Self 分析可以利用更旧数据；
5. Anchor distillation 与固定父策略 churn 同时约束“学新动作时乱改其他状态输出”；
6. 候选必须同时通过 Stability、Plasticity、Self 和 Safety gate，且只在动作安全边界切换。

本轮尚未实现 plastic-unit rejuvenation 和 dynamic expert allocation，所以只能说“建立了可执行门和 learner foundation”，不能说已经彻底解决长期可塑性丢失。

## 7. Phase 7 文档 PR 对照

| v7 项目 | 当前状态 | 说明 |
|---|---|---|
| Gate 0 / PR-0A F6 attribution | 部分完成 | 新因果窗口能归因救援，也如实暴露早期退化；广分布仍缺 |
| Gate 0 / PR-0B phase clock | 接线完成、校准未完成 | 真实影响 policy frame，默认关闭 |
| Gate 0 / PR-0C Phase 6 main | 未完成 | 当前 Phase 7 仍堆叠在 Phase 6 分支基线上 |
| PR-1 RL Trajectory | 基础完成 | 版本、log-prob、body/regime、reward/cost 均已实现 |
| PR-2 Experience Service | 内核完成 | 当前为进程内 store，尚未服务化 |
| PR-3 Inference Service | 未完成 | 只有 learner inference API |
| PR-4 Learner Service | 算法内核完成 | SAC 已跑通，尚未形成长期服务 |
| PR-5 Weight Update | 参考状态机完成 | SIM-only，未接 Registry |
| PR-6 AReaL Adapter | 设计映射完成 | 尚未服务编排 |
| PR-7/8 Anchor/Churn | learner 与 gate 基础完成 | 真实多技能 benchmark 尚缺 |
| PR-9/10 Rejuvenation/Experts | 未完成 | 下一阶段重点 |
| PR-11 SelfState | 合同基础完成 | 在线估计器未完成 |
| PR-12 Forward Self Model | 未完成 | Agency 的必要前置 |
| PR-13 Agency | 证据合同完成 | 分类器和校准实验未完成 |
| PR-14 Capability | belief 合同完成 | 在线 Bayesian/校准更新未完成 |
| PR-15 SelfCore Analyzer | 非因果分析基础完成 | 不允许自证 causal |
| PR-16 SelfCore Protection | 未完成 | 必须等候选通过因果实验 |
| PR-18 Residual Cerebellum | 接口与物理接线完成 | L2 因回归而 fail-closed |

## 8. 下一阶段优先顺序

1. 先把真实 MuJoCo trajectory 转为 `ControlSegment`，替换四卡 smoke 的合成 transition；
2. 建立 rollout actor、versioned inference、experience、learner、weight-update 五个可恢复服务，并把最大 policy lag 固定为 1；
3. 用站立、抗扰恢复、静止球、移动球组成交替 curriculum，先跑不少于 8 个 continual seeds 与 8 个 single-task controls；
4. 实现 dead-unit、effective-rank、fresh-network-gap 和梯度冲突监测，再实现仅限 Plastic Expert 的低效单元重生；
5. 实现 forward self model，做 payload、friction、latency、motor gain/zero bias 变化检测，再训练和校准 Agency；
6. 在共同参考状态库上做 SelfCore 多阈值、多种子发现，并执行同规模 matched freeze/lesion/plastic-control/body-change；
7. 用 Boundary 反例重新学习接触时机估计器；在 holdout 证明无回归前，L2 phase/aim/recovery 保持关闭；
8. 只有历史平均下降 <3%、关键技能下降 ≤5%、critical regression=0、样本效率提升 ≥30% 且 Self 证据完整，才允许 SIM champion 晋级。

真正的下一里程碑不是“训练 loss 下降”，而是：真实 G1 MuJoCo rollout 在四卡异步闭环里持续产生新经验；移动球能力上升；站立、恢复和静止球不遗忘；候选失败时机器门主动拒绝并回滚。
