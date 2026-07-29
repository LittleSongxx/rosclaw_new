# ROSClaw Phase 6 第一阶段实施报告

日期：2026-07-26<br>
分支：`feedback-phase6`（基于当时最新 `origin/main`：`ef26afd`）<br>
范围：`rosclaw_验证和优化v6.md` 的 P5-0、P5-1、P5-2，以及 P5-4 基础协议<br>
证据域：仅 `SIM`，没有连接真实机器人

> 本文冻结第一阶段当时的结论。十轮 ILC、跨工况 Holdout、历史回归、
> COM 瞬态误触发修复和 DDS loopback 的后续结果见
> `ROSCLAW_PHASE6_SECOND_STAGE_REPORT.md`。

## 1. 结论先说

本轮已经把 ROSClaw 从“动作结束后分析成败”推进到“同一动作尚未结束时，读取身体误差并施加有界修正”的第一版 Feedback Plane。

在同一个 G1 踢球动作、同一个场景和同一组参数下：

| 侧向扰动 | Feedback OFF | Feedback ON | 结果 |
|---:|---|---|---|
| 0 N | SUCCESS | SUCCESS | 正常动作完全透明，无退化 |
| 35 N | SUCCESS | SUCCESS | 未触发紧急反射，无退化 |
| 65 N | SUCCESS | SUCCESS | 未触发紧急反射，无退化 |
| 80 N | 关节越界并跌倒 | SUCCESS、未跌倒 | 同一脚球在动作中被救回 |

80 N 档位的躯干 roll 峰值从 `0.6588 rad` 降到 `0.4457 rad`，约下降 `32.3%`；关节越界从 `true` 变成 `false`，跌倒从 `true` 变成 `false`。目标误差保持 `0.0960 m`，说明反射没有靠放弃踢球来换取站稳。

四组 Feedback ON 共执行 `13,040` 个反馈 tick，最终正式运行 deadline miss 为 `0`，每组 p99 控制计算延迟为 `0.137–0.139 ms`，相对 `4 ms` deadline 有充分余量；四组物理轨迹严格回放均通过。

原始报告位于仓库外：`/code/rosclaw/phase6_evidence/feedback-validation.json`。

## 2. 通俗解释：这次到底增加了什么

原来的踢球策略更像一套事先排好的舞蹈：每一拍关节应该到哪里，基本在动作开始前就决定了。机器人被推一下以后，这套舞蹈仍会照常播放，所以严重扰动可能在几秒后逐渐放大成失衡和跌倒。

Feedback Plane 相当于增加了“小脑反射”：

1. 每 4 ms 看一次躯干角度、重心相对支撑脚的位置、脚底滑移和关节状态；
2. 判断当前动作是否仍在允许触发反射的阶段；
3. 正常时输出严格为零，让原策略继续负责；
4. 只有重心偏移跨过危险阈值，才锁存启动髋、踝和腰部的小幅修正；
5. 每个修正先经过 Safety Projector，最多只允许 `0.04–0.08 rad`；
6. 原来的 500 Hz PD 和力矩硬限制仍然是最终执行边界。

因此它不是换了一个新的踢球模型，也不是让大模型在 250 Hz 发指令，而是在已验证动作外面加了一个速度快、范围小、失效会回退的纠偏层。

## 3. 实现内容

### P5-0：反馈协议

已实现并内容寻址：

- `FeedbackLoopSpec`
- `FeedbackFrame` / `ErrorState`
- `ResidualCommand`
- `FeedbackReceipt`
- `ControllerSnapshot`
- `AdaptationSnapshot`

所有身体、控制器、参考、观测、规格、命令和轨迹均可由 SHA-256 绑定。原有 `rosclaw.feedback` 用户反馈/遥测 API 保持兼容，新控制面是增量扩展。

### P5-1：Feedback Runtime

已实现：

- 绝对 deadline 的固定频率调度器；
- 同步 `tick()` 热路径，不经过 EventBus；
- 观测新鲜度、时间单调性和 finite 检查；
- error / derivative / bounded integral 状态；
- 预分配 NumPy ring buffer；
- deadline、jitter、drop、observation age 统计；
- 非法输出、陈旧观测和 deadline miss 回退；
- 同步 Safety Projector；
- 运行后异步生成 Receipt；
- 记录 deadline 决策的严格回放。

### P5-2：G1 Balance Reflex

已实现 G1 专用、相位感知的紧急平衡反射。输入包含 torso roll/pitch、COM 相对支撑脚位置、支撑脚滑移和 29 个关节状态；输出只覆盖腰、双髋和双踝的小幅目标残差。

调试过程中发现，持续开启的普通反馈器会和“有意的大幅踢球动作”互相打架：它能救 80 N 扰动，却可能把 0 N 正常动作修坏。最终采用“早期危险检测窗口 + 锁存 + 有限纠正窗口”的工况感知设计。0/35/65 N 不触发，80 N 的 COM 偏移跨阈值后才触发。

### P5-4：ILC 基础

已实现但尚未完成十脚 G1 实验：

- 有界 ILC 更新；
- Body hash 与 regime hash 隔离；
- 错身体、错工况和轨迹 shape 变化拒绝；
- 有限容量轨迹记忆；
- 误差单调下降、安全干预不增加、能耗增长不超限的收敛检查；
- `AdaptationSnapshot` 绑定来源 Receipt。

## 4. 安全、回放与失败发现

第一次连续反馈版本并没有直接通过。它暴露了两个真实问题：

- 正常踢球中的预期躯干运动会被控制器误当成误差，0 N 场景可能被修到跌倒；
- 一次 deadline miss 若直接把残差清零，会产生控制不连续。

最终处理是：正常工况零残差、触发窗口限制、触发后锁存、输出统一投影，以及 G1 deadline miss 保持上一条已安全投影的残差一拍。陈旧观测和非法输出仍彻底撤销残差。

80 N 救援中发生 `153` 次单输出限幅。这不是越界执行：实际送入 PD 的 projected residual 从未超过规格；它说明控制器正处于安全投影边界。这个数据会保留为后续优化目标，而不是隐藏。

严格回放不是重新“碰运气”计时。第一次运行记录每 tick 的计算时延和 deadline 决策；回放使用相同输入与相同 deadline 决策，要求命令 hash、最终结果和完整 trajectory digest 一致。

## 5. 与 v6 Promotion Gate 的对应关系

| Gate | 当前状态 | 证据/缺口 |
|---|---|---|
| F1 固定频率 | SIM 通过 | 250 Hz，13,040 tick，0 dropped frame |
| F2 deadline | SIM 通过 | 正式运行 0 miss，p99 最大约 0.139 ms / 4 ms |
| F3 stale fail-closed | 单测通过 | 陈旧观测撤销残差；真实 DDS 注入待做 |
| F4/F5 残差边界与投影 | 通过 | 非 allowlist/non-finite 拒绝；实际输出全部投影 |
| F6 tracking error 改善 | **证据不足** | 任务结果和 roll 改善，但 Receipt 局部 RMS 未单调下降 |
| F7/F8 跌倒、关节、力矩不退化 | 4 档 A/B 通过 | 仍需更广 Holdout |
| F9 disturbance Holdout | **未完成** | 当前只有同一名义场景 4 档扰动 |
| F10 wrong-regime | 基础单测通过 | ILC 错 Body/Regime 拒绝；G1 campaign 待做 |
| F11 strict replay | 通过 | 4/4 命令与物理轨迹严格回放 |
| F12 ControllerSnapshot | 通过 | Controller/spec/body/config 全部 hash 绑定 |
| F13 historical regression | **未完成** | 需跑 Phase 4 全历史动作集 |
| F14 DDS chaos | **未完成** | 未打开真实或 loopback DDS canonical path |
| F15 Canary | **未完成** | 未获真实机器人授权，本轮明确不做 |

因此本轮结论是：**GoalForge Reflex A/B 实验通过，但控制器仍是 `NEED_MORE_EVIDENCE`，不能晋级真实机器人或 Registry Champion。**

另外，当前是 Python 仿真参考实现，不是 hard-real-time/零分配执行器。虽然数值 ring buffer 已预分配，`FeedbackFrame` 和 trace record 在热路径仍会创建 Python 对象；真实硬件晋级前仍需 shared-memory 加 C++/Rust executor、调度器实测和 WCET 证明。

## 6. 参考项目与本地固定版本

本地已有参考 checkout，因此没有重复下载或覆盖：

- OpenDriveLab RoboNaldo：顶层 `370b9ac6`，Deploy 子模块 `f60f2445`；使用其 29-DoF free-kick policy、motion 和带球场景。
- Unitree `unitree_mujoco`：`ae6a8403`；核对 2 ms MuJoCo step、LowCmd/LowState 与官方 G1 模型语义。
- Unitree `unitree_rl_mjlab`：`1425b15f`；核对 policy decimation、固定调度和 G1 部署结构。

没有把参考项目硬件 transport 复制进 ROSClaw，也没有把外部模型资产 vendoring 到仓库。

## 7. 验证命令

```bash
.venv/bin/python -m pytest -q tests/feedback/control \
  tests/simforge/test_g1_feedback_validation.py \
  tests/simforge/test_goalforge_phase4_contracts.py \
  tests/feedback

.venv/bin/python -m rosclaw.entrypoint simforge validate g1-goalforge \
  --profile feedback-reflex \
  --asset-root /code/rosclaw/phase4_references/RoboNaldo/RoboNaldo_Deploy \
  --output /code/rosclaw/phase6_evidence/feedback-validation.json
```

代码质量结果：ruff 与格式检查通过；新增模块 mypy 检查 `36 source files` 无问题；Feedback/SimForge 相关套件 `131 passed, 1 skipped, 3 deselected`。仓库默认全量测试为 `5226 passed`，另有 4 个 LeRobot 集成项因 pytest 隔离 home 丢失全局 external-runtime 路径而失败；显式传入已安装的 LeRobot 0.6.1 Python 后，这 4 项单独复跑全部通过。这不是 Feedback Plane 代码回归，但说明测试环境的 LeRobot runtime 发现逻辑仍应后续统一。

## 8. 下一阶段建议顺序

1. 扩展多 seed、多摩擦、零偏、延迟和扰动时刻的 Private Holdout，解决 F6/F9/F13；
2. 将 `FeedbackReceipt` 通过异步旁路落入 Practice/MCAP，不进入 250 Hz 热路径；
3. 完成十脚同工况 ILC，证明误差单调下降且安全/能耗不恶化；
4. 加入 KickTwin 在线参数辨识和短历史 embodiment latent；
5. 在四张 A6000 上训练 teacher-student residual，CPU MuJoCo 做最终真值；
6. 完成 Unitree loopback DDS chaos；真实 G1 Canary 继续等待单独授权。

四张 A6000 本轮已检查可用，但没有为了“用 GPU 而用 GPU”。当前工作是 250 Hz 控制运行时和 CPU MuJoCo 真值闭环；GPU 的合理使用点是后续 residual student 与大规模扰动训练。
