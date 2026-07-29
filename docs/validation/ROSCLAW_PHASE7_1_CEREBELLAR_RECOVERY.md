# ROSClaw Phase 7.1 — G1 射后小脑恢复闭环

## 结论

本阶段针对 Hat Trick v3 中“球已经踢出，但 G1 射后持续晃动”的问题，完成了一个
SIM-only、接触门控、可严格重放的两段式小脑恢复闭环。最终 Hat Trick 三脚全部保持
进球、目标区命中和安全结果，尾段综合晃动分别下降 **42.7%、33.8%、58.1%**。

证据上限仍是 **SIM**。本实现未打开 DDS、ROS 执行接口或真实机器人运输层，不能据此
宣称真实 G1 安全性。

## 根因复核

v3 的 `post_kick_stability_time_sec` 实际表示“结尾连续满足宽松姿态阈值的时长”，不是
“机器人多久停止晃动”。第三脚该值为 0 秒却仍被判定进球成功，说明任务成功 Gate 没有
覆盖宣传视频中明显的射后振荡。

轨迹复核还发现：

- 原 L1 平衡反射只在 kick phase 0.32–0.60 生效，适合受扰瞬态；
- 直接延长同一 PD 反射会在落脚、双支撑和换脚阶段改变控制符号，造成更大振荡甚至越限；
- RoboNaldo 射门 prior 在动作尾段仍输出非静态全身目标，受扰后容易停留在单脚支撑和持续滚转；
- 因此不能简单“加大增益”，需要将瞬态救球与射后站稳分开。

## 实现

### 1. 两段式小脑控制

控制链变为：

```text
RoboNaldo frozen kick prior
  → L1 high-rate balance reflex（受扰瞬态闭环）
  → ball-contact latch
  → kick-foot landing latch
  → 100-frame smooth recovery blend
  → qualified standing pose + bounded roll posture bias
  → existing torque guard / joint limits
```

新增的 `G1CerebellarRecoveryController`：

- 绑定精确的 Body hash、motion hash、scenario commitment 和 29 关节顺序；
- 触球前完全透明；
- 右脚重新落地前完全透明；
- 从 policy frame 420 开始，以 smoothstep 在 100 帧内渐进混合；
- 只改变目标位置，不直接产生力矩；
- 输出 activation、blend fraction、contact/landing latch 和严格重放收据；
- 原始轨迹增加 `recovery_active` 与 `recovery_blend_fraction`，便于复核因果时序。

### 2. 真正的射后稳定指标

新增 `G1RecoveryQuality`，从真实 MuJoCo 轨迹计算：

- 躯干角速度 RMS；
- 躯干倾角 RMS；
- COM 横向速度 RMS；
- 骨盆平面速度 RMS；
- 全身关节速度 RMS；
- 双脚支撑比例与最终双脚支撑；
- 连续动态稳定时长与 settling time；
- 由上述分量组成、仅用于同轨迹 A/B 的 tail wobble index。

Gate 要求目标结果和球速不退化、无跌倒/关节/力矩安全回归、最终双脚支撑、尾段角速度和
骨盆速度至少下降 5%，综合晃动至少下降 10%。静态倾角单独透明报告，并受 0.35 rad 稳态
边界约束，不再错误地把“安静但略有前倾”当成振荡回归。

### 3. Stability-Plasticity / Regime Gate

未校准的恢复段不会无条件运行。实际 holdout 发现固定参数在 0.8 低摩擦和 60 N 扰动下会
放大尾段运动；20 ms 控制延迟时原始 prior 已发生关节越限。因此新增 fail-closed Regime
Gate：

- 支撑摩擦低于 0.95：拒绝恢复残差；
- 控制延迟高于 5 ms：拒绝恢复残差；
- 非零扰动低于 70 N 或高于 80 N：拒绝恢复残差；
- 拒绝时 `recovery_active` 始终为 false，物理轨迹与原策略逐元素相同；
- 拒绝原因进入不可变 recovery receipt。

这不是宣称上述工况已经解决，而是避免尚未验证的可塑性破坏已有稳定性。后续需要为这些
工况单独学习和验证新的 recovery expert。

## 最终严格 A/B 结果

每一脚都执行 recovery-off baseline、baseline strict replay、recovery-on candidate、candidate
strict replay。第三脚另保留 Feedback OFF 的失败对照。三脚 strict replay 均通过。

| 场景 | 综合晃动下降 | 尾段角速度下降 | 尾段骨盆速度下降 | 尾段关节速度下降 | 原/新稳定尾段 |
|---|---:|---:|---:|---:|---:|
| 九宫格重炮 | 42.7% | 76.2% | 63.4% | 73.3% | 7.58 / 7.58 s |
| 滚动球迎射 | 33.8% | 60.1% | 58.0% | 75.3% | 3.72 / 5.08 s |
| 80 N 受扰救球 | 58.1% | 80.9% | 42.6% | 80.4% | 0.00 / 3.66 s |

第三脚恢复前最终不是双脚支撑，且没有满足动态稳定窗口；恢复后最终双脚支撑，并获得
2.46 秒连续动态稳定尾段。目标误差保持 0.096 m，球速保持 4.93 m/s，未出现跌倒、关节
限位或力矩限位回归。

滚动球的首次 settling time 从 5.32 s 轻微变为 5.44 s，但其最终连续稳定时长由 3.72 s
提升到 5.08 s，尾段三个动态速度指标均显著下降；报告保留这一细节，不用单一指标掩盖。

## 证据与视频

- 报告：`/code/rosclaw/phase7_1_evidence/hat-trick-v4-final/goalforge-hat-trick.json`
- 报告 SHA-256：`3428f72b3560ec02414e813ce05b72b2a43704d740edbd496582068c9146dbe7`
- 视频：`/code/rosclaw/phase7_1_evidence/hat-trick-v4-final/rosclaw-g1-goalforge-hat-trick-stable.mp4`
- 视频 SHA-256：`3500b4db3e1c5e5c5d2e44ecf4a405845cb6bc398355c8f9b256148a3744a770`
- 视频规格：1280×720、30 FPS、24.4 秒、732 帧、约 3.4 MiB

视频仍是 visualization-only。新版时间线在看完球路后把镜头切回 G1，并延长到触球后 7.5
秒，避免在恢复动作真正发生前结束。第三脚保留 Feedback OFF / Feedback + Cerebellum
并排对照。

## 软件回归

- 变更文件 `ruff check` 与 `ruff format --check`：通过；
- 变更的 5 个 source 文件 mypy：通过；
- SimForge + Feedback control + GoalForge adapter：134 passed、1 skipped、3 deselected；
- 全仓默认 pytest：5324 passed、67 skipped、27 deselected；4 个 LeRobot 导出测试因默认解释器
  未配置外部 runtime 失败；
- 显式设置
  `ROSCLAW_TEST_LEROBOT_PYTHON=/code/rosclaw/phase6_runtime/lerobot-0.6.1/bin/python`
  后，全部 `tests/integrations/test_lerobot*.py` 为 230 passed、1 skipped；上述 4 项单独重跑
  也全部通过；
- 仓库级 `ruff check .` 仍有 245 个既存问题，集中在本分支未修改的
  `examples/rh56_rps` 等文件；未借本任务改动无关示例。

复现命令：

```bash
.venv/bin/python -m rosclaw.entrypoint goalforge hat-trick run \
  --asset-root /code/rosclaw/phase4_references/RoboNaldo/RoboNaldo_Deploy \
  --output-dir /an/external/new/hat-trick-v4 \
  --source-checkout /code/rosclaw/rosclaw_test

.venv/bin/python -m rosclaw.entrypoint goalforge hat-trick export \
  /an/external/new/hat-trick-v4/goalforge-hat-trick.json \
  --asset-root /code/rosclaw/phase4_references/RoboNaldo/RoboNaldo_Deploy \
  --output /an/external/new/hat-trick-v4/hat-trick-stable.mp4 \
  --source-checkout /code/rosclaw/rosclaw_test --fps 30
```

## 未完成与下一步

当前恢复 expert 只在明确校准的 SIM regime 内启用。下一步应优先：

1. 为 0.8–0.95 摩擦、5–20 ms 延迟、40–70 N 扰动分别收集独立 episode；
2. 使用 Forward Self Model / Regime Encoder 预测恢复段，而不是读取仿真 Scenario 标签；
3. 建立 Nominal / LowGrip / Disturbed recovery experts，并用 Anchor 防止互相干扰；
4. 将 settling-time、双脚支撑和 wobble 指标接入 Candidate Promotion；
5. 在更多随机扰动时刻和方向上做多种子验证，再扩大校准区间。
