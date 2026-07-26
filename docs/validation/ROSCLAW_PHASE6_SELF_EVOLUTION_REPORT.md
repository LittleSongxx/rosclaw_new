# ROSClaw Phase 6 自进化闭环实施与证据报告

日期：2026-07-26<br>
分支：`feedback-phase6`<br>
证据域：仅 `SIM`；无真实机器人授权、连接或控制

## 1. 结论

本阶段把“ILC 学到一个更好的残差”升级成了 ROSClaw 可审计的自进化闭环：

```text
Feedback / Holdout / ILC / DDS evidence
                    ↓
重新加载并验证 selected feed-forward artifact
                    ↓
绑定 Body + Regime + ControllerSnapshot + safety kernel
                    ↓
生成内容寻址 Controller Candidate
                    ↓
F1-F15 独立评估
          ↙                    ↘
缺证据：保留离线候选       失败：拒绝并回滚
```

实际生成的 SIM-only candidate 为：

```text
sha256:8aa55b1080ad7f260e6966b8424c7add3283486e84c47073c0c0ac0cd23d8594
```

F1-F5、F7-F14 共 13 项通过；F6 和 F15 缺证据，最终决策为：

```text
NEED_MORE_EVIDENCE
```

系统没有激活候选，没有修改 Registry，没有发送硬件命令，并记录父 ControllerSnapshot 为 rollback target。这是一次完整且 fail-closed 的自进化循环，而不是自动晋级。

## 2. 为什么原实现还不算完整自进化

上一阶段已经完成十轮 ILC，并把关节跟踪 RMS 从 `0.182407` 降至 `0.162131`，下降 `11.12%`。但是最终前馈只存在于每轮 trial NPZ 和摘要哈希中，存在三个产品缺口：

1. 无独立 selected artifact，后续流程不能只加载最终胜者；
2. 摘要没有证明“磁盘数组重新加载后仍得到同一个 trajectory hash”；
3. ILC、A/B、Holdout、历史回归和 DDS chaos 尚未汇合为一个候选晋级决策。

因此，“学习算法跑过”不等于“ROSClaw 已形成自进化闭环”。本阶段针对这三个缺口补齐产品链路。

## 3. 新增实现

### 3.1 可重新加载的 ILC Candidate

ILC schema 升级为 `rosclaw.g1_ilc.validation.v2`。十轮 campaign 结束后，系统单独写出：

```text
/code/rosclaw/phase6_evidence/ilc-validation-trials/selected-feedforward.npz
```

候选 manifest 绑定：

- 29 个 G1 joint 的固定顺序；
- `[652, 29]` 数组 shape；
- Body hash 与 Operating Regime hash；
- residual hard limit `0.008 rad`；
- source FeedbackReceipt hashes；
- value hash、trajectory hash、artifact SHA-256；
- ILC lineage trial 与 campaign selected trial。

实际重新加载验证得到：

| 字段 | 值 |
|---|---|
| ILC candidate hash | `sha256:1cd04a0cdbd566b098062e0d9e83fba021cd7de67206859e2a35dbff5dfff160` |
| feed-forward trajectory hash | `sha256:281e7486ce57a651bb24f5615fa97ac72cc1bbea9304b280725016b5021765e6` |
| artifact hash | `sha256:ffca091f3e022e95b226ff5b8b121a579f15cc6c4880b092198caad766e4f598` |
| 接受更新 | Trial 2、3、4，共 3 次 |
| 回滚/保持 | Trial 5-10，共 6 次 |

加载器使用 `allow_pickle=False`，限制 JSON/NPZ 大小，检查数组名称、shape、finite、残差边界、artifact hash、value hash、trajectory hash、candidate hash 和最终 trial 绑定。任一字段被修改都会停止候选生成。

### 3.2 Controller Candidate 与回滚谱系

新的组合候选不是单独一份数组，而是：

```text
current G1 balance controller
+ selected regime-bound ILC feed-forward
+ immutable safety-kernel commitment
+ four source-evidence commitments
```

候选同时绑定：

- Body hash；
- RoboNaldo kick-prior hash；
- controller / loop-spec / ControllerSnapshot hash；
- ILC Regime 与 feed-forward hash；
- Safety Projector limits 与三类 fallback；
- A/B、Holdout、ILC、DDS chaos 的 canonical content hash；
- parent snapshot 与 rollback target。

证据文件的磁盘路径不参与候选身份，因此证据可以迁移；证据内容哈希参与身份，因此内容不能偷换。

### 3.3 机器执行 F1-F15

聚合器不是读取报告里的一个 `passed=true` 就放行。它重新执行过期观测和极端残差探针，重新构建当前 `ControllerSnapshot`，并核对旧 Receipt 中的 controller/spec/snapshot hash。

| Gate | 状态 | 本阶段机器判定 |
|---|---|---|
| F1 | PASS | 250 Hz contract、jitter 和 dropped-frame 证据一致 |
| F2 | PASS | A/B、Holdout、ILC 无 deadline miss |
| F3 | PASS | stale observation 清空 residual 并进入 base-policy fallback |
| F4 | PASS | runtime residual 与 ILC residual 均不超过固定上限 |
| F5 | PASS | residual 经同步 Safety Projector |
| F6 | MISSING | ILC RMS 改善 11.12%，但 emergency reflex 局部误差仍未证明因果下降 |
| F7 | PASS | A/B 与 Holdout fall rate 不增加 |
| F8 | PASS | torque/joint-limit 分类不退化 |
| F9 | PASS | 11-regime disturbance Holdout 通过 |
| F10 | PASS | wrong-regime feed-forward 拒绝 |
| F11 | PASS | A/B、Holdout、ILC strict replay 通过 |
| F12 | PASS | 当前代码重新导出同一 ControllerSnapshot |
| F13 | PASS | 12 个历史动作结果与物理轨迹不变 |
| F14 | PASS | isolated canonical DDS 与 11 类 chaos 通过 |
| F15 | MISSING | 无候选绑定且独立证明的真实 G1 Canary |

F15 没有 `--canary-passed` 布尔开关，避免调用者自报成功。未来必须增加候选绑定、独立签名和明确授权的真实 Canary 流程。

## 4. 故障与反例验证

新增测试覆盖：

- 正常证据生成离线候选，但 F6/F15 缺失时严格返回 `NEED_MORE_EVIDENCE`；
- 候选 artifact 追加一个字节后，加载阶段因 SHA-256 不匹配直接失败；
- 候选不自动激活、不修改 Registry、不发送硬件命令；
- ILC 必须至少十轮、最终 trial 必须绑定 selected feed-forward；
- 输出仍必须位于 source checkout 外。

这意味着自进化链既能“向前产生候选”，也能“发现证据或工件被破坏后停止并保留回滚目标”。

全仓复查还发现并修复了两个与本轮候选无直接关系、但影响 ROSClaw 可复现安装和测试闭环的问题：

- 固定 LeRobot 0.6.1 runtime 是无 pip 的最小 uv 环境。旧 RH56 reference-policy 安装器只能执行 `python -m pip`，现在会自动回退到 `uv pip --python`，若系统无 uv 再尝试 `ensurepip`；
- `MockModbusTransport.max_step_per_tick=50` 的实现错误地又除以 `tick_hz=100`，每个反馈 tick 实际只移动 `0.5 raw`。修正后 synthetic fixture 按契约每 tick 最多移动 50 raw，合法的 20 raw 小步能在反馈中被验证。真实 SerialModbusTransport 未修改。

全量仓库首轮为 `5249 passed, 59 skipped, 27 deselected`，5 项失败全部指向外部 LeRobot runtime 缺少 RH56 plugin。安装插件后其中 4 项通过，剩余 1 项暴露上述 mock tick 缺陷；完成修复后，相关 RH56 transport、execution kernel、reference policy 与 real-worker fixture 共 `31 passed`。Feedback + SimForge 完整专项为 `147 passed, 1 skipped, 3 deselected`。

## 5. 复现命令

```bash
# 先运行 47 个 MuJoCo Episode，产出 v2 摘要和 selected artifact
.venv/bin/python -m rosclaw.entrypoint simforge validate g1-goalforge \
  --profile feedback-ilc \
  --asset-root /code/rosclaw/phase4_references/RoboNaldo/RoboNaldo_Deploy \
  --output /code/rosclaw/phase6_evidence/ilc-validation.json

# 聚合证据、重新加载候选并执行 F1-F15
.venv/bin/python -m rosclaw.entrypoint simforge validate g1-goalforge \
  --profile feedback-evolution \
  --feedback-evidence /code/rosclaw/phase6_evidence/feedback-validation.json \
  --holdout-evidence /code/rosclaw/phase6_evidence/feedback-holdout.json \
  --ilc-evidence /code/rosclaw/phase6_evidence/ilc-validation.json \
  --chaos-evidence /code/rosclaw/phase6_evidence/dds-chaos-v2/goalforge-chaos.json \
  --output /code/rosclaw/phase6_evidence/feedback-evolution.json
```

第二条命令当前预期退出码为 `2`，因为决策是 `NEED_MORE_EVIDENCE`；输出文件仍会完整落盘。这不是 crash，而是 Promotion Gate 阻断激活。

## 6. 证据哈希

| 文件 | SHA-256 |
|---|---|
| `feedback-validation.json` | `7722f33efb96b4782a448c67cab00ef5ca6a172243c9d452b50f60728bdb9bf8` |
| `feedback-holdout.json` | `b207b53cf0f5031243c1399fd3c3b1cdd8c1faee7eef84fd87e7abb187d18649` |
| `ilc-validation.json` | `75205d7d8eb4bc49de45883bd29370da231ac9ae25301267a4dbd1ccbf85c44c` |
| `selected-feedforward.npz` | `ffca091f3e022e95b226ff5b8b121a579f15cc6c4880b092198caad766e4f598` |
| `goalforge-chaos.json` | `496b81885bdc6147b5cbf2389c3f2b27a67fc3f713ecece7b1d574cb7ac3d271` |
| `feedback-evolution.json` | `382c1a05f41397dc92f097d864bd4a67c144f80cc0ce65e3587de40f74e6ba62` |

## 7. 当前能力边界与下一步

本阶段实现的是 C2：同工况、回合间、有界前馈轨迹自进化，不是 C3 神经策略在线自由改权重。下一步优先级为：

1. 修正 F6 的评价窗口和 controller-local error attribution，证明反射触发后的局部误差确实下降；
2. 实施 P5-5 KickTwin Online Identification，针对当前 Holdout 暴露的低摩擦、20/40 ms 延迟和负零偏失败；
3. 将 P5-3 kick phase modulation 接入 MuJoCo policy clock，形成 L2 同一脚物理反馈；
4. 等真实 G1、人员、场地和独立观测授权后，再设计 F15 Canary；在此之前不得提高 activation ceiling。

因此，ROSClaw 现在已经能自主产生、验证、拒绝和回滚一个控制候选；它还不能声称这个候选已经具备真实机器人部署资格。
