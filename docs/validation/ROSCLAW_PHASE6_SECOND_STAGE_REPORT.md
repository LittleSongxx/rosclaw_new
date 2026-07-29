# ROSClaw Phase 6 第二阶段实施与证据报告

日期：2026-07-26<br>
分支：`feedback-phase6`，基线 `origin/main@ef26afd`<br>
范围：P5-2 复查与优化、P5-3 控制契约、P5-4 十轮 ILC、F9/F13/F14 证据<br>
证据域：仅 `SIM`；没有连接、授权或控制真实机器人

## 1. 结论

第二阶段不只是补测试，还发现并修复了一个跨工况误触发问题，完成了可执行的十轮 ILC，并把 Holdout、历史动作回归和 canonical DDS loopback 跑通。

关键结果如下：

| 项目 | 结果 |
|---|---:|
| 同场景 0/35/65/80 N A/B | 4/4 通过；80 N 从跌倒/关节越界救回 SUCCESS |
| 多工况 Holdout | 11 个 regime，57 个仿真 Episode，协议通过 |
| Holdout 成功率 | `4/11 = 36.36%` → `5/11 = 45.45%` |
| Holdout 成功案例退化 | `0` 个 |
| Holdout 新增救援 | `1` 个 |
| Phase 4 历史动作回归 | 12/12 结果与物理轨迹完全一致 |
| 十轮 ILC 关节跟踪 RMS | `0.182407` → `0.162131`，下降 `11.12%` |
| 十轮 ILC 能耗代理 | `21364.2` → `16868.8`，下降约 `21.0%` |
| 十轮 ILC 安全干预 | 每轮 `0` |
| ILC 最终回放 / 错工况拒绝 | 通过 / 通过 |
| Unitree DDS loopback | 50 LowCmd、383 LowState、489 physics steps，通过 |
| executor chaos | 11/11 fail-safe，通过 |

这些结果仍不构成真实机器人晋级。反射器自身的 Receipt 局部误差没有证明下降；P5-3 技能反馈目前完成了受限控制契约和单测，还未接入 MuJoCo 动作相位；真实 G1 Canary 也没有获授权。因此整体结论仍是 `NEED_MORE_EVIDENCE`。

## 2. 这轮修掉的真实问题

第一阶段反射器只要看到 COM 横向位移超过阈值就会锁存。在支撑面摩擦从 `1.0` 降到 `0.75` 时，正常踢球姿态也可能短暂越过这个阈值。旧控制器会误判为外力扰动，把一个原本 `SUCCESS` 的案例修成关节越界和跌倒。

第二阶段给 COM 触发增加了瞬态变化率条件：只有“位移已经危险”且“位移仍在快速变化”时，COM 分支才允许锁存。roll/pitch 的紧急边界仍保留。固定控制器后重新验证：

- 80 N、摩擦 `1.0` 的救援仍然成功；
- 80 N、摩擦 `0.75` 从旧版的反馈后失败恢复为 `SUCCESS`；
- 0/35/65 N 正常场景继续零修正；
- Phase 4 的 12 个历史场景全部零修正，物理 trace digest 与 feedback-off 完全相同。

这说明“正常时透明”不能只在一个名义场景上判断，必须跨摩擦、延迟和本体偏置检查。

## 3. P5-3 GoalForge Skill Feedback

新增了 L2 技能反馈控制器和固定 profile，输入契约包括：

- 预测触球 phase 与期望触球 phase；
- 球相对踢球脚的横向误差；
- 是否已经触球；
- 触球后的 torso roll/pitch。

输出被 Safety Projector 限制为：

- `skill:kick_phase_rate`：触球前的动作相位速度指令；
- 右髋 yaw 和右踝 roll 小幅瞄准残差；
- 触球后腰和双髋恢复残差。

控制器已验证触球前 replanning 与触球后 recovery 的状态切换、限幅和 deadline 语义。但 `skill:kick_phase_rate` 尚未接入 RoboNaldo policy 的物理播放时钟，因此本报告不声称它提升了进球率，也不将 P5-3 标为完成。下一步必须在 moving-ball 场景做同场景 A/B 和严格回放。

## 4. P5-4 十轮真实物理 ILC

这里的“真实物理”指 MuJoCo 动力学真实执行，不是合成一串下降数字，也不是实物机器人。

每轮保存：

```text
policy joint reference - MuJoCo actual joint position
```

作为完整的 `[time, 29 joints]` error trajectory。下一轮生成平滑、有界、Body hash 和 regime hash 绑定的 joint feed-forward residual；最终残差与实时反馈相加后，还要再次经过 G1 联合残差边界和原力矩限制。

十轮选定结果：

| Trial | RMS | 能耗代理 | 学习倍率 | 是否接受更新 |
|---:|---:|---:|---:|---|
| 1 | 0.182407 | 21364.2 | 0.0 | 基线 |
| 2 | 0.168264 | 17728.5 | 5.0 | 是 |
| 3 | 0.165924 | 17912.4 | 1.0 | 是 |
| 4 | 0.162131 | 16868.8 | 1.0 | 是 |
| 5–10 | 0.162131 | 16868.8 | 0.0 | 否，回滚并保持 |

第 4 轮后不是继续“强行学习”，而是进入收敛平台。后续候选要么误差更高，要么能耗超过基线 `10%` 上限，所以事务式更新保留第 4 轮前馈。这是非增单调序列，最终有 `11.12%` 的实质下降。

为了选择安全步长，campaign 运行了 45 个 line-search probe，加 1 个初始 Episode 和 1 个最终回放，共 47 个仿真 Episode。所有 probe 都计入报告，没有隐藏。这个选择过程适合仿真；在真实机器人上不能把多个候选都直接试一遍，必须改成数字孪生预筛选、shadow 或更保守的一步更新。

十轮原始 error trajectory 位于：

```text
/code/rosclaw/phase6_evidence/ilc-validation-trials/
```

## 5. F9 多工况 Holdout

控制器冻结后，Holdout 同时覆盖：

- 支撑面摩擦：`1.00 / 0.75 / 0.60 / 0.55`；
- 控制延迟：`0 / 20 / 40 ms`；
- 关节零偏：`0 / +0.015 / -0.015 rad`；
- 侧向扰动：`0 / 65 / 80 N`；
- 两个多因素组合工况。

11 个 regime 中，baseline 成功 4 个，feedback 成功 5 个；没有把 baseline 成功案例变成失败，跌倒、关节越界和力矩越界的分类计数没有新增，11/11 feedback trajectory 均严格回放，deadline miss 为 0。

这个 `passed` 的准确含义是“已声明包络内安全不退化且至少救回一个案例”，不是“所有极端组合都能成功”。20/40 ms 延迟、低摩擦和负零偏的多个组合在 feedback off/on 下都失败，说明当前 emergency reflex 不能替代 P5-5 KickTwin、P5-6 embodiment adapter 或新 base policy。扰动时刻目前仍固定在 backend 的 `4.6–4.8 s`，随机扰动时刻尚未覆盖。

## 6. F13 历史动作回归

重新执行 Phase 4 冻结的 12 个 practice 场景。控制器在这些动作中全部保持零残差：

- feedback-off 与 feedback-on 的 GoalForge result 完全相同；
- 去掉 feedback-only trace 后，物理 trajectory digest 完全相同；
- 原本的成功、射偏或关节越界标签均原样保留，没有把旧失败伪装成通过。

因此 F13 在当前 RoboNaldo motion/参数集合与 `SIM` 证据域内通过。

## 7. F14 canonical DDS 与故障注入

使用 Unitree 官方 `rt/lowcmd` / `rt/lowstate` 语义、隔离 DDS domain `83` 和 loopback `lo` 接口运行：

- 29 个 G1 actuator；
- 50 条命令发布；
- 383 条状态接收；
- 489 个 MuJoCo physics step；
- IMU、命令反馈、finite state 全部有效；
- `real_hardware_opened=false`。

执行器 chaos 覆盖 agent kill、worker crash、DDS loss、state/IMU stale、policy timeout、触发前/恢复中取消、陈旧验证、daemon restart replay 和新 session。11/11 满足：触发前故障进入 `SAFE_STOP`；已经触发的物理后果进入 `RECOVERY`；旧 Action ID 不能重放；陈旧状态不能升级成 `TASK_VERIFIED`。

## 8. Promotion Gate 更新

| Gate | 第二阶段状态 | 说明 |
|---|---|---|
| F1/F2 | SIM 通过 | 250 Hz；正式 A/B 与 Holdout 无 deadline miss |
| F3 | 单测与 chaos 通过 | stale fail-closed；真实总线尚未做 |
| F4/F5 | 通过 | allowlist、finite、联合 residual limit、Safety Projector |
| F6 | **部分通过** | ILC joint RMS 下降 11.12%；emergency reflex 自身 Receipt 局部 RMS 未改善 |
| F7/F8 | SIM 包络通过 | Holdout 无新增成功退化或安全分类退化 |
| F9 | SIM 限定通过 | 11 regime；扰动时刻仍固定，极端组合不保证成功 |
| F10 | 通过 | wrong-body、wrong-regime、joint-order、shape 均拒绝 |
| F11 | 通过 | A/B 4/4、Holdout 11/11、ILC final strict replay |
| F12 | 通过 | Controller/spec/body/config hash 绑定 |
| F13 | SIM 通过 | 12/12 历史结果和物理轨迹完全相同 |
| F14 | SIM loopback 通过 | canonical LowCmd/LowState + 11 类 executor chaos |
| F15 | **未执行** | 无真实 G1 授权，明确不做 |

最终仍为 `NEED_MORE_EVIDENCE`，主要阻塞项是 F6 的 reflex-local 误差证明、P5-3 物理闭环、hard-real-time executor 和 F15 真实机器人 Canary。不能晋级为真实机器人 Controller Champion。

## 9. 证据与哈希

| 文件 | SHA-256 |
|---|---|
| `/code/rosclaw/phase6_evidence/feedback-validation.json` | `7722f33efb96b4782a448c67cab00ef5ca6a172243c9d452b50f60728bdb9bf8` |
| `/code/rosclaw/phase6_evidence/feedback-holdout.json` | `b207b53cf0f5031243c1399fd3c3b1cdd8c1faee7eef84fd87e7abb187d18649` |
| `/code/rosclaw/phase6_evidence/ilc-validation.json` | `75205d7d8eb4bc49de45883bd29370da231ac9ae25301267a4dbd1ccbf85c44c` |
| `/code/rosclaw/phase6_evidence/dds-chaos-v2/goalforge-chaos.json` | `496b81885bdc6147b5cbf2389c3f2b27a67fc3f713ecece7b1d574cb7ac3d271` |

所有 raw evidence 均在 source checkout 外。仓库只保留实现、测试和报告，不提交数百 MB 的轨迹或外部模型。

注：ILC 摘要后来由自进化闭环升级为 schema v2，物理结果未变化；新增了可重新加载的 selected feed-forward artifact，因此本表使用升级后的文件哈希。完整的候选生成和 F1-F15 结果见 `ROSCLAW_PHASE6_SELF_EVOLUTION_REPORT.md`。

## 10. 复现命令

```bash
.venv/bin/python -m rosclaw.entrypoint simforge validate g1-goalforge \
  --profile feedback-reflex \
  --asset-root /code/rosclaw/phase4_references/RoboNaldo/RoboNaldo_Deploy \
  --output /code/rosclaw/phase6_evidence/feedback-validation.json

.venv/bin/python -m rosclaw.entrypoint simforge validate g1-goalforge \
  --profile feedback-holdout \
  --asset-root /code/rosclaw/phase4_references/RoboNaldo/RoboNaldo_Deploy \
  --output /code/rosclaw/phase6_evidence/feedback-holdout.json

.venv/bin/python -m rosclaw.entrypoint simforge validate g1-goalforge \
  --profile feedback-ilc \
  --asset-root /code/rosclaw/phase4_references/RoboNaldo/RoboNaldo_Deploy \
  --output /code/rosclaw/phase6_evidence/ilc-validation.json

.venv/bin/python -m rosclaw.entrypoint chaos run g1-goalforge \
  --faults agent-kill,worker-crash,dds-loss,state-stale \
  --output-dir /code/rosclaw/phase6_evidence/dds-chaos-v2 \
  --asset-root /code/rosclaw/phase4_references/RoboNaldo/RoboNaldo_Deploy \
  --unitree-mujoco-root /code/rosclaw/phase4_references/unitree_mujoco \
  --domain-id 83
```

四张 A6000 已确认健康，但本阶段的真值执行是 CPU MuJoCo，控制器和 ILC 不需要 GPU。没有为了制造“4 卡使用率”而运行无意义训练。P5-7/P5-8 teacher-student residual campaign 才是四卡的合理使用阶段。

## 11. 代码质量与测试结果

- ruff 与格式检查：通过；
- mypy：40 个相关 source file 无问题；
- Feedback/SimForge 专项：`144 passed, 1 skipped, 3 deselected`；
- 仓库全量默认环境：`5239 passed, 67 skipped, 27 deselected`，另有 4 个 LeRobot 集成项失败；
- 失败原因：pytest 隔离 home 没有自动发现外部 LeRobot runtime；
- 新建隔离 runtime：`/code/rosclaw/phase6_runtime/lerobot-0.6.1`，状态 `ready`，Python 3.12.13、LeRobot 0.6.1、Torch 2.11.0+cu128；
- 显式设置 `ROSCLAW_TEST_LEROBOT_PYTHON` 后，4 个失败项单独复跑为 `4 passed`。

LeRobot runtime 位于 source checkout 外，不属于本次代码提交。这个结果证明 4 项不是 Feedback Plane 回归，但也暴露了测试 fixture 对全局 external runtime 的自动发现不够稳定；后续应统一 runtime 注册路径，而不是依赖机器状态。
