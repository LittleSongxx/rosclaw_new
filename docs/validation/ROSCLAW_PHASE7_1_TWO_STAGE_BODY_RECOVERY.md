# ROSClaw Phase 7.1：G1 踢球后退与抖动分析及两阶段身体恢复

## 1. 结论

G1 踢球后出现“先向前晃、随后后退、最后轻微抽动”，不是单一关节增益过大，而是三个动作阶段没有正确衔接：

1. 触球时骨盆仍带有明显的前向、侧向和竖直速度；
2. 踢球先验在右脚重新落地后仍继续播放较长的 recovery 片段；
3. ROSClaw 原恢复器直到动量已经反向后才开始向站立姿态混合，只能把机器人停住，无法阻止先前发生的后退。

本阶段把恢复改成接触与落脚双门控的两阶段动作：

- 第一阶段提前卸力，逐步退出踢球先验；
- 第二阶段增加站立权重、减小侧倾，并仅用腰俯仰做缓慢回正；
- 上肢目标同步低通，避免手臂和躯干在策略帧之间抽动；
- 新增以后续球飞行方向为坐标轴的后退与侧摆指标，候选必须相对直接上一版 parent 通过严格重放 A/B 门禁。

最终三球 Hat Trick 闭环通过。第三球在相同种子、相同 80 N 干扰、相同球速和落点下，后退回撤降低 13.2%，侧向峰值回摆降低 29.3%，尾段 wobble 降低 10.3%。

## 2. 为什么踢完会后退和抖动

### 2.1 触球后仍有未卸掉的身体动量

上一版轨迹的第一次球接触发生在约 5.28 s。此时骨盆速度约为：

- 前向：0.190 m/s；
- 侧向：0.425 m/s；
- 竖直：0.317 m/s。

右脚约在触球后 0.30 s 再次落地。落脚只代表获得了支撑，并不代表动量已经消失。机器人仍需要通过迈步、髋踝协调、摆臂和躯干姿态把这些动量安全地导入地面。

### 2.2 恢复开始太晚

上一版上肢平滑在策略帧 400 才启动，约为触球后 2.92 s；站立姿态混合在策略帧 421 后才真正激活，约为触球后 3.40 s。此时骨盆已先向前、向侧方运动，随后踢球先验的剩余恢复动作又把身体带回相反方向。

轨迹表现为：

- 骨盆先走到前向峰值；
- 然后反向回撤约 0.543 m；
- 侧向峰值到最终位置的回摆约 0.881 m；
- 晚启动的站立混合只负责最后“刹住”，不能消除已经发生的往返运动。

### 2.3 “抖动”和“姿态不正”必须分开看

搜索中曾出现一种候选：关节 jerk 和骨盆速度都下降，但 wobble 指数反而上升。分解后发现其原因不是继续乱动，而是机器人以较大的后倾姿态安静地停住。wobble 指数同时包含：

- 躯干角速度；
- 骨盆平面速度；
- 躯干倾角；
- 全身关节速度。

因此仅做动作低通，或者仅看视频是否“抽动”，都会漏掉停姿不正的问题。

## 3. 实施内容

### 3.1 两阶段小脑恢复器

`G1CerebellarRecoveryConfig` 新增第二阶段参数：

- `settling_start_policy_frame`；
- `settling_blend_frames`；
- `settling_standing_pose_blend`；
- `settling_roll_posture_bias_rad`；
- `settling_waist_pitch_bias_rad`。

两个阶段都使用 smoothstep，不产生瞬时目标跳变。控制器仍保持以下因果边界：

- 未观察到球接触时透明；
- 未观察到踢球脚重新落地时透明；
- 超出已校准摩擦、延迟和扰动力范围时 fail closed；
- 只修改 MuJoCo SIM 中的关节目标，不发送力矩，也不开启机器人传输。

最终候选参数为：

```text
第一阶段：frame 300，100 帧混合，站立权重 0.30，roll bias -0.05 rad
第二阶段：frame 400，100 帧混合，站立权重增至 0.45，roll bias 收至 -0.02 rad
腰部回正：waist pitch +0.09 rad
上肢平滑：alpha 0.60，从 frame 300 开始
```

只对腰俯仰施加回正，而不使用髋、踝共同硬顶，是因为局部搜索显示后者容易把姿态修正变成新的步态振荡，甚至触发关节限制。

### 3.2 可审计的第二阶段证据

恢复 receipt 升级为 v3，增加：

- 第二阶段首次激活的策略帧与仿真时间；
- 第二阶段峰值比例；
- 完整两阶段配置；
- 控制器、Body、motion 和 regime 哈希。

轨迹增加 `recovery_settling_fraction`。最终证据中：

- 第一阶段激活：policy frame 300，约 6.18 s；
- 第二阶段激活：policy frame 401，约 8.20 s；
- 两阶段峰值比例均为 1.0；
- candidate 与 parent 均严格重放一致。

### 3.3 身体效果指标与晋级门禁

恢复质量 schema 升级为 v4，新增：

- `post_contact_forward_peak_advance_m`；
- `post_contact_backward_reversal_m`；
- `post_contact_lateral_peak_return_m`。

坐标轴使用球离脚后的平面飞行方向，而不是机器人触球瞬间已经旋转的骨盆方向。这样“向球门前进”和“向后撤”在不同站姿下仍有一致含义。

侧向回摆取轨迹中任一侧向峰值到最终位置的最大距离。它能覆盖“先向一侧卸力、再越过中线停到另一侧”的情形，不会因为最终点恰好是最大绝对值而误报为零。

自然度门禁 schema 升级为 v2。除原有成功、安全、严格重放、全身/腿/腰/手臂/tail jerk、路径和稳定时间约束外，新增最低要求：

- tail wobble 至少降低 5%；
- 后退回撤至少降低 10%；
- 侧向峰值回摆至少降低 15%。

## 4. 匹配 A/B 结果

第三球的 parent 是上一阶段已经晋级的 V6 自然跟随动作，不是更弱的早期基线。

| 指标 | V6 parent | V7 candidate | 改善 |
|---|---:|---:|---:|
| 球速 | 5.6011 m/s | 5.6011 m/s | 保持 |
| 目标误差 | 0.1352 m | 0.1352 m | 保持 |
| 骨盆总路径 | 1.8787 m | 1.6200 m | 13.8% |
| 最终平面位移 | 0.5220 m | 0.2645 m | 49.3% |
| 前进后反向回撤 | 0.5429 m | 0.4711 m | 13.2% |
| 侧向峰值回摆 | 0.8814 m | 0.6229 m | 29.3% |
| 支撑切换次数 | 49 | 37 | 24.5% |
| 稳定时间 | 4.30 s | 4.18 s | 2.8% |
| 全身 joint jerk RMS | 729.00 | 635.93 | 12.8% |
| 手臂 joint jerk RMS | 366.26 | 252.98 | 30.9% |
| 腰部 joint jerk RMS | 661.39 | 507.36 | 23.3% |
| tail joint jerk RMS | 1.0287 | 0.8411 | 18.2% |
| tail wobble index | 0.12166 | 0.10907 | 10.3% |

安全结果：

- `SUCCESS`；
- 无跌倒；
- 无关节限制越界；
- 无力矩限制越界；
- 无执行器饱和；
- 末端双脚支撑；
- 支撑脚滑移 0.0266 m；
- 三球及各自重放全部通过。

## 5. 被数据拒绝的方案

以下方案没有进入正式代码路径：

1. 直接冻结落脚后的策略目标：造成长时间单脚支撑、漂移或关节限制越界；
2. 按骨盆速度持续施加髋踝阻尼：能减少少量后退，但放大 tail jerk；
3. 只调整 recovery step length/yaw：最好仅减少约 8% 后退，且 wobble 有回归；
4. 单阶段提前切站立：可明显减小路径和后退，但把能量变成末段小抖动；
5. 髋、踝、腰联合俯仰回正：对符号和幅值敏感，部分组合触发关节限制。

这些搜索结果说明，动作自然度不能靠一个更强的阻尼或更大的站立权重解决。正确结构是“先卸动量，再独立收姿”。

## 6. 最终证据与复现

最终证据目录：

```text
/code/rosclaw/phase7_1_evidence/hat-trick-v7-body-effect-final/
```

最终视频：

```text
/code/rosclaw/phase7_1_evidence/hat-trick-v7-body-effect-final.mp4
```

内容哈希：

```text
report sha256: 9ff1d6ff0951214a9d4b10cd27c5bd06a4c596b17649f2747107304ca6357569
video  sha256: f66ada799e9d1507413c00a54ee219a950d22de6035f66523549f9f5bfbdc8cd
manifest sha256: a4b18f5aff522f2d59ee0a96f445a6c763b91c37031e5a1318e4ad95243aee93
```

复现命令：

```bash
.venv/bin/python -m rosclaw.entrypoint goalforge hat-trick run \
  --asset-root /code/rosclaw/phase4_references/RoboNaldo/RoboNaldo_Deploy \
  --output-dir /code/rosclaw/phase7_1_evidence/hat-trick-v7-body-effect-final \
  --source-checkout /code/rosclaw/rosclaw_test

.venv/bin/python -m rosclaw.entrypoint goalforge hat-trick export \
  /code/rosclaw/phase7_1_evidence/hat-trick-v7-body-effect-final/goalforge-hat-trick.json \
  --asset-root /code/rosclaw/phase4_references/RoboNaldo/RoboNaldo_Deploy \
  --output /code/rosclaw/phase7_1_evidence/hat-trick-v7-body-effect-final.mp4 \
  --source-checkout /code/rosclaw/rosclaw_test \
  --fps 50
```

## 7. 边界与下一步

本报告只证明 SIM 中、已校准的相同 Body/motion/regime 下改善成立，不声明真实 G1 已验证。当前控制仍是确定性的动作整形，不是完整的在线学习策略。

下一步应在保留本阶段门禁的前提下：

1. 把两阶段参数放入受约束的 continual-learning 候选空间；
2. 在摩擦、延迟、球位、扰动方向和扰动力 holdout 上做域随机化；
3. 学习接触后捕获步，而不是只使用固定策略帧；
4. 将后退、侧摆、wobble、jerk 和落点共同作为多目标 replay buffer；
5. 只有在严格 parent/candidate A/B、回滚和稳定性门禁同时通过后才晋级。
