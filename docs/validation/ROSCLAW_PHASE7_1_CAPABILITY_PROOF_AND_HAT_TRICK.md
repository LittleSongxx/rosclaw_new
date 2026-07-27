# ROSClaw Phase 7.1：Capability Proof 与 GoalForge Hat Trick 实施报告

日期：2026-07-27

分支：`phase7-1-capability-proof-demo`

基线：Phase 7 提交 `556a85f96099b692bf84a0713ca4fee112b36cb0`

证据域：MuJoCo `SIM` 与四卡 `CUDA_SCREENING`；没有真实机器人执行、DDS Hardware、Registry 写入或候选激活

## 1. 结论先行

本轮完成了 Phase 7.1 文档中最优先的 Candidate Motion Effect Proof，并做成了可公开展示的三球仿真 Demo，但没有把“出现改善均值”误报成候选已经晋级。

核心结果如下：

- 实现只读 Candidate Loader、NumPy actor inference、episode 版本锁、推理 receipt 和产品 CLI；真实 seed 17103 的 v3 artifact、parent、Body、动作/观察合同和 tensor 全部通过校验；
- 在 4 张 A6000 上完成 8 个 learner seed、每个 10 次 residual SAC 更新，8 个候选均保持有限值和有界动作；
- 对筛选出的 seed 17103 完成 250 个独立 MuJoCo 场景、四个 matched arm、共 1000 条 episode 结果；
- v3 相对 Parent 的成功率提高 1.6 个百分点，失败惩罚目标误差平均改善 3.43cm，但 95% CI 为 `[1.37cm, 6.10cm]`，下界没有超过预设 2cm 最小改善，而且出现 4 个新的关键安全回归；最终门禁为 `REJECTED`，v3 没有 stage、activate 或获得硬件授权；
- 新增不确定性感知 `ContactTimingBelief`、延迟发球物理和有界 `MovingBallInterceptAdapter`；低置信度下 kick phase residual 自动归零；
- `G1 GoalForge Hat Trick` 三球全部在 MuJoCo 真实 stepping 中成功并严格重放：随机九宫格目标重炮、移动球一脚迎射、80N 扰动下 Feedback OFF 失稳而 ON 救回；
- 生成 1280×720、30fps、18.2s 的证据下游宣传视频。视频绑定输入 report/trajectory 哈希，明确标记 `SIM EVIDENCE REPLAY · VISUALIZATION ONLY`，不产生或修改任务证据。

因此本轮的准确结论是：ROSClaw 已经从“候选能训练”推进到“候选能真实执行、能被 matched suite 证伪、不能自行越过安全门”；G1 的展示能力通过高层拦截适配器和反馈平衡得到可见扩展。当前 residual learner 出现了正向学习信号，但还不是可晋级的运动冠军。

## 2. 闭环架构

```text
4× A6000 / 8 learner seeds
          │ immutable candidate artifact
          ▼
read-only loader ── hash / parent / Body / contract / tensor validation
          │ NumPy deterministic actor + episode version lock
          ▼
Matched MuJoCo suite
  Frozen Prior / Parent v2 / Candidate v3 / Fresh Network
          │
          ├── Recent 50
          ├── Anchor 50
          ├── Boundary 100
          └── Self 50
          │  bootstrap CI + safety/retention/replay gates
          ▼
      REJECTED v3 ── active v2 unchanged

Verified MuJoCo trajectories
          │ one-way visualization input
          ▼
GoalForge Hat Trick MP4 (never feeds labels back into evaluation)
```

## 3. ROSClaw 优化与开发内容

### 3.1 Candidate 从 artifact 到动作的只读路径

新增 `src/rosclaw/continual/inference/`：

- `loader.py` 解析现有 residual SAC v1 二进制格式，限制 artifact 大小，只接受 exact float32 actor tensor 集合，并验证 SHA-256、直接父子谱系、Body、controller/safety identity、观察/动作合同、shape 和有限值；
- `policy_runtime.py` 使用 NumPy 执行确定性 actor，不依赖训练时 PyTorch/CUDA；四个 residual 被显式映射到 waist roll、右髋 roll/yaw 和 kick phase rate；
- `version_lock.py` 在一次运动段内锁定 version、version hash、artifact 和 Body；
- `receipt.py` 记录推理次数、动作边界、归一化动作 RMS、最大边界比例、contact timing 统计、零版本切换、零 Registry 写和未打开 DDS；
- `src/rosclaw/continual/cli.py` 提供 `inspect`、`evaluate` 和可恢复分片 `merge`；CLI 不读写 active slot。

实际检查的 seed 17103：

| 项目 | 值 |
|---|---|
| Candidate | v3 `sha256:a0b198a55cda50bb67ac777decf05bd3c23a358f62a09d17226efc4ee773c8e1` |
| Artifact | `sha256:02160b6e7aac6ca1f3f47b790ea0ecaaa6637ea1111bd34fe66f21cfd5328ba7` |
| Parent v2 | `sha256:1009007f5b8c37d7c4a8edece96d09504e7806a7d8e8ad3e77f2282a3b06172e` |
| Body | `sha256:1525aafc953293dd340475e8660b58fae7ea1a22715222ab06331b644340626c` |
| Actor | 8 observations、`64×64` hidden、4 bounded actions |
| 状态 | read-only；Registry 0；DDS false；activated false；hardware false |

### 3.2 Matched Candidate Evaluation

新增 `g1_candidate_evaluation.py`，每个场景固定相同 seed、初态、物理参数、扰动和 ShotParameters，只改变四个 arm 的控制策略：

1. Frozen RoboNaldo Prior；
2. Active Parent v2（显式 zero residual）；
3. Candidate v3；
4. 固定 seed 的 Fresh Random Network。

评测记录成功、触球、过门、失败惩罚目标误差、条件目标误差、球速、跌倒、关节/力矩/饱和、COM 裕度、支撑脚滑移、tracking RMS、能量代理、动作漂移、轨迹哈希、推理 receipt 和版本切换。Frozen Prior 与 Parent 的完整运动轨迹必须逐场景一致；每个分区至少对一个 Candidate episode 做严格 replay。

评测支持 disjoint suite shard、逐场景原子 checkpoint 和崩溃续跑。MuJoCo 对“没有可测单支撑窗口”的 COM accumulator 使用 `+inf`；现在只对这一已知哨兵保守编码为 `-1.0m`，所有其他非有限 evidence 仍然硬拒绝，JSON 使用 `allow_nan=False`。

### 3.3 Contact timing 与移动球

新增 `ContactTimingBelief`，融合：

- 球相对右脚的三维位置和速度；
- 当前 policy phase 与静态先验；
- 控制时延、sensor quality；
- 最近预测 phase 的 churn；
- 已观测触球状态。

只有球确实移动、预测时域有效、拦截误差在界内且 confidence 达标时，Candidate 的 `kick_phase_rate` 才可进入物理控制；静止、已触球、低质量或高不确定状态全部 fail closed 为零。

GoalForge 场景新增 `ball_launch_delay_sec`。Backend 在发球时刻之前保持球静止，到时再施加真实 MuJoCo 速度，解决“球从 t=0 就滚过机器人、踢腿尚未开始”的时序错误。`MovingBallInterceptAdapter` 只接受不超过 0.20m/s、预测接触误差不超过 0.16m 的温和来球，并且只输出已有高层 ShotParameters，不写低层关节。

### 3.4 Hat Trick 与宣传视频

新增：

- `g1_hat_trick.py`：三球执行、严格 replay、trajectory 哈希和结果报告；
- `g1_hat_trick_cli.py`：`goalforge hat-trick run/export`；
- `g1_hat_trick_video.py`：MuJoCo evidence replay、慢动作触球、球迹、九宫格目标、进球视角和 Feedback OFF/ON 分屏；
- 所有 raw trajectory、report、视频和预览图均位于 checkout 外。

视频导出前重新验证 report 已通过、Body 相同、主轨迹和 comparison 轨迹哈希相同；宣传层不会回写评测、经验池或候选状态。

## 4. 实际验证结果

### 4.1 8 个训练 seed / 4 张 A6000

每张物理 GPU 运行两个不同 seed；每个候选完成 10 次更新，CUDA 峰值分配 21,251,584 bytes，hidden activation、loss 和动作均为有限/有界。8 个 candidate hash 全部不同。

四场景初筛结果：

| Seed | GPU | 成功率差 | 目标误差改善 | 新关键回归 | 决定 |
|---:|---:|---:|---:|---:|---|
| 17101 | 0 | 0.00 | +0.70cm | 0 | REJECTED |
| 17102 | 1 | -25.00pp | -44.42cm | 1 | REJECTED |
| 17103 | 2 | 0.00 | +1.46cm | 0 | REJECTED；进入 250 场景复核 |
| 17104 | 3 | 0.00 | -0.66cm | 0 | REJECTED |
| 17105 | 0 | 0.00 | -0.87cm | 0 | REJECTED |
| 17106 | 1 | 0.00 | +0.42cm | 0 | REJECTED |
| 17107 | 2 | 0.00 | -0.62cm | 0 | REJECTED |
| 17108 | 3 | 0.00 | +0.60cm | 0 | REJECTED |

初筛只用于选择更值得花费 250 场景预算的 seed，不作为晋级证据。

### 4.2 250 场景 matched 结果

规模：Recent 50、Anchor 50、Boundary 100、Self 50；每个场景四个 arm，共 1000 个 episode row；四分区 Candidate strict replay 全部通过。

| 指标 | Frozen Prior | Parent v2 | Candidate v3 | Fresh Network |
|---|---:|---:|---:|---:|
| 成功率 | 20.4% | 20.4% | 22.0% | 18.8% |
| 触球率 | 59.2% | 59.2% | 59.6% | 61.2% |
| 过门率 | 55.6% | 55.6% | 57.6% | 57.6% |
| 失败惩罚目标误差 | 1.6385m | 1.6385m | 1.6043m | 1.6646m |
| Fall rate | 67.2% | 67.2% | 64.0% | 68.0% |
| Joint violation | 69.6% | 69.6% | 67.6% | 68.8% |
| Torque violation | 0% | 0% | 0% | 0% |
| 支撑脚滑移 | 0.0409m | 0.0409m | 0.0388m | 0.0381m |
| Tracking RMS | 0.5795rad | 0.5795rad | 0.5650rad | 0.5757rad |
| Energy proxy | 41029.0 | 41029.0 | 41043.9 | 41741.3 |

Paired Candidate - Parent：

- 成功率差 `+1.6pp`，95% CI `[+0.4pp, +3.2pp]`；
- 失败惩罚目标误差改善 `+3.43cm`，95% CI `[+1.37cm, +6.10cm]`；
- Anchor 成功率差 `0.0pp`，满足历史平均下降小于 3%；
- 运动段版本切换为 0；四分区 replay 全通过；
- 出现 4 个新关键回归：3 个 Boundary、1 个 Self，均从 Parent 的非 joint-limit 失败变为 v3 `JOINT_LIMIT_EXCEEDED`，其中 2 个同时被判为 fall。

最终 gate：

| 检查 | 结果 |
|---|---|
| 250 场景 | PASS |
| 8 training seeds | PASS |
| 四分区 strict replay | PASS |
| motion version switch = 0 | PASS |
| Anchor degradation < 3% | PASS |
| 关键技能成功率下降 ≤ 5% | PASS（实际 +1.6pp） |
| 95% CI 下界 > 2cm | **FAIL**（1.37cm） |
| Critical regression = 0 | **FAIL**（4） |
| 决定 | **REJECTED** |

正式证据：`/code/rosclaw/phase7_1_evidence/seed-17103-matched-250-v2.json`

SHA-256：`bb13ca3b7dd682c248fd284607f611524d7b750acf9116b81a249393e54b969a`

这个结果说明 v3 确实改变了物理运动并出现整体改善信号，但仍不满足稳定性—可塑性门。整体 fall rate 下降不能抵消四个 previously-safe 场景的新关键回归；ROSClaw 必须先把这些反例加入 Boundary replay 再训练，而不是平均数好看就晋级。

### 4.3 G1 GoalForge Hat Trick

| 球 | 场景与规划 | 结果 | 球速 | 目标误差 | 稳定性 |
|---|---|---|---:|---:|---|
| 1：九宫格重炮 | seed 202607289 从 3×3 选择右下区 `y=-0.75,z=0.20` | SUCCESS、目标区命中 | 6.98m/s | 0.326m | 不倒；无 joint/torque violation；滑移 0.0021m |
| 2：移动球迎射 | t=4.0s 发球，初始 x=1.12m，vx=-0.08m/s；预测接触 x=1.0176m | SUCCESS、5.28s 触球并进球 | 6.22m/s | 0.349m | 不倒；无 joint/torque violation；滑移 0.0106m |
| 3：80N 救援 | 同一 seed/physics，Feedback OFF vs ON | OFF：`JOINT_LIMIT_EXCEEDED` 且 fall；ON：SUCCESS | 4.93m/s | 0.096m | ON 不倒且无 joint/torque violation |

三球每次执行 6520 个 MuJoCo physics step，主轨迹全部严格重放。第三球 receipt 的 `tracking_improved=false` 被原样保留：runtime 自身的整段 RMS 指标受早期瞬态影响，但 matched outcome 明确显示 OFF 发生 joint-limit/fall、ON 保持安全并成功。这只支持该固定 80N SIM 场景的局部救援结论。

Hat Trick 证据：`/code/rosclaw/phase7_1_evidence/hat-trick-v3/goalforge-hat-trick.json`

SHA-256：`98a9899a1a73d7cfa3ac435c049099ee124f312d1d7bc98718bc1453b3ffe99c`

### 4.4 宣传视频

视频：`/code/rosclaw/phase7_1_evidence/hat-trick-v3/rosclaw-g1-goalforge-hat-trick.mp4`

参数：H.264、yuv420p、1280×720、30fps、546 帧、18.2s、1,589,023 bytes。

视频 SHA-256：`8c8ed367ff87ef415894407ac658a482ec853ea399948b733022de610beb5997`

Manifest SHA-256：`21e5a61e564bd79e18a5424b6fd54caebd14e27f89e1e94bead757dcb829c018`

### 4.5 软件验证

- 新增 inference、contact timing、candidate evaluation 和 moving-ball 定向测试：12 passed；
- Ruff：通过；
- Mypy：13 个相关 source file 无问题；
- compileall：通过；
- 最终完整仓库回归：5306 passed、59 skipped、27 deselected、26 warnings，耗时 17m11s，无失败；
- 26 个 warning 均来自既有 `pyseekdb.Configuration` 与 `tar.extractall` 弃用提示。

## 5. 复现命令

Candidate 检查：

```bash
.venv/bin/python -m rosclaw.entrypoint continual candidate inspect \
  /code/rosclaw/phase7_1_evidence/training-seeds/seed-17103-candidate.bin \
  --asset-root /code/rosclaw/phase4_references/RoboNaldo/RoboNaldo_Deploy
```

单个 matched shard：

```bash
.venv/bin/python -m rosclaw.entrypoint continual candidate evaluate \
  /code/rosclaw/phase7_1_evidence/training-seeds/seed-17103-candidate.bin \
  --asset-root /code/rosclaw/phase4_references/RoboNaldo/RoboNaldo_Deploy \
  --recent 50 --anchor 0 --boundary 0 --self-count 0 \
  --training-seed-count 8 --suite-shard recent-final \
  --output /code/rosclaw/phase7_1_evidence/seed-17103-recent-50.json
```

Hat Trick 证据与视频：

```bash
.venv/bin/python -m rosclaw.entrypoint goalforge hat-trick run \
  --asset-root /code/rosclaw/phase4_references/RoboNaldo/RoboNaldo_Deploy \
  --output-dir /code/rosclaw/phase7_1_evidence/hat-trick-v3

.venv/bin/python -m rosclaw.entrypoint goalforge hat-trick export \
  /code/rosclaw/phase7_1_evidence/hat-trick-v3/goalforge-hat-trick.json \
  --asset-root /code/rosclaw/phase4_references/RoboNaldo/RoboNaldo_Deploy \
  --output /code/rosclaw/phase7_1_evidence/hat-trick-v3/rosclaw-g1-goalforge-hat-trick.mp4
```

## 6. 与 RoboNaldo prior 的关系

本轮没有声称重写或全面超过 RoboNaldo。RoboNaldo 仍提供成熟的全身踢球 prior；ROSClaw 增加的是 prior 外层的可校验能力：

- 对 learned residual 做可执行的 hash/lineage/Body/contract 验证；
- 用 Parent、Candidate、Fresh Control 的 matched 物理实验判断“是否真的变好”；
- 即便 Candidate 的总体均值更好，只要有新关键回归仍拒绝晋级；
- 在冻结 prior 的前提下增加移动球发射/拦截高层适配和扰动反馈救援；
- 把物理证据与宣传可视化单向隔离。

所以当前优势不是“所有射门指标都已超过 RoboNaldo”，而是 ROSClaw 能让 prior 周围的持续学习和新技能扩展变得可证伪、可重放、可拒绝、可展示。

## 7. 没有完成或不能声称的内容

- 250 场景 candidate 没有晋级，active parent 仍是 v2；不能声称 online RL 已经稳定提高 G1；
- 九宫格本轮是 seed-committed 随机选择并命中一个区域，不是 9 个区域的统计命中率；
- 移动球成功只覆盖 `vx=-0.08m/s` 的有界来球；当前宣传 shot 由 deterministic intercept adapter 驱动，不是低置信度 Candidate phase residual；
- 第三球是固定 80N 场景，不是随机时刻、方向和幅值的扰动成功率；
- 没有球旋转/空气动力插件，因此没有 Magnus 弧线球主张；
- 没有真实机器人、sim-to-real 或硬件安全授权；
- Phase 7.1 文档后续的五个可恢复服务、Adaptation Trigger、Forward Self Model、Regime Encoder、Agency estimator、plastic-unit rejuvenation、residual experts 和 Motion Forge 仍是下一批实施，不属于本轮已完成结论；
- 这里的 Self 仍是可测试的具身状态/证据合同，不是主观意识证明。

## 8. 下一步建议

下一轮不应该放宽门槛，而应把本轮 4 个关键回归自动加入 Boundary replay，扩大真实 rollout 与 learner update 数量，再用同一 250 场景 suite 做 paired freeze。足球侧应先扩展 moving-ball 的速度/方向网格和 80N 扰动时刻/方向随机化，形成成功率而不是单个旗舰 shot；随后再接 Adaptation Trigger、Forward Self Model 和 Regime Encoder，使“什么时候学习、当前身体处于什么工况、学习是否伤害旧技能”进入同一闭环。
