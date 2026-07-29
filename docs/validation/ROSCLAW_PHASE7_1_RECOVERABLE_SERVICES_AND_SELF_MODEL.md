# ROSClaw Phase 7.1：可恢复持续学习服务与本体自我模型

日期：2026-07-27

证据域：`SIM`

硬件：本地 4×NVIDIA RTX A6000；本轮只使用空闲物理 GPU 2，未干扰 GPU 0/1/3 上的既有任务

## 1. 结论

本轮完成了 Phase 7.1 PR-3 至 PR-7 的第一版工程闭环：Rollout、Experience、Inference、Learner、Weight Update 五个可恢复服务已经落地；预测偏差触发器、解析物理加有界神经残差的 Forward Self Model、三时间尺度 Regime Encoder 和四分类 Agency Estimator 已实现。

真实 MuJoCo 服务闭环 11/11 检查通过，但上一轮 Candidate v3 仍然被拒绝，没有被激活：它虽然在 250 个相同场景上把平均目标误差改善了 3.43cm，95% CI 下界只有 1.37cm，低于预设 2cm 门槛，而且相对 Parent 新增 4 个关键安全回归。系统把这 4 个回归自动写入 Boundary 重跑队列，Adaptation 状态机转入 `ROLLBACK`，Inference 保持 Parent v2。

这意味着本轮证明的是“ROSClaw 已能安全地持续收集、学习、检查和拒绝”，不是“新策略已经让 G1 踢得更好”。

## 2. 五个可恢复服务

```text
MuJoCo Scenario
  → Rollout Service（场景分配、Body/Policy 绑定、动作内版本锁）
  → Experience Service（不可变日志、SQLite 目录、四分区缓存）
  → Learner Service（只训练，不控制机器人；完整优化器检查点）
  → Weight Update Service（publish → verify → stage → activate/freeze/rollback）
  → Inference Service（active/candidate/rollback slot；COMPLETE/ABORT 前不换版本）
```

### Rollout Service

- 一个高动态动作只允许一个 `PolicyVersion`，输出 `VersionedTrajectory`；
- 服务崩溃后，RUNNING 任务自动转为 ABORTED；
- 不重放崩溃前的旧动作；
- 场景 commitment、Body、controller 和 policy lineage 全部校验。

### Experience Service

- 文件级 append-only hash-chain 是事实源；
- SQLite 只作为可重建目录，启动时逐字段核对 trajectory、partition、policy version、event sequence 和 Boundary 完成状态；
- 内存层仍是有界 Recent/Anchor/Boundary/Self replay；
- 原始轨迹和可变服务状态强制放在源码 checkout 外。

### Inference 与 Weight Update Service

- `active`、`candidate`、`rollback` 三个 slot；
- artifact 采用内容寻址并在原子 link 后 fsync 目录；
- motion lease 存在时激活请求只会冻结，不能中途换权重；
- Weight Update 的 REQUESTED 与 Inference 事件做恢复对账：回调已完成时补写 recovered completion，未发生 mutation 时 abort，出现歧义时冻结 Inference。

### Learner Service

- 只消费版本化 batch，没有 Registry、DDS 或硬件权限；
- actor、三个 critic/target、三个 optimizer、temperature、constraint multipliers、update index、CPU/CUDA RNG 全量检查点；
- 已完成 batch 幂等返回；不确定的 optimizer update 在恢复时 quarantine，绝不盲目重放。

真实闭环第一次运行正好触发了这个门禁：`ResidualSACUpdate.schema_version` 被误放进只允许数值的 metrics，Learner 将任务隔离而不是继续发布。修复适配器并增加回归测试后，第二次运行通过。这是恢复机制在真实工作流中发现接口错误的直接证据。

## 3. 什么时候才允许学习

`prediction_monitor.py` 实现固定状态机：

```text
NORMAL → SUSPECTED_SHIFT → CONFIRMED_SHIFT → SHADOW_LEARNING
       → CANDIDATE_READY → CONSOLIDATED / ROLLBACK
```

身体、触球结果、接触模式、控制延迟、能量和任务性能六类预测残差必须持续超过阈值，单次噪声不会开启学习。只有 `SHADOW_LEARNING` 状态允许更新神经残差或 learner；Candidate 还必须同时满足样本量、新任务改善、Anchor 保持、收敛和零关键安全回归。

本轮 4 个真实匹配安全回归使 shift 得到确认并打开 shadow learning，但 Candidate gate 最终把状态送入 `ROLLBACK`。

## 4. “本体自我模型”通俗解释

这里的 Self 不是主观意识，也不声称机器人有感受。它是一个可测量、可证伪的控制系统：

- Forward Self Model 回答“我发出这个动作后，下一刻关节、骨盆、质心、脚接触、球、能量和跌倒风险应该怎样变化”；
- Regime Encoder 回答“现在是不是换了地面、球的摩擦、延迟、电机增益、负载或外部扰动”；
- Agency Estimator 比较预测和实际后果，回答“这是我自己的动作造成、外力造成、传感器坏了，还是证据不足”。

Forward Model 采用解析动力学作为稳定底座，神经网络只学习幅度受限的 residual，且输出层从零开始。真实 G1 轨迹的 96 个 transition 做 within-trace shadow calibration 后，均方预测误差从 `0.00299398` 降到 `0.00291725`，降低约 2.56%。这个数字只能说明同一轨迹上的有界校准有效，不能当作未见工况泛化成绩。

80N Boundary 场景被 Agency 判为 `EXTERNAL_DISTURBANCE`。这只是一次功能闭环，尚未达到文档要求的多类测试集 `Agency classification ≥ 90%`。

## 5. 真实 MuJoCo 闭环证据

执行命令：

```bash
CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m rosclaw.entrypoint \
  continual services validate \
  --asset-root /code/rosclaw/phase4_references/RoboNaldo/RoboNaldo_Deploy \
  --candidate /code/rosclaw/phase7_1_evidence/training-seeds/seed-17103-candidate.bin \
  --matched-report /code/rosclaw/phase7_1_evidence/seed-17103-matched-250-v2.json \
  --state-root /code/rosclaw/phase7_1_evidence/service-validation-v2-state \
  --output /code/rosclaw/phase7_1_evidence/service-validation-v2.json \
  --source-checkout /code/rosclaw/rosclaw_test \
  --learner-device cuda:0 \
  --learner-updates 3
```

结果：

| 检查 | 结果 |
| --- | --- |
| 四个 replay partition 都执行 MuJoCo physics | PASS |
| 四个轨迹都完成相同场景严格重放 | PASS |
| motion 内 policy version switch | 0 |
| Experience SQLite 从 append-only truth 恢复 | PASS |
| 4 个 matched safety regression 进入 Boundary 队列 | PASS |
| CUDA Learner 完整 checkpoint 恢复 | PASS |
| Forward calibration error 下降 | PASS |
| Rollout crash 注入后旧动作重放 | 0 |
| Inference crash 注入后旧 motion 恢复 | 0 |
| 被拒 Candidate 激活 | 0 |
| Adaptation 最终状态 | ROLLBACK |

主报告：`/code/rosclaw/phase7_1_evidence/service-validation-v2.json`

- 文件 SHA-256：`7dcef1746bd9a77852fbf1f928a78f1c147b8420d76d44f674f55cb7d8c99b64`
- 内容内 report hash：`sha256:31bd58b930629b19d72662130b14e2b716c1eaf9d54dc2063f1358289dff0f3a`
- 输入 matched report hash：`sha256:bb13ca3b7dd682c248fd284607f611524d7b750acf9116b81a249393e54b969a`
- Registry mutation：0；DDS opened：0；hardware authorized：false。

四条原始轨迹：

| Partition | NPZ SHA-256 |
| --- | --- |
| Anchor | `ad51435156c874c2081efcd253f589dc350f8e68da7f4504b555658ffe79c69d` |
| Boundary | `37aee1a81315674b03ab0dd65551bc5b7ca7aedcb6551fcc74ad15cc25eb612b` |
| Recent | `0b7cf8c4b9609e8fb64beef644c92659a31e70bfdd4faa9d84a43362035a69e3` |
| Self | `cd7749f578f473dff59ab4c9d690b4ea6a61cd20f225e1887d81897cb8fc432b` |

## 6. 软件回归

- 本轮 continual/self/service 定向 Ruff、format 与 mypy：PASS；
- 故障注入、服务恢复、SAC checkpoint、自我模型和相关 SimForge 定向测试：PASS；
- 全仓测试首次运行：`5320 passed, 67 skipped, 27 deselected, 4 failed`；4 个失败均因 pytest 隔离 HOME 未继承已经安装的 LeRobot runtime 路径；
- 显式绑定现有 LeRobot 0.6.1 解释器后，原 4 个失败用例：`4 passed`；
- 带同一 runtime 配置重新运行全仓：`5332 passed, 59 skipped, 27 deselected`，26 个既有 warning，耗时 1034.83 秒。

仓库全局 Ruff 仍有 244 个既有问题，集中在本轮范围外的 `examples/rh56_rps`。本轮没有借机修改该并行/用户工作区；本轮变更文件的 Ruff 与 format 均通过。

## 7. 新增代码

- `src/rosclaw/continual/services/`：五个 durable service 与 persistence primitive；
- `src/rosclaw/continual/boundary_feedback.py`：matched regression → Boundary re-rollout request；
- `src/rosclaw/continual/service_validation.py` 与 `service_cli.py`：真实 MuJoCo 服务闭环；
- `src/rosclaw/self_model/prediction_monitor.py`：适应触发和 stop-learning gate；
- `src/rosclaw/self_model/forward_model.py`：解析预测加有界神经 residual；
- `src/rosclaw/self_model/regime.py`：Fast/Episode/Persistent latent 与 regime expert reuse；
- `src/rosclaw/self_model/agency.py`：SELF/EXTERNAL/SENSOR/UNKNOWN 四分类；
- `src/rosclaw/continual/learner.py`：完整 CPU/CUDA 可恢复检查点；
- 故障注入、持久化篡改、版本锁、恢复对账、自我模型测试。

## 8. 尚未完成

- 当前 Candidate v3 仍不满足安全与显著性门槛，不能宣传成已自进化成功；
- Forward Model 需要跨 seed、跨摩擦、跨负载的 holdout benchmark；
- Agency 需要平衡四类标注数据和准确率/ECE 测试；
- Regime Expert 目前证明了重复工况复用接口，尚未证明跨 episode 快速恢复收益；
- Plastic unit rejuvenation、Fresh Network Gap、dead-unit/effective-rank 长期曲线和动态 expert allocation 尚未实现；
- 下一轮应重跑 4 个 Boundary 场景并训练新 Candidate，再用同一 250 场景冻结 suite 做 matched evaluation；
- 足球侧仍需移动球速度/方向网格和随机扰动时刻的成功率实验，旗舰 Hat Trick 视频不能替代统计验证。
