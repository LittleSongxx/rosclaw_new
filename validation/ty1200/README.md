# ROSClaw × TY1200 全栈验证工作区

本目录实现《ROSClaw on TY1200 全栈测试与验证实施大纲》v1.0（`rosclaw闭环.md`）。

**验证原则：** 仿真先行、证据优先、默认失效关闭、禁止虚假通过。
**默认执行模式：** FIXTURE / DRY_RUN / REPLAY / SIMULATION / SHADOW。REAL 模式默认禁止。

## 使用方式

```bash
# 0. 端口与环境（所有脚本统一从此读取，禁止硬编码端口）
cp configs/ports.env.example configs/ports.env   # 站点本地化一次
set -a; source configs/ports.env; set +a

# 1. 环境冻结
bash scripts/collect_system_baseline.sh reports/<run_id>

# 2. 全量验证（阶段可选）
bash scripts/validate_all.sh            # 全阶段
bash scripts/validate_phase.sh stage1   # 单阶段

# 3. 报告
python3 scripts/generate_report.py reports/<run_id>
```

## 目录

| 路径 | 内容 |
|---|---|
| `configs/` | 端口规划、provider/runtime/seekdb/wiki 配置模板 |
| `fixtures/` | missions、practice、wiki 语料、问答题库、脱敏样本、故障定义 |
| `scripts/` | validate_all / validate_phase / 基线采集 / 故障注入 / 恢复 / 报告生成 |
| `benchmarks/` | provider / seekdb / memory / wiki / trace / eventbus 基准 |
| `tests/` | 平台、runtime+eventbus、rosclawd 边界、provider/trace/practice/seekdb/memory/wiki/how-auto-darwin、agent 黑盒 |
| `reports/<run_id>/` | environment.json、commands.log、module_matrix.json、metrics、traces、validation_summary.json、VALIDATION_REPORT.md、SHA-256 清单 |

## 输出纪律

每次运行必须同时产出：人类可读 Markdown、机器可读 JSON、原始命令日志、
SHA-256 manifest、源码快照标识、环境版本，以及 PASS / WARN / FAIL /
BLOCKED_EXTERNAL 状态。站点地址、token、模型权重路径一律不进入公共仓库
（DeepSeekV4 站点配置放 `~/.rosclaw/providers/`）。

## 验证等级

V0 静态存在 → V1 组件通过 → V2 本机运行 → V3 跨模块闭环 → V4 故障恢复 →
V5 性能验证 → V6 Agent 黑盒 → V7 长稳 → V8 真机（本轮不强制）。
