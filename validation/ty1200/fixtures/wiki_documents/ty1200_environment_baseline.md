# TY1200 环境与基线（本次实施冻结）

## 硬件

- 设备：天数智芯 TY1200 具身智能盒子
- CPU：Intel Core Ultra 7 255H（14 核）
- GPGPU：Iluvatar MRC-V100，16GB HBM2e（ixsmi 查询：MRC-V100-0x18, 16384 MiB, 15824 MiB, 46 C, 0 %）
- 内存：30GB；Swap：2GB

## 内核与实时性

- 内核：Linux x001 6.8.0-85-generic #85~22.04.1-Ubuntu SMP PREEMPT_DYNAMIC Fri Sep 19 16:18:59 UTC 2 x86_64 x86_64 x86_64 GNU/Linux
- 调度：PREEMPT_DYNAMIC（不是 PREEMPT_RT 硬实时内核）

## 时间同步

- NTP：System clock synchronized: yes；NTP service: active
- 时区：Asia/Shanghai

## 模型服务

- Qwen3-Embedding-0.6B：127.0.0.1:8000（OpenAI 兼容 /v1/embeddings，1024 维）
- Cosmos-Reason2-2B：127.0.0.1:8001（OpenAI 兼容 /v1/chat/completions）
- 恢复方式：厂商容器 PID1 为 bash，docker kill 后需在容器内手动重启 vllm 服务

## 软件版本

- rosclaw：rosclaw 1.0.1（rosclaw source: codeload tarball of ros-claw/rosclaw@main (2026-07-31), NOT a git checkout）
- Python：Python 3.12.13
- ty1200-platform：Version: 0.1.0
- modeld：active
