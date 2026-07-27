# ROSClaw ROS1 CMU ARE 应用线

这条路线把 CMU Autonomous Exploration Development Environment 和 ARiADNE2-ROS-Planner 复制到 ROSClaw 内部的 `third_party/ros1/are/src/`，默认用 Docker Desktop 管理的 ROS Noetic 容器运行。

## 本地配置

复制示例配置并填写本地 API key：

```bash
cp .env.example .env
```

`.env` 已被 `.gitignore` 忽略，不应提交；提交时只保留 `.env.example`。常用参数包括：

- `DEEPSEEK_API_KEY`
- `CMU_ARE_WORLD`
- `USE_RVIZ`
- `ARIADNE2_USE_RVIZ`
- `ARIADNE2_LAYERED_PROJECTION`
- `CMU_SPEED`
- `CMU_NAV_TIMEOUT`
- `CMU_NAV_TOLERANCE`
- `CMU_MAX_RELATIVE_M`
- `CMU_CHAT_PROGRESS_INTERVAL`
- `CMU_MAX_SEQUENCE_STEPS`
- `CMU_CIRCLE_SEGMENTS`
- `CMU_MAX_CIRCLE_RADIUS`
- `CMU_EXPLORATION_ON_MANUAL`
- `CMU_DASHBOARD_HOST`
- `CMU_DASHBOARD_PORT`
- `CMU_DASHBOARD_MAX_POINTS`
- `DETACHED`

## 大型 mesh 资源

`third_party/ros1/are/src/vehicle_simulator/mesh/` 包含 campus 等 Gazebo 大型网格资源，其中单个 `campus.dae` 超过 GitHub 普通 Git 的 100MB 文件限制。该目录默认保留在本机用于 Docker 构建和仿真运行，但不会提交到远程仓库。

如果在全新 clone 上运行 CMU/Gazebo 仿真，需要从本机原始 ARE 资源或已有工作区重新复制该目录到：

```text
third_party/ros1/are/src/vehicle_simulator/mesh/
```

ROSClaw Python 控制、任务中控台、CLI、ARiADNE2/CMU 源码和测试代码不依赖该目录入仓；真实 Gazebo campus 场景渲染依赖本地 mesh 资源存在。

## 启动常驻仿真

```bash
docker compose -f docker/ros1/docker-compose.ros1-are.yml up -d --build rosclaw seekdb
```

默认配置：

- `CMU_ARE_WORLD=campus`
- `HEADLESS=true`
- `START_ARIADNE2=true`
- `ARIADNE2_ACTIVE=false`
- `ARIADNE2_USE_RVIZ=false`
- `ARIADNE2_LAYERED_PROJECTION=false`

也就是说，Gazebo/ARE 和 ARiADNE2 进程都会启动，但 ARiADNE2 默认不发布探索 waypoint，等待 ROSClaw 指令；默认只打开 CMU `vehicle_simulator` 的 RViz，不打开 ARiADNE2 自带 RViz，以降低 CPU 负载。

## 两终端 RViz 演示

推荐演示方式是：一个终端长期运行仿真和 RViz，另一个终端只负责下达 ROSClaw 指令。

终端 A：启动仿真 + RViz，并保持前台常驻：

```bash
./scripts/start_cmu_rviz.sh
```

这里脚本默认设置 `HEADLESS=true`、`USE_RVIZ=true`、`ARIADNE2_USE_RVIZ=false`：Gazebo 不弹出独立 GUI 窗口，但仿真仍在运行；只打开一个 CMU RViz。若同时需要 Gazebo GUI：

```bash
HEADLESS=false ./scripts/start_cmu_rviz.sh
```

如果不想在终端 A 里持续看到 ROS/Gazebo 日志，可以后台启动：

```bash
DETACHED=true ./scripts/start_cmu_rviz.sh
```

终端 B：连接同一个常驻容器下达移动或探索指令：

```bash
docker compose -f docker/ros1/docker-compose.ros1-are.yml exec rosclaw rosclaw app cmu-check
docker compose -f docker/ros1/docker-compose.ros1-are.yml exec rosclaw rosclaw app cmu-go "向上走 3 米"
docker compose -f docker/ros1/docker-compose.ros1-are.yml exec rosclaw rosclaw app cmu-go "去 inspection_a"
docker compose -f docker/ros1/docker-compose.ros1-are.yml exec rosclaw rosclaw app cmu-explore pause
docker compose -f docker/ros1/docker-compose.ros1-are.yml exec rosclaw rosclaw app cmu-explore resume
```

不要在终端 B 再运行 `cmu-launch` 或新的 `docker compose -f docker/ros1/docker-compose.ros1-are.yml up`；控制命令应该通过 `docker compose -f docker/ros1/docker-compose.ros1-are.yml exec rosclaw ...` 进入终端 A 已经启动的同一个 ROS master。

ROS1 EOL 提示窗口已通过 `DISABLE_ROS1_EOL_WARNINGS=1` 在 compose、镜像和启动脚本中默认关闭。

停止终端 A 里的仿真和 RViz：

```bash
./scripts/stop_cmu_rviz.sh
```

默认 `SPAWN_CAMERA=false`，因此不会启动 Gazebo camera 模型，也不会发布 `/camera/image`。默认 RViz 布局也已移除 Image display，避免启动时弹出 `Image / No Image` 窗口。CMU ARE 的 LiDAR、车辆、local planner 和 ARiADNE2 主流程不依赖这个 camera spawn。若后续需要测试相机，可在本地 `.env` 中改为 `SPAWN_CAMERA=true`，并在 RViz 中手动添加 Image display。

默认 `ARIADNE2_LAYERED_PROJECTION=false`，也就是使用 ARiADNE2 原来的全高度 2D 投影。上一版全局当前高度层投影会破坏远处 2D 连通性，导致 waypoint 飘到看似可达但实际错误的区域，因此不再默认开启。

如果要实验性修复隧道入口的上下层混叠，可以在 `.env` 中临时设置 `ARIADNE2_LAYERED_PROJECTION=true`。当前实现不会再全局替换地图，而是先生成旧版全高度投影，再只在机器人附近 `ARIADNE2_PROJECTION_OVERLAY_RADIUS` 米内用当前高度层的已知格子做局部覆盖。这样远处全局连通性仍保留旧行为；若局部高度层有效数据不足，也会直接保留旧版全高度投影。

## 检查环境

```bash
docker compose -f docker/ros1/docker-compose.ros1-are.yml exec rosclaw rosclaw app cmu-check
docker compose -f docker/ros1/docker-compose.ros1-are.yml exec rosclaw rostopic list
```

## 目标点导航

```bash
docker compose -f docker/ros1/docker-compose.ros1-are.yml exec rosclaw rosclaw app cmu-go "向上走 3 米"
docker compose -f docker/ros1/docker-compose.ros1-are.yml exec rosclaw rosclaw app cmu-go "去 inspection_a"
```

`cmu-go` 默认使用确定性规则解析，不需要 LLM API。当前屏幕方向语义为：

- “向上” = RViz 屏幕向上，map `+X`
- “向下” = RViz 屏幕向下，map `-X`
- “向左” = RViz 屏幕向左，map `+Y`
- “向右” = RViz 屏幕向右，map `-Y`

places 文件和显式 `x=... y=...` 坐标仍按原始 map 坐标执行。

`cmu-go` 会发布：

- `/way_point: geometry_msgs/PointStamped`
- `/speed: std_msgs/Float32`
- `/stop: std_msgs/Int8`

并记录：

- `summary.json`
- `odom_trace.jsonl`
- `cmd_vel.jsonl`
- `path_trace.jsonl`
- `waypoints.jsonl`
- `ros_topics.txt`

## 探索任务切换

```bash
docker compose -f docker/ros1/docker-compose.ros1-are.yml exec rosclaw rosclaw app cmu-explore start
docker compose -f docker/ros1/docker-compose.ros1-are.yml exec rosclaw rosclaw app cmu-explore pause
docker compose -f docker/ros1/docker-compose.ros1-are.yml exec rosclaw rosclaw app cmu-explore resume
docker compose -f docker/ros1/docker-compose.ros1-are.yml exec rosclaw rosclaw app cmu-explore stop
```

`pause` 会同时发布 `/stop=1` 和 `/speed=0`，并让 vendored ARiADNE2 停止发布新 `/way_point`。它不会杀掉 ARiADNE2 进程，因此 planner 内部图、历史 waypoint 和 policy 状态保留在内存里。`resume` 发布 `/stop=0` 后让同一个 planner 继续。

`start` 和 `resume` 会重新发布 `/speed=${CMU_SPEED}`，避免 `pause` 后 `/speed=0` 的 latch 状态让车辆恢复规划但不移动。

## LLM 交互式控制

终端 A 先启动常驻仿真和单个 CMU RViz：

```bash
./scripts/start_cmu_rviz.sh
```

终端 B 进入交互式控制台。脚本会自动读取本地 `.env`：

```bash
./scripts/cmu_chat.sh
```

进入后可以直接输入：

```text
向上走20米
右上走5米
前进3米
先向上走5米，再向右走3米
以半径2米转一圈
开始探索
暂停
继续
停止
退出
```

`cmu-chat` 必须使用 LLM；如果没有 `DEEPSEEK_API_KEY` 或 `OPENAI_API_KEY` 会直接退出，不会降级到规则 parser。LLM 只能输出受限 JSON，地点必须来自 `docker/ros1/places.campus.yaml`，相对移动默认限制不超过 `CMU_MAX_RELATIVE_M=20m`，不清楚的指令会反问。

当前能力边界是：地点导航、显式坐标导航、屏幕方向相对移动、机器人朝向相对移动、多步 waypoint 顺序任务、圆形 waypoint 轨迹，以及探索控制 `start/pause/resume/stop`。相对移动默认上限由 `CMU_MAX_RELATIVE_M=20` 控制；圆形轨迹默认半径上限由 `CMU_MAX_CIRCLE_RADIUS=6` 控制。ROSClaw 不会直接接管 `/cmd_vel` 或执行未校验的底层速度曲线，避免和 CMU `pathFollower` 竞争。

`cmu-chat` 不会因为目标“可能不可达”而提前拒绝；只要 schema 和配置边界合法，就会下发给仿真执行。不可达区域会在执行后以 timeout、最终误差和 artifact 形式报告失败。

交互控制台的执行结果会基于真实执行状态生成，例如“已成功开始探索”“已成功暂停探索”“已成功向右移动 5 米，最终误差 0.8 米”。LLM 只负责在事实摘要基础上润色回复；如果润色失败，会回退到确定性状态文案。

交互输出按阶段显示：接受任务时输出“开始...”，执行中按 `CMU_CHAT_PROGRESS_INTERVAL` 输出“正在...”，结束时输出“成功/失败/已取消”。如果一个移动任务执行中输入新的移动任务，新任务会自动抢占旧任务并先发布 `/stop=1`；如果自主探索中输入人工移动，ROSClaw 会先暂停 ARiADNE2，移动结束后不会自动恢复，必须由用户明确输入“继续探索”。

## Web 任务中控台

任务中控台读取 `practice_data/app_runs` 下的 CMU artifact，展示任务列表、成功率、耗时、最终误差、阶段事件和 2D 轨迹回放。它是只读的，不会从网页下发机器人控制命令，因此不会和 `cmu-chat` 抢任务管理权。

启动方式：

```bash
./scripts/cmu_dashboard.sh
```

默认地址：

```text
http://localhost:18770
```

也可以直接在容器内运行：

```bash
docker compose -f docker/ros1/docker-compose.ros1-are.yml exec rosclaw rosclaw app cmu-dashboard
```

常用配置：

- `CMU_DASHBOARD_PORT=18770`
- `CMU_DASHBOARD_HOST=0.0.0.0`
- `CMU_DASHBOARD_MAX_POINTS=2000`

HTTP API 包括：

- `GET /api/health`
- `GET /api/tasks`
- `GET /api/tasks/{episode_id}`
- `GET /api/tasks/{episode_id}/trajectory`
- `GET /api/stats`

## GUI / RViz

WSL2 + WSLg 下可以启动 RViz-only GUI：

```bash
./scripts/start_cmu_rviz.sh
```

如果还要 Gazebo GUI：

```bash
HEADLESS=false ./scripts/start_cmu_rviz.sh
```

GUI 依赖宿主 WSLg/X11 状态；纯 headless 是最低验收主线。

RViz 默认只展示 CMU 仿真主界面，不包含相机 Image 面板。
