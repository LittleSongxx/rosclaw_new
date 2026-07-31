# TY1200 外设启用 Runbook

本文件覆盖 `ty1200-ops peripheral_inventory` / `doctor` 报告为
"硬件存在但未启用"（WARN）的项目。所有启用操作均为 **L5（仅管理员手工）**；
启用后用对应 Skill 操作验收。截至 2026-07-31 的本机状态见各节"现状"。

> 原则：驱动/内核级改动不在 ty1200-platform-ops Skill 范围内；
> 本 runbook 是给管理员的操作手册，Skill 负责启用前后的状态验收。

---

## 1. CAN-FD（2 路）

**现状**：PCI 控制器 `04:00.0 CANBUS [1c29:2004]` 存在，未绑定驱动，
无 can0/can1（doctor 报 `hardware_present_no_driver`）。

**启用步骤**

1. 向天数支持索取该控制器（1c29:2004）的 Linux 驱动（通常命名为
   `iluvatar-can` 或基于 SJA1000/CTU CAN FD IP 的内核模块）。
2. 安装模块：`sudo insmod <driver>.ko` 或放入
   `/lib/modules/$(uname -r)/extra/` 后 `depmod -a`。
3. 配置接口：
   ```bash
   sudo ip link set can0 up type can bitrate 1000000 dbitrate 5000000 fd on
   sudo ip link set can1 up type can bitrate 1000000 dbitrate 5000000 fd on
   ```
4. 持久化：写入 `/etc/systemd/network/` 或 netplan；驱动加入
   `/etc/modules-load.d/can.conf`。

**验收**

```bash
ty1200-ops '{"operation":"peripheral_inventory"}'   # can.status == "ready"
candump can0                                        # 收发实测（机器人侧）
```

---

## 2. EtherCAT（2× RJ45, Intel igb）

**现状**：`enp5s0` / `enp6s0`（igb 驱动）存在，operstate=down。

EtherCAT 使用方式有两种，按机器人方案选择：

### 方案 A：IgH EtherCAT Master（推荐用于实时控制）

1. 安装 IgH master（源码编译或厂商包）：
   ```bash
   # 以 IgH etherlab 为例
   ./configure --with-linux-dir=/lib/modules/$(uname -r)/build \
               --enable-generic --disable-8139too
   make && sudo make install
   ```
2. 配置主站绑定网口 MAC（`/etc/ethercat.conf` 的 `MASTER0_DEVICE`）。
3. 加载：`sudo systemctl start ethercat`（`ec_generic` 或 `ec_igb` 模块）。

### 方案 B：SOEM（用户态，开发期更快）

无需内核主站，直接用 igb 口原始 socket；把口 up 起来即可：
```bash
sudo ip link set enp5s0 up
```

**验收**

```bash
ty1200-ops '{"operation":"peripheral_inventory"}'   # ethercat_nics[*].operstate
ethercat slaves   # 方案 A；或 SOEM 的 slaveinfo（方案 B）
```

注意：EtherCAT 主站扫描/从站控制属于机器人运动域，不经过本 Skill。

---

## 3. GPIO（8 路）

**现状**：无 `/dev/gpiochip*`（doctor 报 `not_exposed`，WARN）。

**启用步骤**

1. GPIO 挂在 Intel PCH pinctrl 上，确认内核配置：
   ```bash
   zgrep PINCTRL_METEORLAKE /boot/config-$(uname -r)
   ```
2. 需要 `pinctrl-meteorlake`（或对应 PCH）模块；若主线内核未启用，
   向天数索取定制内核/配置。
3. 排针电平定义（1.8V/3.3V）以硬件原理图为准。

**验收**：`ls /dev/gpiochip*`；`ty1200-ops '{"operation":"peripheral_inventory"}'`
中 `gpio.status == "ready"`；`gpiodetect` / `gpioinfo` 查看线数。

---

## 4. SPI（1 路）

**现状**：无 `/dev/spidev*`（`not_exposed`）。

**启用步骤**：SPI 控制器为 Intel 00:1f.5（Serial bus controller）。
主线内核通常将其作为 LPSS SPI；用户态访问需要：
1. 加载 `spi-pxa2xx-platform`（或对应 LPSS 驱动）；
2. 通过 ACPI 覆盖或 `spidev` 绑定启用用户态节点：
   ```bash
   echo spidev | sudo tee /sys/bus/spi/devices/spi-X.Y/driver_override
   ```
具体节点名随驱动而定；联系天数确认该口的 ACPI 描述。

**验收**：`ls /dev/spidev*`；peripheral_inventory 中 `spi.status == "ready"`。

---

## 5. PREEMPT_RT 实时内核

**现状**：`6.8.0-85-generic`，完整抢占（PREEMPT full），**非 RT**
（`/sys/kernel/realtime` 不存在）。ROS2 常规场景够用；硬实时闭环控制
（小脑，EtherCAT 1kHz+ 周期）建议 RT 内核。

**启用步骤**

1. 联系天数提供 TY1200 的 PREEMPT_RT 内核 ISO/包（官方支持 PREEMPT_RT，
   但自行编译需重签 CoreX 驱动）。
2. 安装后确认驱动：`export LD_LIBRARY_PATH=/usr/local/corex/lib; ixsmi`。
3. 可选强化（cmdline）：
   `isolcpus=<E-core列表> nohz_full=<同> rcu_nocbs=<同> irqaffinity=<P-core列表>`

**验收**

```bash
ty1200-ops '{"operation":"rt_check"}'     # is_preempt_rt == true
cyclictest -m -p 99 -i 1000 -l 100000     # 延迟分布（rt-tests 包）
```

**警告**：内核升级会丢失天数 GPGPU 驱动（官方文档明确说明），升级后必须
重装 `corex-driver-linux64-*.run`；模型栈会自动恢复（compose 自启），
但首次启动前用 `ty1200-preflight` 确认驱动。

---

## 6. 5G/GNSS（M.2 B-Key）

**现状**：空槽（`not_installed`）。装入 3042/3052 模组后应出现
`/dev/cdc-wdm*`（MBIM）或 `/dev/ttyUSB*`（AT），用 ModemManager 管理。

**验收**：peripheral_inventory 中 `modem_5g.status == "ready"`；
`mmcli -L` / `mmcli -m 0`。

---

## 7. NPU（Intel AI Boost）

**现状**：`intel_vpu` 已绑定 `/dev/accel/accel0`（**可直接用**）。

**使用方式**：OpenVINO 2024+ 的 `NPU` 设备。宿主机未装 openvino
（PyPI 不可达），建议在容器中用（或从天数 APPS 渠道获取 whl）：
```python
import openvino as ov
core = ov.Core()
print(core.available_devices)   # 应含 'NPU'
```

**验收**：`ty1200-ops '{"operation":"accelerator_check"}'` 中
`npu.driver == "intel_vpu"`；OpenVINO benchmark_app 实测。

---

## 8. 加密芯片 ATSHA204A（选配）

**现状**：未检测到（选配，本机大概率未贴）。若硬件在板，通常挂在某
I2C 总线 0x60；用 `i2cdetect -y -r <bus>`（**只读探测，管理员执行**）
在 3 条 DesignWare 外部总线上确认，然后用 Microchip cryptoauthlib 对接。

**验收**：peripheral_inventory 中 `secure_chip.status == "detected"`。
