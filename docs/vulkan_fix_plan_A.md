# Vulkan/Isaac 修复计划 A：最小修复

> 目标：恢复 `vulkaninfo` 能枚举 RTX 4090，让 Isaac Sim/Lab 的 `evaluate.py` 能在 headless 下启动。
>
> 适用环境：Ubuntu 20.04 + RTX 4090 + NVIDIA 驱动 565.77（runfile 安装）+ SSH/无桌面会话。
>
> 不动内核模块、不重启、不停桌面；全部步骤可逆，预计耗时 5–10 分钟。

---

## 0. 背景与根因

`vulkaninfo` 报 `Could not get 'vkCreateInstance' via 'vk_icdGetInstanceProcAddr' for ICD libGLX_nvidia.so.0` 已确认有三层独立原因：

1. 当前用户 `pyun` 不在 `render`/`video` 组 → 无法打开 `/dev/dri/renderD128` 与 `/dev/dri/card0`。`libGLX_nvidia.so.0` 内部初始化时打开 DRM 设备失败，loader 拿不到 `vkCreateInstance`。
2. `/usr/share/glvnd/egl_vendor.d/` 仅有 `50_mesa.json`，缺 `10_nvidia.json` → 没有可用的 NVIDIA EGL ICD。这是 565.77 通过 .run 安装时漏掉的用户态文件。
3. `DISPLAY=:1` 无对应 X 服务（`/tmp/.X11-unix/` 只有 `X0`），且 SSH 无 Xauthority → GLX 路径上必然报 `XCB failed to connect to the X server`。Isaac headless 必须走 EGL 而非 GLX。

`nvidia-smi`、`/proc/driver/nvidia/version`、CUDA PyTorch 训练均正常，说明**内核模块没问题**，问题完全在用户态。

> ICD 文件本身（`/usr/share/vulkan/icd.d/nvidia_icd.json`）此前被 Docker 误挂载成目录，已通过 `sudo rm -rf` + `sudo cp /etc/vulkan/icd.d/nvidia_icd.json ...` 修复，**这一步无需再做**。

---

## 1. 步骤总览

| # | 动作 | 何处执行 | 是否需 sudo | 是否需重新登录 |
|---|------|----------|-------------|----------------|
| A1 | 把 `pyun` 加入 `render`、`video` 组 | 本机或 SSH | ✅ | ✅（或 `newgrp`） |
| A2 | 写 `/usr/share/glvnd/egl_vendor.d/10_nvidia.json` | 任意终端 | ✅ | ❌ |
| A3 | 设环境变量 `VK_ICD_FILENAMES`、`XDG_RUNTIME_DIR` | 写入 `~/.bashrc` | ❌ | ❌（`source` 即可） |
| A4 | 验证 `vulkaninfo` 看到 RTX 4090 | 任意终端 | ❌ | ❌ |
| A5 | 验证 Isaac 单环境 `evaluate.py` 启动 | `isaaclab` conda env | ❌ | ❌ |

> 顺序敏感：A1 必须在 A4/A5 之前，且 **A1 完成后必须重新登录 SSH**，否则 `groups` 不会更新，`vulkaninfo` 仍会报权限问题。

---

## 2. 详细步骤

### A1. 加入 GPU 组（必做）

在本机或当前 SSH 终端执行：

```bash
sudo usermod -aG render,video pyun
```

**完全退出当前 SSH 会话并重新登录**。tmux 内的旧 shell 不会自动获得新组，必要时也要在 tmux 内部 `exit` 重开。

验证：

```bash
groups | tr ' ' '\n' | grep -E '^(render|video)$'
# 应该输出两行：render / video

python3 -c "import os; fd = os.open('/dev/dri/renderD128', os.O_RDWR); os.close(fd); print('DRI OK')"
# 应输出：DRI OK
```

### A2. 写 EGL vendor ICD（必做）

```bash
sudo tee /usr/share/glvnd/egl_vendor.d/10_nvidia.json >/dev/null <<'EOF'
{
    "file_format_version" : "1.0.0",
    "ICD" : {
        "library_path" : "libEGL_nvidia.so.0"
    }
}
EOF

ls -la /usr/share/glvnd/egl_vendor.d/
# 应同时看到 10_nvidia.json 和 50_mesa.json
```

> 注意：`library_path` **不要**写绝对路径；让 ld.so 通过 `ldconfig` 找 `libEGL_nvidia.so.565.77`。

### A3. 设置环境变量（必做）

把以下行追加到 `~/.bashrc`（幂等，可重复执行）：

```bash
cat >> ~/.bashrc <<'EOF'

# --- NVIDIA Vulkan headless (Isaac Sim) ---
export VK_ICD_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
if [ -z "${XDG_RUNTIME_DIR:-}" ]; then
    export XDG_RUNTIME_DIR=/tmp/runtime-$USER
    mkdir -p "$XDG_RUNTIME_DIR" && chmod 700 "$XDG_RUNTIME_DIR"
fi
# 在 SSH/headless 下避免被 :1 / :0 误导
unset DISPLAY 2>/dev/null || true
# --- end NVIDIA Vulkan headless ---
EOF

source ~/.bashrc
```

> 如果你**确实**有桌面会话且要 GUI 渲染，则不要 `unset DISPLAY`，改为：
> `export DISPLAY=:0`、`export XAUTHORITY=/run/user/$(id -u)/gdm/Xauthority`。本仓库 Isaac 用法是 `--headless`，所以默认 `unset` 更稳。

### A4. 验证 Vulkan

```bash
vulkaninfo 2>&1 | grep -E 'deviceName|driverName|ERROR|llvmpipe' | head -10
```

期望：

```text
deviceName       = NVIDIA GeForce RTX 4090
driverName       = NVIDIA
deviceName       = llvmpipe (LLVM 12.0.0, 256 bits)   # 这是软件渲染回退，正常
```

不应再出现：

```text
ERROR ... loader_scanned_icd_add: Could not get 'vkCreateInstance' ...
ERROR_INITIALIZATION_FAILED
```

也跑一遍仓库自带的脚本：

```bash
cd ~/Docker/DynamicVLA/Code/DynamicVLA
./docker/scripts/verify-gpu.sh 2>&1 | tail -30
```

应在 `vulkaninfo --summary` 段看到 NVIDIA GPU。

### A5. 验证 Isaac 单环境冒烟

新开 tmux 窗口（继承新组与新环境变量）：

```bash
conda activate isaaclab
cd ~/Docker/DynamicVLA/Code/DynamicVLA/simulations

python evaluate.py \
  --enable_cameras --headless -n 1 \
  --env_cfg ../tests/1-1_place_franka_beer07d_O02_01048625_91e1.json \
  2>&1 | tee /tmp/isaac_smoke.log
```

期望日志里**不再**出现：

```text
[Error] [omni.platforminfo.plugin] failed to create an instance for graphics API: vulkan
Driver Version: 0
No device could be created
```

应能看到 `omni.kit` 起来、相机被建出。Stage load 后再启动 inference 端做闭环测试。

---

## 3. 常见故障与处理

### 3.1 `groups` 看不到 `render`/`video`
- 你没有完全断开 SSH。`exit` 整个 SSH 会话再 `ssh` 进来；tmux 中也要 `exit` 重开 shell。
- 仍不行：`getent group render | grep pyun`、`getent group video | grep pyun` 看 `/etc/group` 是否已写入。若没写入，重跑 `sudo usermod -aG render,video pyun`。

### 3.2 DRI 仍 `Permission denied`
- 检查 `ls -l /dev/dri/renderD128`，组应是 `render`。如果是 `video` 而你只加了 `render`：把两个组都加上（步骤 A1 已是 `-aG render,video`）。
- 检查 udev：`udevadm info /dev/dri/renderD128 | grep -i group`。

### 3.3 `vulkaninfo` 仍报 `Could not get vkCreateInstance ... libEGL_nvidia.so.0`
- 确认 `VK_ICD_FILENAMES` 已生效：`echo $VK_ICD_FILENAMES`。
- 确认 `libEGL_nvidia.so.0` 可被 ld 找到：`ldconfig -p | grep libEGL_nvidia`。应输出 `libEGL_nvidia.so.0 => /lib/x86_64-linux-gnu/libEGL_nvidia.so.565.77`。
- 用 loader debug 看具体在哪一步挂：

```bash
VK_LOADER_DEBUG=all VK_ICD_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json \
  vulkaninfo 2>&1 | grep -iE 'icd|nvidia|error' | head -30
```

- 看是否被某个旧 ICD 干扰：`ls /usr/share/vulkan/icd.d/`，确保 `nvidia_icd.json` 是 565.77 的（`api_version` 1.3.x），且**是文件不是目录**。

### 3.4 只看到 `llvmpipe`，没有 NVIDIA 设备
- 99% 是 `/dev/dri/*` 没权限；回到 3.2。
- 内核模块没起来（极少）：`lsmod | grep nvidia`，应有 `nvidia_drm`、`nvidia_modeset`、`nvidia_uvm`、`nvidia`。缺了就 `sudo modprobe nvidia_drm modeset=1`。
- `dmesg | grep -i nvidia | tail -30` 看是否有 `module mismatch` 或 `Failed to initialize NVML`；若有，说明驱动状态坏了，需要走「计划 B」的 runfile 重装。

### 3.5 `evaluate.py` 仍报 `Driver Version: 0` / `No device could be created`
- 先确保 A4 通过。**`vulkaninfo` 都看不到 GPU 时 Isaac 必然挂**。
- 确认环境变量在 conda 子 shell 里也存在：`conda activate isaaclab && env | grep VK_ICD`。
- Isaac 需要的扩展可能依赖 `VK_KHR_external_semaphore_fd` 等：`vulkaninfo | grep -i 'external_semaphore_fd'`。NVIDIA proprietary 都支持，没看到通常是走到了 llvmpipe，回到 3.4。
- 试加：`export ENABLE_VULKAN_VALIDATION=0`、`export OMNI_KIT_ALLOW_ROOT=1`。

### 3.6 `XDG_RUNTIME_DIR` 警告残留
- 不阻塞运行，但如果想消除：`mkdir -p /tmp/runtime-$USER && chmod 700 /tmp/runtime-$USER`，并确保 A3 已 source。

### 3.7 Docker 内运行 Isaac 时再次报错
- 容器内必须设 `NVIDIA_DRIVER_CAPABILITIES=all`（至少 `compute,graphics,utility`）。
- `docker run --gpus all --runtime=nvidia ... --env NVIDIA_DRIVER_CAPABILITIES=all ...`。
- 容器内同样需要 `/usr/share/glvnd/egl_vendor.d/10_nvidia.json` 与 `VK_ICD_FILENAMES`。仓库已有 `docker/scripts/fix-vulkan-icd.sh` 可在容器入口跑。

### 3.8 桌面登录失效（Wayland/GNOME）
- 计划 A 不动 X/Wayland；如果你后来发现本机登录 GNOME 卡在 logo，说明你额外做了别的事（比如 B 路径），不是 A 的副作用。

---

## 4. 失败回滚

逐项可逆：

```bash
# 取消组权限
sudo gpasswd -d pyun render
sudo gpasswd -d pyun video
# 删除 EGL ICD
sudo rm -f /usr/share/glvnd/egl_vendor.d/10_nvidia.json
# 移除 ~/.bashrc 末尾追加段（手工编辑或 sed）
sed -i '/--- NVIDIA Vulkan headless (Isaac Sim) ---/,/--- end NVIDIA Vulkan headless ---/d' ~/.bashrc
```

---

## 5. 何时升级到「计划 B：重装驱动」

只有以下条件**至少一项**成立时才考虑 B：

1. A1–A4 全部做完且重新登录后，`vulkaninfo` 仍报 `ERROR_INITIALIZATION_FAILED`、且 `VK_LOADER_DEBUG=all` 显示 NVIDIA ICD 直接失败而不是 llvmpipe 兜底。
2. `dmesg | grep -i nvidia` 出现 `module mismatch`、`API mismatch`、`Xid` 异常。
3. `/proc/driver/nvidia/version` 与 `nvidia-smi --query-gpu=driver_version` 报的版本不一致。
4. `nvidia-smi` 本身开始失败。

升级时**优先选 B2**（同版本 565.77 .run 重装），保留 CUDA/Docker 不动；不要走 B1（apt 装其他版本）。

---

## 6. 验收清单（旧，参考用；以 §8 为准）

- [ ] `groups` 输出包含 `render` 与 `video`
- [ ] `python3 -c "import os; os.close(os.open('/dev/dri/renderD128', os.O_RDWR))"` 不报错
- [ ] `ls /usr/share/glvnd/egl_vendor.d/10_nvidia.json` 存在
- [ ] `echo $VK_ICD_FILENAMES` 输出 `/usr/share/glvnd/egl_vendor.d/10_nvidia.json`
- [ ] `vulkaninfo | grep deviceName` 中包含 `NVIDIA GeForce RTX 4090`
- [ ] `./docker/scripts/verify-gpu.sh` 全部 [OK]
- [ ] `python evaluate.py --enable_cameras --headless -n 1 --env_cfg ../tests/1-1_*.json` 启动后能加载相机、不报 `Driver Version: 0`
- [ ] `inference.py` 端连上 evaluate 端，跑完 1 个 trial（任意成功/失败都算通过冒烟）

通过以上 8 项后，恢复 1800 trials 计划：见 `.cursor/plans/table1_测试计划_*.plan.md`。

---

## 7. 实测发现（2026-05-24，§1–§5 的事后修订）

> §1–§5 是**最初的诊断与计划**，基于 GLX/EGL/组权限 三层假设。实际跑 Isaac `evaluate.py` 后发现：**§1–§5 的修复是必要的「热身」，但不是 root cause**。真正阻塞 Isaac 的是另一条独立原因——**重复 ICD**。本节记录决定性证据、最终生效的修复，以及最新系统状态。

### 7.1 决定性证据

`/home/pyun/isaacsim/kit/logs/Kit/Isaac-Sim/4.5/kit_20260524_085829.log` 第 1910–1915 行（Isaac kit 自报错误）：

```text
[Error] [gpu.foundation.plugin] Multiple Installable Client Drivers (ICDs) are
found for the same GPU on the system. This causes the same GPU to be reported
multiple times by more than one driver and leads to instability or crash.

To verify the proper NVIDIA driver installation, only one of the following
icd.d folders should contain nvidia_icd.json file:
/etc/vulkan/icd.d
/usr/share/vulkan/icd.d
```

紧接的 GPU 列表里**同一张** RTX 4090（UUID 完全相同 `c4523744..`）被列出两次，Isaac kit 主动 reject 这种状态，输出：

- `[Error] omni.gpu_foundation_factory.plugin: Failed to create any GPU devices, including an attempt with compatibility mode.`
- 接着 `carb.graphics-vulkan.plugin: VkResult: ERROR_INCOMPATIBLE_DRIVER` / `vkCreateInstance failed. Vulkan 1.1 is not supported`

后两条是 reject 的兜底翻译，**不是真的驱动不兼容**。

### 7.2 重复 ICD 怎么来的

直接归因到一次「修复」操作。Docker 单文件 bind-mount 把宿主 `/usr/share/vulkan/icd.d/nvidia_icd.json` 误变成目录后，本机用：

```bash
sudo rm -rf /usr/share/vulkan/icd.d/nvidia_icd.json
sudo cp /etc/vulkan/icd.d/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json
```

恢复了文件，但等于**在 vulkan loader 的两条默认搜索路径里同时写了同一份 ICD JSON**：

```
/etc/vulkan/icd.d/nvidia_icd.json          ← .run 安装产物
/usr/share/vulkan/icd.d/nvidia_icd.json    ← 上面 cp 出来的副本
```

vulkan loader 把同一物理 GPU 通过两条 ICD 各加载一次。系统 `vulkaninfo` 容忍这种重复（只是会列两次同一 GPU），**Isaac kit 不容忍**。

### 7.3 真因（按贡献度排序）

| # | 真因 | 贡献度 | 证据 |
|---|------|--------|------|
| **A** | 同一 ICD JSON 出现在 vulkan loader 两个默认搜索路径里 | **决定性**：触发 Isaac kit 主动 reject | kit log 第 1910 行 |
| **B** | 默认 ICD 指向 `libGLX_nvidia.so.0`，SSH 无 X 时 ICD 初始化失败 | 高：让 host `vulkaninfo` 也走不通 GLX 路径 | `XCB failed to connect to the X server` |
| **C** | `pyun` 不在 `render`/`video` 组，无法 `O_RDWR` 打开 `/dev/dri/*` | 中：让所有 ICD 都打不开 GPU 设备节点 | `getfacl /dev/dri/renderD128`、`Permission denied` |
| **D** | `DISPLAY=:1` 但只有 `X0`，且 SSH 无 Xauthority | 低：仅周边噪声 | `XDG_RUNTIME_DIR not set` 等 |

GUI 桌面下 A/B/C/D 都被 GDM + 有效 X session 掩盖；SSH headless 下四个全部暴露，且 A 是 Isaac 唯一的硬阻塞。

### 7.4 最终生效的修复（按时间分两阶段）

#### 阶段 I：恢复 SSH 下 vulkan 可用（先做，但只是热身）

| 改动 | 目的 |
|------|------|
| `sudo usermod -aG render,video pyun` + 重新登录 SSH | 修 **C**：让进程能打开 `/dev/dri/*` |
| `unset DISPLAY` + `export XDG_RUNTIME_DIR=/tmp/runtime-$USER` | 修 **D**：消除无 X 的警告 |
| 写 `/usr/share/glvnd/egl_vendor.d/10_nvidia.json` + `export VK_ICD_FILENAMES` 指向它 | **当时误判**为 EGL ICD 缺失，临时绕开 GLX |

阶段 I 后 host `vulkaninfo` 能看到 RTX 4090，但 **Isaac 仍挂**（A 没解决）。

#### 阶段 II：消除重复 ICD（决定性修复）

```bash
# 1. 默认搜索路径里只保留一份 ICD
sudo mv /usr/share/vulkan/icd.d/nvidia_icd.json \
        /usr/share/vulkan/icd.d/nvidia_icd.json.dup-bak

# 2. 把保留的那份 ICD 改成 EGL 路径（headless 友好）
sudo cp /etc/vulkan/icd.d/nvidia_icd.json /etc/vulkan/icd.d/nvidia_icd.json.glx-bak
sudo tee /etc/vulkan/icd.d/nvidia_icd.json >/dev/null <<'EOF'
{
    "file_format_version" : "1.0.1",
    "ICD": {
        "library_path": "libEGL_nvidia.so.0",
        "api_version" : "1.3.289"
    }
}
EOF

# 3. 撤回阶段 I 的 EGL vendor 文件，避免某些 loader 把它也当 ICD（防新增重复）
sudo mv /usr/share/glvnd/egl_vendor.d/10_nvidia.json \
        /usr/share/glvnd/egl_vendor.d/10_nvidia.json.bak

# 4. 当前 shell 取消 VK_ICD_FILENAMES（默认路径已直接指向 EGL，不再需要）
unset VK_ICD_FILENAMES
```

### 7.5 最终系统状态

```
/etc/vulkan/icd.d/
├── nvidia_icd.json          ← 唯一在用，library_path: libEGL_nvidia.so.0
└── nvidia_icd.json.glx-bak  ← 备份（原 GLX 版本）

/usr/share/vulkan/icd.d/
├── intel_icd.x86_64.json
├── lvp_icd.x86_64.json
├── nvidia_icd.json.dup-bak  ← 备份（重复的副本，已挪走）
└── radeon_icd.x86_64.json

/usr/share/glvnd/egl_vendor.d/
├── 10_nvidia.json.bak       ← 备份（阶段 I 误加的，已挪走）
└── 50_mesa.json
```

→ **Isaac kit 现在不再依赖任何环境变量**（`VK_ICD_FILENAMES` 不需要），只依赖唯一一份 `/etc/vulkan/icd.d/nvidia_icd.json`。

### 7.6 实测验证（2026-05-24 09:09 通过）

| 节点 | 证据 |
|------|------|
| Vulkan 初始化 | kit log: `\| 0 \| NVIDIA GeForce RTX 4090 \| Yes: 0 \|`（Active: Yes） |
| Scene 加载 | `Recovering test environment from ../tests/1-1_*.json` |
| Trial 完成 | `[Test01] Test done with SUCCESS in 150 steps and 143 actions` |
| Success rate | `success_rate: 1.0`（1/1） |
| Video 输出 | `output/evaluation_smoke_video/dynamic-vla-DOM/0000/1-1_place_franka_beer07d_O02_01048625_91e1-0524-090953-SUCCESS.mp4` |
| Inference chunk 时间 | `~0.10–0.11 s/chunk`（4090，跟之前 1800-trial 基准一致） |

### 7.7 回滚（一键可逆）

```bash
# 全量回到修复前
sudo mv /etc/vulkan/icd.d/nvidia_icd.json.glx-bak \
        /etc/vulkan/icd.d/nvidia_icd.json
sudo mv /usr/share/vulkan/icd.d/nvidia_icd.json.dup-bak \
        /usr/share/vulkan/icd.d/nvidia_icd.json
sudo mv /usr/share/glvnd/egl_vendor.d/10_nvidia.json.bak \
        /usr/share/glvnd/egl_vendor.d/10_nvidia.json
# 组权限若也想撤回：
sudo gpasswd -d pyun render
sudo gpasswd -d pyun video
```

### 7.8 副作用与已知风险

- **EGL 路径会作为系统默认 vulkan 路径**：所有 vulkan 应用（Steam/Chrome/Obsidian/Isaac）都会走 NVIDIA EGL Vulkan 实现。NVIDIA EGL 是完整 Vulkan 1.3 实现，**没有功能丢失**；唯一可能的副作用是某些桌面 GLX swapchain 路径需要重新协商，但 Isaac 训练/评估不受影响。
- **GUI 桌面下未做对照**：本次修复在 SSH headless 下验证通过；下次坐到物理屏幕前登录 GNOME，建议用 `vkcube` / Steam 等检验一下 vulkan 应用仍正常。如有问题，按 §7.7 一键回滚 GLX 版本即可。
- **Docker 容器内的 vulkan 不会自动修好**：本次只修了 host。在容器里跑 Isaac 时同样需要：单一 ICD JSON、`NVIDIA_DRIVER_CAPABILITIES=all`、容器内用户在 `render`/`video` 组、必要时挂载 host 的 ICD 进去。

---

## 8. 最终验收清单（替代 §6）

环境层：

- [x] `groups` 输出包含 `render` 与 `video`
- [x] `python3 -c "import os; os.close(os.open('/dev/dri/renderD128', os.O_RDWR))"` 不报错
- [x] **只**在 `/etc/vulkan/icd.d/` 看到一份 `nvidia_icd.json`，且 `library_path` 是 `libEGL_nvidia.so.0`
- [x] `/usr/share/vulkan/icd.d/nvidia_icd.json` 不存在（只剩 `.dup-bak`）
- [x] `/usr/share/glvnd/egl_vendor.d/10_nvidia.json` 不存在（只剩 `.bak`）
- [x] `vulkaninfo` 在不设 `VK_ICD_FILENAMES` 时只列出**一个** `NVIDIA GeForce RTX 4090` + 一个 `llvmpipe`，无 ERROR

Isaac 层：

- [x] `evaluate.py --enable_cameras --headless -n 1 --env_cfg ../tests/1-1_*.json` 启动 kit log 中出现 `Active: Yes: 0`，无 `Multiple ICDs` / `Failed to create any GPU devices`
- [x] `inference.py` 连上 evaluate 端，跑完 1 个 trial 写出 mp4
- [x] mp4 可正常播放，能看到 Franka 操作贴图正确的啤酒罐（联动验证 §0 之外的 texture 修复）

后续待办（未在本次修复范围）：

- [ ] 把 `XDG_RUNTIME_DIR` + SSH-only `unset DISPLAY` 持久化到 `~/.bashrc`（`VK_ICD_FILENAMES` 已不需要）
- [ ] 重跑 Table I（先 90×1 校验，再 90×20 = 1800 trials），与 2026-05-22 的 SR=51.06% 基准对比
- [ ] 更新 `docker/scripts/fix-vulkan-icd.sh` / `verify-gpu.sh`，加上「重复 ICD 检测」（同名文件同时存在于 `/etc/vulkan/icd.d/` 与 `/usr/share/vulkan/icd.d/` 时 fail）
- [ ] 容器内 Isaac 的 vulkan 修复（如果之后要回到 Docker 跑）

