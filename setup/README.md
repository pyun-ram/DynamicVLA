# 新机器一键配置

前提：已安装 **Docker**、**NVIDIA 驱动**（≥ 535.129）、**nvidia-container-toolkit**，`nvidia-smi` 正常。  
**不需要** `docker login nvcr.io`。

## 流程

```bash
git clone https://github.com/hzxie/DynamicVLA.git
cd DynamicVLA

# 1) 下载资源
./download_assets.sh                           # 仅 Isaac Sim zip（构建镜像用）
./download_assets.sh --train                   # 仅训练集（不会下 Isaac）
./download_assets.sh --train --isaac           # 训练集 + Isaac zip

# 其它可选
./download_assets.sh --test --objects          # 跑 benchmark
./download_assets.sh --scenes                  # 完整场景（较大）
./download_assets.sh --all                     # test + objects + scenes + train（不含 Isaac）
./download_assets.sh --model                   # 预训练权重

# 2) 构建镜像
./build_docker.sh

# 3) 开发容器
./docker/run-dev.sh
```

## 数据目录

默认 **数据根目录** = 仓库的**上一级目录**（与 README 中 `PROJECT_ROOT` 一致）：

```text
/path/to/PROJECT_ROOT/          ← --data-root 可改
├── objects/
├── scenes/
├── tests/
├── test-envs.txt
├── datasets/DOM/               # --train
└── DynamicVLA/                 # git clone
    ├── docker/vendor/*.zip     # Isaac Sim（--isaac）
    └── pretrained_weights/     # --model
```

自定义数据根：

```bash
./download_assets.sh --data-root /data/dynamicvla --test --objects
export DYNAMICVLA_DATA_ROOT=/data/dynamicvla
```

## 下载源（`setup/urls.sh`）

| 资源 | 来源 |
|------|------|
| Isaac Sim 4.5 zip | NVIDIA CDN（`--isaac`） |
| DOM Testing Set | Google Drive + **gdown**（`--test`） |
| DOM 3D Objects | Google Drive + **gdown**（`--objects`） |
| DOM 3D Scenes | infinitescript gateway（`--scenes`） |
| DOM 训练集 | Hugging Face `hzxie/DOM` |
| 预训练模型 | Hugging Face `hzxie/dynamic-vla-DOM` |

## 容器内路径示例

挂载仓库后，若数据在仓库上一级（默认）：

```bash
conda activate isaaclab
cd /workspace/DynamicVLA
python simulations/evaluate.py \
  --scene_dir ../scenes \
  --output_dir ../output/evaluation \
  --env_cfg ../test-envs.txt \
  --enable_cameras --headless -n 20 --save
```

## 故障排查

- **`--test` / `--objects` 需要 gdown**：`pip install gdown`（脚本会自动 `pip install --user gdown`）
- **Google Drive 大文件**：若 gdown 报权限/配额，浏览器下载 zip 后放到 `${DATA_ROOT}/.download_cache/DOM-Test.zip` 或 `DOM-3D-Objects.zip`，再重新运行脚本（会跳过已存在文件）
- **Isaac zip 下载失败**：浏览器下载后放到 `docker/vendor/isaac-sim-standalone-4.5.0-linux-x86_64.zip`，再 `./build_docker.sh`
- **`--train` / `--model`**：需要能访问 Hugging Face（或镜像）
- **gateway 返回 HTML 而不是压缩包**（如 `DOM-Test.archive: HTML document`）：
  1. 脚本会先自动尝试从 HTML 中提取真实下载链接并重试；
  2. 若仍失败，按报错提示手动下载真实 `.zip/.tar/.tar.gz`；
  3. 将文件移动到报错里给出的缓存路径（例如 `${DATA_ROOT}/.download_cache/DOM-Test.archive`）；
  4. 重新运行原命令（如 `./download_assets.sh --test --objects`）。

### Hugging Face 下载慢（~500 KB/s）

进度条 `15.2M/15.3M` 是**当前某一个文件**，不是整个 DOM 数据集；全集可能很大，总时间会很长。

加速（在运行前 `export`）：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_MAX_WORKERS=8
# 不要用 hf_transfer + 镜像（容易卡在 0.00B）：
unset HF_HUB_ENABLE_HF_TRANSFER

./download_assets.sh --train
```

卡在 `0.00B` 不动时：

```bash
# 1) 关掉 hf_transfer，清锁后重试
unset HF_HUB_ENABLE_HF_TRANSFER
find ~/Docker/DynamicVLA/Code/datasets/DOM -name '*.lock' -delete
./download_assets.sh --train

# 2) 仍卡住：看是否列出 Repo file count
export HF_HUB_VERBOSITY=debug
./download_assets.sh --train
```
- **gateway 解压结构异常**：查看 `${DATA_ROOT}/.download_cache` 与目标目录，必要时手动整理
