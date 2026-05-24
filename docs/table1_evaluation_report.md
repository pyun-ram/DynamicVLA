# Phase 4：Table I 全维度 DOM 评估报告

> **论文**：[DynamicVLA](https://arxiv.org/abs/2601.22153)（arXiv:2601.22153）Table I  
> **完成时间**：2026-05-22 03:24:55  
> **硬件**：NVIDIA RTX 4090（单卡）

---

## 1. 实验概述

本报告汇总 **Phase 4** 在 **Dynamic Object Manipulation (DOM)** 测试集上复现论文 **Table I** 的结果。DOM 评估的是**动态物体操作**（物体在桌面上以不同速度/轨迹运动，机器人需闭环抓取并放置），**不包含可变形物体（deformable）** 任务。

| 项目 | 说明 |
|------|------|
| 论文指标 | Table I：九个子维度成功率（%）、平均路径长度（Path Len）、任务完成时间（Time）等 |
| 试验规模 | **1800 trials** = 9 子维度 × 10 场景/维 × 20 trials/场景（`-n 20`） |
| 环境数量 | 90 个仿真环境（`test-envs.txt`，对应 Table I） |
| 物体与运动 | 刚体日常物体（水果、容器等）；速度采样约 **0–0.75 m/s**（见论文 Sec. IV-C） |
| 策略推理 | `scripts/inference.py`，`-r euler -d -s`（Continuous Inference + streaming） |
| 仿真服务端 | `simulations/evaluate.py`，**未**使用 `--save` |
| 输出目录 | `output/evaluation_table1` |
| 主日志 | `output/logs/inference_table1.log` |

### 1.1 典型运行命令

**终端 A — Isaac Lab（`isaaclab`）**

```bash
cd /path/to/DynamicVLA/simulations
python3 evaluate.py \
    --scene_dir ../scenes \
    --object_dir ../objects \
    --output_dir ../output/evaluation_table1 \
    --env_cfg ../test-envs.txt \
    --enable_cameras --headless -n 20
```

**终端 B — 策略推理（`dynamicvla-train`）**

```bash
cd /path/to/DynamicVLA
python3 scripts/inference.py \
    -p pretrained_weights/dynamic-vla-DOM \
    -a dynamic-vla-DOM \
    -r euler -d -s \
    2>&1 | tee output/logs/inference_table1.log
```

先启动 `evaluate.py`，再启动 `inference.py`（ZMQ `127.0.0.1:3186/3188`）。

---

## 2. DOM 九维定义（论文 Sec. IV-B）

论文将 DOM 分为 **Interaction / Perception / Generalization** 三大类，每类 3 个子维度。仓库中环境 ID 以 `{维前缀}_` 开头（如 `1-1_place_...`、`1-3_long-horizon_...`）。

### 2.1 Interaction（交互：对**运动物体**的闭环响应）

| 代号 | 英文名 | 论文考察内容 |
|------|--------|----------------|
| **CR** | Closed-loop Reactivity | 物体以**不同速度**运动时的闭环反应能力 |
| **DA** | Dynamic Adaptation | **运动突变**（改向、碰撞后扰动等）后的适应 |
| **LS** | Long-horizon Sequencing | **长时程、多事件**连续交互中的动作编排与优先级 |

### 2.2 Perception（感知：动态场景中的视觉/语言 grounding）

| 代号 | 英文名 | 论文考察内容 |
|------|--------|----------------|
| **VU** | Visual Understanding | 区分**形状/纹理/材质相似**的物体 |
| **SR** | Spatial Reasoning | **杂乱或变化场景**中的位置与相对空间关系 |
| **MP** | Motion Perception | 解读物体**速度与方向**等运动线索 |

### 2.3 Generalization（泛化：未见物体/场景/运动模式）

| 代号 | 英文名 | 论文考察内容 |
|------|--------|----------------|
| **VG** | Visual Generalization | **未见外形/外观/场景布局** |
| **MG** | Motion Generalization | **未见速度范围、摩擦、轨迹模式** |
| **DR** | Disturbance Robustness | **外扰**（意外推碰、碰撞、传感器噪声等）下的稳定性 |

> **说明**：此前报告中将 DA 误写为 “Deformable Arrangement”、MP 误写为 “Multi-object Placement” 等，与论文不符；上表以 [`docs/2601.22153.pdf`](2601.22153.pdf) Sec. IV-B 为准。

---

## 3. 运行状态

| 状态项 | 结果 |
|--------|------|
| 整体状态 | **成功完成** |
| 环境数 | 90 |
| 试验总数 | **1800** |
| 日志确认 | `Evaluation completed. Total tests run: 1800`（2026-05-22 03:24:55） |
| 指标来源 | **`inference_table1.log`**（`evaluate_table1.log` 无 Table I 级汇总） |

运行末期偶发 `The output queue is full. Skipping an action.`；未加 `--save`，无评估视频。

---

## 4. 结果对比：本复现 vs 论文 DynamicVLA（Table I）

聚合方式：每维 10 个环境，`SR_dim = mean(每环境 success_rate) × 100`（每环境已为 20 trial 平均）；**Average** = 9 维 SR 的算术平均（与 1800 trial 全局成功率一致）。

| 大类 | Dim | 子维度（论文） | Ours % | Paper % | Δ |
|------|-----|----------------|--------|---------|---|
| Interaction | **CR** | Closed-loop Reactivity | 64.50 | 60.50 | +4.00 |
| Interaction | **DA** | Dynamic Adaptation | 56.00 | 38.50 | +17.50 |
| Interaction | **LS** | Long-horizon Sequencing | 35.00 | 40.50 | −5.50 |
| Perception | **VU** | Visual Understanding | 37.50 | 51.50 | −14.00 |
| Perception | **SR** | Spatial Reasoning | 48.50 | 48.00 | +0.50 |
| Perception | **MP** | Motion Perception | 55.00 | 33.50 | +21.50 |
| Generalization | **VG** | Visual Generalization | 64.00 | 59.50 | +4.50 |
| Generalization | **MG** | Motion Generalization | 80.00 | 65.00 | +15.00 |
| Generalization | **DR** | Disturbance Robustness | 19.00 | 26.50 | −7.50 |
| — | **Avg** | 九维平均 SR | **51.06** | **47.06** | **+4.00** |

### 4.1 路径长度与全局成功率

| 指标 | Ours | Paper (DynamicVLA) | Δ |
|------|------|---------------------|---|
| **Path Len**（m，九维 env 平均后再整体平均） | 1.91 | 2.50 | −0.59 |
| **Global trial SR** | **51.06%**（919 / 1800） | — | — |
| **Task Time**（Table I “Time”列） | 未从日志汇总 | 8.53 s | — |

> Table I 的 **Time** 为**任务完成时间**（物体开始运动→任务结束），**不是** policy chunk 推理时间（Appendix **I.Time ≈ 0.226 s @ A6000**）。本跑日志中逐步 `Inference Time` 约 **0.10–0.12 s**（RTX 4090），仅作参考。

---

## 5. 结果分析（按论文维度语义）

### 5.1 总体

- 九维平均 **51.06%**，较论文 **47.06%** 高约 **4.0 pp**；全局 trial 成功率同为 **51.06%**。
- 平均路径长度 **1.91 m**，低于论文 **2.50 m**（约 −24%），可能与成功轨迹更短或统计口径差异有关。

### 5.2 Interaction

| 维度 | Δ | 解读（对齐论文定义） |
|------|---|----------------------|
| **CR** | +4.0 | 对不同运动速度的闭环反应略优于论文 |
| **DA** | +17.5 | 对运动突变/扰动的适应明显优于论文报告值 |
| **LS** | −5.5 | 长时程多事件序列（`1-3_long-horizon_*`）仍偏弱 |

### 5.3 Perception

| 维度 | Δ | 解读 |
|------|---|------|
| **VU** | −14.0 | 相似外观物体区分不足，差距最大 |
| **SR** | +0.5 | 与论文基本持平 |
| **MP** | +21.5 | 对速度/方向等运动线索的利用明显优于论文（需结合具体场景复查） |

### 5.4 Generalization

| 维度 | Δ | 解读 |
|------|---|------|
| **VG** | +4.5 | 未见外观/场景（如 `*_99d_*` 泛化资产）略优 |
| **MG** | +15.0 | 未见运动模式（速度/摩擦/轨迹）泛化较好 |
| **DR** | −7.5 | 外扰/鲁棒性仍最弱之一；多环境 success_rate 接近 0 |

### 5.5 与论文趋势对照（定性）

论文强调动态操作的核心瓶颈是 **感知–执行时间对齐**（Continuous Inference、Latent-aware Action Streaming），而非可变形建模。本复现在 **DA / MP / MG** 上显著高于论文、在 **VU / DR / LS** 上低于论文，与「运动相关能力尚可、视觉区分与外扰鲁棒不足」的定性画像部分一致，但单次复现存在方差，不宜过度解读单项 ±20 pp 量级差异。

---

## 6. 复现检查清单

1. DOM 资源、`test-envs.txt`（90 条）、`dynamic-vla-DOM` 权重就绪。  
2. `-n 20`；先 `evaluate.py` 再 `inference.py`。  
3. 日志末尾：`Evaluation completed. Total tests run: 1800`。  
4. 按环境 ID 前缀 `1-1`…`3-3` 映射上表九维（**勿**使用旧版错误英文全称）。  
5. 指标以 `inference_table1.log` 中 `Test suite done with success rates` / `Test results` 为准。

```bash
tail -n 3 output/logs/inference_table1.log | grep "Evaluation completed"
```

---

## 7. 局限与注意事项

| 类别 | 说明 |
|------|------|
| **任务域** | DOM 为**刚体动态抓取/放置**，非可变形、非双臂协作等 |
| **硬件** | RTX 4090 vs 论文 A6000；推理时间与算力不可直接对比 |
| **随机性** | 物体初速、摩擦、场景采样带来 run-to-run 方差 |
| **日志** | `evaluate_table1.log` 无汇总；`output queue full` 可能影响极少数 step |
| **视频** | 本次无 `--save`，失败 case 无法目视复查 |

---

## 8. 参考文献

- Haozhe Xie et al., *DynamicVLA: A Vision-Language-Action Model for Dynamic Object Manipulation*, arXiv:2601.22153. Table I；Sec. IV-B（Benchmark Dimensions）；Sec. IV-C（仿真物体与 0–0.75 m/s 速度范围）。  
- 仓库：`simulations/evaluate.py`、`scripts/inference.py`、`Data/test-envs.txt`  
- 日志：`output/logs/inference_table1.log`

---

*报告修订：更正九维名称与论文 Sec. IV-B 一致，删除错误的 “deformable / multi-object placement” 等表述。*
