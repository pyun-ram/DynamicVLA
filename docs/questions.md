DONE 1. 评估的时候 CR/DA/LS/VU/SR/MP/VG/MG/DR 是怎么评估的？
DONE 2. 在线推理的时候如何进行的异步推理：仿真/模型推理异步进行
DONE 3. 模型结构是什么样子的？哪些部分是 pretrained？输入是什么？输出是什么？形状是什么？做了归一化吗？
DONE 4. Loss Function 是什么样子的？
DONE 5. 如何进行数据采集的？数据结构是什么样子的？
## Q1: 训练集物体速度范围的矛盾
==========
论文正文 Sec IV-C 写 "Object speeds are sampled from **0–0.75 m/s**"，
但 Figure 3 的 Objects and Dynamics 框里写的是 "Speed: **0–1 m/s**"。

`simulations/configs/sim_cfg.yaml` 中 `moving_speed: [0.15, 0.75]` 与正文一致。
CR 测试集实测最高速 0.726 m/s，也支持 0.75 上界。

**待确认**：Figure 3 的 0–1 m/s 是笔误/图示泛化标注，还是训练时真实用过 >0.75 m/s 的数据？
==========
## TODO: 解决 texture 渲染问题
