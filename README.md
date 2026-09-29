# SBFNav 在 CityNav 上的复现

本仓库用于复现论文 **Map the Possibilities: Spatial Belief Fields for Language-Goal Aerial Navigation** 中的 SBFNav。  
这是一个**基于论文描述的独立复现版本**，不是论文作者发布的官方实现。

## 论文与数据集

- SBFNav 论文（arXiv）：https://arxiv.org/abs/2609.05841
- SBFNav PDF：https://arxiv.org/pdf/2609.05841
- CityNav 项目主页：https://water-cookie.github.io/city-nav-proj/
- CityNav 论文：https://openaccess.thecvf.com/content/ICCV2025/html/Lee_CityNav_A_Large-Scale_Dataset_for_Real-World_Aerial_Navigation_ICCV_2025_paper.html

本复现使用 **Revised CityNav** 设置，并按照 CityNav/HETT 已发布评估代码采用二维平面距离作为主要 benchmark 指标（`benchmark_2d`）。仓库同时保留 3D 距离诊断结果，但不用于与论文表格直接比较。

## 当前复现结果

当前公开结果对应 **Epoch 7** checkpoint，在完整 Val-Seen 和 Val-Unseen split 上重新评估。

- Val-Seen：2,470 个 episode
- Val-Unseen：2,697 个 episode
- Seed：0
- GPU：NVIDIA GeForce RTX 5090
- Python：3.10.21
- PyTorch：2.9.1+cu128
- Transformers：4.57.6
- Epoch-7 checkpoint SHA-256：`ba8b48217ed948380ee8939df89b9eaeb8352e79f1bc5f9d0acc3a493e645564`

### 与论文 Revised SBFNav 的对比

| Split | 结果 | NE ↓ | SR ↑ | OSR ↑ | SPL ↑ |
|---|---|---:|---:|---:|---:|
| Val-Seen | 论文 | 32.50 | 43.24% | 53.77% | 40.93% |
| Val-Seen | 本复现 Epoch 7 | 34.00 | 40.61% | 58.34% | 34.06% |
| Val-Unseen | 论文 | 49.20 | 20.24% | 35.41% | 19.15% |
| Val-Unseen | 本复现 Epoch 7 | 49.79 | 19.84% | 37.37% | 15.25% |

对应差值（本复现 - 论文）：

| Split | ΔNE | ΔSR | ΔOSR | ΔSPL |
|---|---:|---:|---:|---:|
| Val-Seen | +1.50 m | -2.63 pp | +4.57 pp | -6.87 pp |
| Val-Unseen | +0.59 m | -0.40 pp | +1.96 pp | -3.90 pp |

从当前结果看，**Val-Unseen 的 NE 和 SR 已与论文非常接近**；OSR 略高，而 SPL 仍存在明显差距，说明当前策略能够较好地接近目标区域，但闭环轨迹效率仍弱于论文报告结果。

> 论文 Revised SBFNav 还报告了 Test-Unseen 结果，但本仓库当前不把 Test-Unseen 作为开发阶段调参依据，也不在这里混用未冻结的测试结果。

## 仓库内容

本仓库已经清理掉原工程中与 SBFNav 复现无关的 HETT、VLNCE、GSAM-LLaVA 等代码，仅保留 SBFNav 复现所需内容：

```text
.
├── sbfnav/                  # SBFNav 核心实现
├── configs/sbfnav/          # 主配置与消融配置
├── scripts/                 # 训练、评估、候选分析、可视化和监督运行脚本
├── tests/                   # SBFNav 单元测试
├── docs/                    # 复现假设、实现说明和完整实验记录
├── artifacts/sbfnav/        # Epoch 1-7 训练日志、评估日志、指标和可视化
├── requirements.txt
└── README.md
```

其中最重要的记录包括：

- `docs/sbfnav_reproduction.md`：论文到代码的实现对应关系，以及论文未公开细节的复现假设；
- `docs/sbfnav_assumptions.md`：所有未由论文明确给出的超参数与实现选择；
- `docs/sbfnav_results.md`：从数据检查、smoke test、NMS 修复到 Epoch 7 完整验证的全过程；
- `artifacts/sbfnav/formal_epochs_1_5/`：Epoch 1-5 正式训练日志；
- `artifacts/sbfnav/formal_epochs_6_7/`：Epoch 6-7 正式训练日志；
- `artifacts/sbfnav/epoch7_full_validation/`：完整 Val-Seen / Val-Unseen 指标与评估日志；
- `artifacts/sbfnav/epoch7_candidate_analysis/`：候选点覆盖率与 selector 分析；
- `artifacts/sbfnav/epoch7_case_visualizations/`：成功/失败案例可视化。

为了保持独立仓库体积可控，5 个体积较大的逐 episode prediction JSONL 没有迁入；对应的最终指标、训练/评估日志、provenance、配置和可视化均已保留。

## 方法概览

复现实现遵循论文公开的核心流程：

1. 将历史观测构造成世界坐标对齐的 9 通道 planning state；
2. 使用语言条件化的 **Spatial Belief Field (SBF)** 预测目标位置分布；
3. 从 belief field 中通过 NMS 产生候选位置；
4. 使用局部视觉、完整指令、地标文本和几何关系对候选点重新排序；
5. 预测高度，并通过 receding-horizon 闭环方式执行导航；
6. 每一步重新利用新观测更新 spatial belief 和候选决策。

论文没有公开完整 Supplement 中的若干训练/控制参数，因此本复现将这些选择全部显式写入 YAML，并在 `docs/sbfnav_assumptions.md` 中记录。

## 环境

正式复现实验使用：

```text
Python 3.10.21
PyTorch 2.9.1+cu128
CUDA 12.8
Transformers 4.57.6
NVIDIA GeForce RTX 5090
```

建议先根据本机 CUDA 安装合适的 PyTorch，然后安装其余依赖：

```bash
pip install -r requirements.txt
```

SigLIP 主配置使用：

```text
google/siglip-base-patch16-224
```

主实验默认离线加载已缓存的模型权重，因此首次运行前需要提前准备 Hugging Face 模型缓存。

## 数据准备

下载 CityNav 数据：

```bash
bash scripts/download_data.sh
```

也可以直接使用已有数据目录，并在运行时覆盖 `data.root`：

```bash
--override data.root=/path/to/citynav/data
```

数据目录至少需要包含：

```text
cityrefer/
processed_citynav/
rgbd/
```

## 训练

直接运行主配置：

```bash
python scripts/train_sbfnav.py \
  --config configs/sbfnav/main.yaml \
  --output-dir runs/sbfnav_main \
  --override data.root=/path/to/citynav/data
```

为了同时保存运行 provenance、GPU 状态、配置快照和日志，推荐使用监督脚本：

```bash
python scripts/supervise_experiment.py \
  --run-dir runs/sbfnav_main \
  --python "$(which python)" \
  --config configs/sbfnav/main.yaml \
  --phase train-eval \
  --config-override data.root=/path/to/citynav/data
```

## 评估

```bash
python scripts/evaluate_sbfnav.py \
  --config configs/sbfnav/main.yaml \
  --checkpoint /path/to/checkpoint.pt \
  --split val_unseen \
  --output-dir runs/eval_val_unseen \
  --override data.root=/path/to/citynav/data
```

候选点覆盖率分析：

```bash
python scripts/analyze_sbf_candidates.py \
  --config configs/sbfnav/main.yaml \
  --checkpoint /path/to/checkpoint.pt \
  --split val_unseen \
  --output-dir runs/candidate_analysis \
  --prefix-fraction 0.5 \
  --override data.root=/path/to/citynav/data
```

## 测试

```bash
python -m pytest tests/test_sbf_*.py
```

## 关于复现边界

论文公开版本未给出完整的 epoch、batch size、优化器参数、学习率调度、部分 loss 超参数、controller 细节等信息。因此，本仓库的目标是：

- 严格实现论文明确公开的结构与约束；
- 对未公开细节给出可审计、可复现的明确假设；
- 保留训练日志、评估日志、配置、哈希和失败案例；
- 不把该实现描述为作者官方代码，也不把尚未达到的指标表述为完全复现。

详细过程请查看 `docs/sbfnav_results.md`。
