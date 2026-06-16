# ModelNet40 点云分类项目

本项目用于完成 ModelNet40 点云分类任务。输入点云格式为 `x, y, z, nx, ny, nz`，模型根据点云预测三维物体所属类别，并输出符合要求的 `id,类别` CSV 文件。

## 项目结构

- `pointnet_final/`：训练、评估、推理代码
- `docs/`：设计思路、运行说明和实验结果记录
- `runs/clean_stage1_seed2026/`：严格验证集选模的第一阶段权重
- `runs/clean_stage2_balanced_seed2026/`：严格验证集选模的第二阶段类别均衡微调权重
- `runs/clean_validation_selection/`：只基于训练集内部验证集得到的推理配置选择记录
- `runs/clean_official_evaluation/`：配置冻结后的官方测试集最终评估记录

## 方法概述

模型采用 DGCNN 分类网络，同时使用点坐标和法向量信息。训练阶段对点云进行归一化、固定点数采样和随机增强；第二阶段从第一阶段验证集最优权重继续微调，加入类别均衡采样和类别加权损失，以提升 Class Accuracy。

本版本采用严格实验流程：从老师训练集内部按类别分层划分训练集和验证集，`best.pt`、`best_class.pt`、`best_balanced.pt` 只依据验证集保存。官方测试集不参与训练、权重选择或推理参数选择，只在最终配置冻结后评估一次。

## 环境

推荐使用已有 Conda 环境 `pointnet`：

```powershell
conda activate pointnet
python -m pip install -r requirements.txt
```

已验证环境：

- Python 3.11.15
- PyTorch 2.11.0 + CUDA 12.8
- NumPy 2.4.4
- Windows 11 + NVIDIA GPU

检查 GPU：

```powershell
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## 现场推理

推荐使用统一入口。`max` 模式加载第一阶段和第二阶段由验证集选择的权重，按 0.5/0.5 加权集成，并执行 7 次随机采样投票：

```powershell
.\run_inference.ps1 -TestRoot "<测试集目录>" -Output "<赛道1-组员1姓名学号-组员2姓名学号-组员3姓名学号.csv>" -Mode max
```

`Mode` 可选：

- `max`：两阶段权重 0.5/0.5 加权集成，推荐用于最终提交
- `stable`：第一阶段单模型 3 票，速度更快
- `fast`：第一阶段单模型 1 票，用于快速检查输入和输出
- `legacy`：第二阶段单模型 3 票，用于对比回退

完整命令：

```powershell
conda run -n pointnet python -m pointnet_final.predict_advanced --test-root <测试集目录> --cache-name onsite_test_advanced --checkpoints runs/clean_stage1_seed2026/best.pt runs/clean_stage2_balanced_seed2026/best.pt --model-weights 0.5 0.5 --output <赛道1-组员姓名学号.csv> --votes 7 --sampling random --batch-size 24 --workers 4 --force-cache
```

本次严格实验的最终冻结配置在官方测试集上的一次性评估结果为：

- Instance Accuracy：92.59%
- Class Accuracy：90.62%

更完整的训练、推理和提交说明见 `docs/run_instructions.md`。
