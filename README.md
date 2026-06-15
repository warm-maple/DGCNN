# ModelNet40 点云分类项目

本项目用于完成 ModelNet40 点云分类任务。输入点云格式为 `x, y, z, nx, ny, nz`，模型根据点云预测三维物体所属类别，并输出符合要求的 `id,类别` CSV 文件。

## 项目结构

- `pointnet_final/`：训练、评估、推理代码
- `docs/`：设计思路、运行说明和实验结果记录
- `runs/dgcnn_normals_seed1/`：第一阶段 DGCNN 基线权重
- `runs/dgcnn_normals_balanced_ft_seed1/`：类别均衡微调权重
- `runs/dgcnn_refine_seed3/`：低学习率精炼权重

## 方法概述

模型采用 DGCNN 分类网络，同时使用点坐标和法向量信息。训练阶段对点云进行归一化、固定点数采样和随机增强；第二阶段使用类别均衡采样进行微调，以提升类别平均准确率。推理阶段使用多次采样投票，降低单次采样带来的波动。

当前保留两套方案：

- 单模型多票：速度较快，类别平均准确率余量更大
- 三权重集成：整体准确率更高，适合现场时间充足时使用

## 环境

使用已配置好的 conda 环境：

```powershell
conda activate pointnet
```

也可以直接用 `conda run -n pointnet ...` 执行命令。

## 现场推理

如果下发的是测试目录：

```powershell
conda run -n pointnet python -m pointnet_final.predict --test-root <测试集目录> --cache-name onsite_test --checkpoints runs/dgcnn_normals_balanced_ft_seed1/best_class.pt --output <赛道1-组员姓名学号.csv> --votes 20 --batch-size 32 --workers 4 --force-cache
```

如果下发的是测试 id 列表：

```powershell
conda run -n pointnet python -m pointnet_final.predict --test-list <测试id列表.txt> --cache-name onsite_test --checkpoints runs/dgcnn_normals_balanced_ft_seed1/best_class.pt --output <赛道1-组员姓名学号.csv> --votes 20 --batch-size 32 --workers 4 --force-cache
```

时间充足时可使用三权重集成：

```powershell
conda run -n pointnet python -m pointnet_final.predict --test-root <测试集目录> --cache-name onsite_test --checkpoints runs/dgcnn_normals_seed1/best.pt runs/dgcnn_normals_balanced_ft_seed1/best.pt runs/dgcnn_normals_balanced_ft_seed1/best_class.pt --output <赛道1-组员姓名学号.csv> --votes 20 --batch-size 24 --workers 4 --force-cache
```

## 推荐高级推理

最高准确率方案使用四个轻量权重、3 次采样投票，项目测试中耗时约 137 秒：

```powershell
conda run -n pointnet python -m pointnet_final.predict_advanced --test-root <测试集目录> --cache-name onsite_test_advanced --checkpoints runs/dgcnn_normals_seed1/best.pt runs/dgcnn_normals_balanced_ft_seed1/best.pt runs/dgcnn_refine_seed3/best.pt runs/dgcnn_refine_seed3/last.pt --model-weights 0.42 0.38 0.10 0.10 --output <赛道1-组员姓名学号.csv> --votes 3 --sampling random --batch-size 20 --workers 4 --force-cache
```

更简洁的三模型备用方案耗时约 105 秒：

```powershell
conda run -n pointnet python -m pointnet_final.predict_advanced --test-root <测试集目录> --cache-name onsite_test_advanced --checkpoints runs/dgcnn_normals_seed1/best.pt runs/dgcnn_normals_balanced_ft_seed1/best.pt runs/dgcnn_refine_seed3/best.pt --model-weights 0.34 0.36 0.30 --output <赛道1-组员姓名学号.csv> --votes 3 --sampling random --batch-size 24 --workers 4 --force-cache
```

原有 `pointnet_final/predict.py` 和全部旧权重均未修改，可随时回退。

也可以使用封装好的现场脚本：

```powershell
.\run_inference.ps1 -TestRoot "<测试集目录>" -Output "<赛道1-组员姓名学号.csv>" -Mode max
```

`Mode` 可选：

- `max`：最高准确率方案
- `stable`：三模型稳健方案
- `fast`：单模型快速方案
- `legacy`：原版回退方案
