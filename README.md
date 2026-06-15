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

推荐使用 Python 3.10 或 3.11。项目直接依赖较少，核心依赖为 PyTorch 和 NumPy。

本项目已验证的环境为：

- Python 3.11.15
- PyTorch 2.11.0 + CUDA 12.8
- NumPy 2.4.4
- Windows 11 + NVIDIA GPU

### 使用现有环境

如果机器上已经存在 `pointnet` Conda 环境：

```powershell
conda activate pointnet
python -m pip install -r requirements.txt
```

也可以直接用 `conda run -n pointnet ...` 执行命令。

### 创建新环境

```powershell
conda create -n pointnet python=3.11 -y
conda activate pointnet
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` 中使用兼容版本范围，便于在不同机器上安装。若需要 NVIDIA GPU 加速，建议根据机器驱动和 CUDA 环境，从 [PyTorch 官方安装页面](https://pytorch.org/get-started/locally/)选择对应的 CUDA 安装命令，再安装其他依赖。例如 CUDA 12.8：

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install numpy
```

安装完成后检查 GPU：

```powershell
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

当输出 `CUDA available: True` 时，训练和高级推理会自动使用 GPU；没有 CUDA 时仍可运行，但速度会明显下降。

## 现场推理

推荐直接使用统一入口。最高准确率模式加载四个权重进行加权集成，并执行 3 次随机采样投票：

```powershell
.\run_inference.ps1 -TestRoot "<测试集目录>" -Output "<赛道1-组员1姓名学号-组员2姓名学号-组员3姓名学号.csv>" -Mode max
```

`Mode` 可选：

- `max`：四权重加权集成，推荐用于最终提交
- `stable`：三权重集成，速度更快
- `fast`：单模型单次采样，用于快速检查
- `legacy`：原始单模型多票方案

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

更完整的训练、推理和提交说明见 `docs/run_instructions.md`。
