# ModelNet40 点云分类项目

本项目用于完成 ModelNet40 点云分类任务。输入点云文件每行包含 `x,y,z,nx,ny,nz` 六列，模型输出每个样本对应的预测类别，并生成符合提交要求的 CSV 文件。

## 项目结构

- `pointnet_final/`：训练、数据处理、模型和推理代码。
- `docs/brief_design.md`：一页版设计思路。
- `docs/run_instructions.md`：训练和推理运行说明。
- `runs/clean_stage1_seed2026/`：第一阶段模型权重。
- `runs/clean_stage2_balanced_seed2026/`：第二阶段模型权重。
- `run_inference.ps1`：现场推理统一入口。
- `requirements.txt`：Python 依赖说明。

## 环境安装

推荐使用 Conda 环境：

```powershell
conda create -n pointnet python=3.11 -y
conda activate pointnet
python -m pip install -r requirements.txt
```

如果机器已存在 `pointnet` 环境，直接激活并安装依赖即可：

```powershell
conda activate pointnet
python -m pip install -r requirements.txt
```

检查 GPU：

```powershell
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## 推理命令

测试目录可以是直接存放 `.txt` 点云文件的目录，例如：

```text
test_data/
  test_000001.txt
  test_000002.txt
  ...
```

推荐使用 `max` 模式：

```powershell
.\run_inference.ps1 -TestRoot "<测试集目录>" -Output "<赛道1-组员1姓名学号-组员2姓名学号-组员3姓名学号.csv>" -Mode max
```

例如：

```powershell
.\run_inference.ps1 -TestRoot "F:\Python Project\DGCNN\test\test_data" -Output "赛道1-李良顺-2023210984-周柄名-2023210992-唐振桓-2023210990.csv" -Mode max
```

输出 CSV 无表头，每行格式为：

```csv
test_000001,airplane
test_000002,chair
```

更多训练和推理命令见 `docs/run_instructions.md`。
