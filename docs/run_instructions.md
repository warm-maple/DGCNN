# 运行说明

## 1. 安装环境

推荐使用 Conda：

```powershell
conda create -n pointnet python=3.11 -y
conda activate pointnet
python -m pip install -r requirements.txt
```

如果已经配置好 `pointnet` 环境：

```powershell
conda activate pointnet
python -m pip install -r requirements.txt
```

检查 CUDA 是否可用：

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## 2. 数据格式

训练数据按类别目录存放：

```text
dataset/train/
  airplane/
    airplane_0001.txt
  chair/
    chair_0001.txt
```

测试数据可以直接放在一个目录下：

```text
test_data/
  test_000001.txt
  test_000002.txt
```

每个 `.txt` 文件每行至少包含三列坐标，推荐六列：

```text
x,y,z,nx,ny,nz
```

程序支持英文逗号或空白分隔；若只有三列坐标，法向量会自动补零。

## 3. 直接推理

推荐命令：

```powershell
.\run_inference.ps1 -TestRoot "<测试集目录>" -Output "<赛道1-组员1姓名学号-组员2姓名学号-组员3姓名学号.csv>" -Mode max
```

四种模式如下：

- `max`：两阶段权重加权集成，7 次采样投票，推荐最终提交。
- `stable`：第一阶段单模型，3 次采样投票。
- `fast`：第一阶段单模型，1 次采样投票，用于快速检查输入输出。
- `legacy`：第二阶段单模型，3 次采样投票。

等价的完整命令：

```powershell
conda run -n pointnet python -m pointnet_final.predict_advanced --test-root <测试集目录> --cache-name onsite_test_advanced --checkpoints runs/clean_stage1_seed2026/best.pt runs/clean_stage2_balanced_seed2026/best.pt --model-weights 0.5 0.5 --output <赛道1-组员1姓名学号-组员2姓名学号-组员3姓名学号.csv> --votes 7 --sampling random --seed 2026 --batch-size 24 --workers 4 --force-cache
```

输出 CSV 无表头，每行两列：

```csv
test_000001,airplane
test_000002,chair
```

## 4. 训练命令

第一阶段训练：

```powershell
conda run -n pointnet python -m pointnet_final.train --data-root data/modelnet40 --train-root dataset/train --cache-dir cache/modelnet40_clean --run-dir runs/clean_stage1_seed2026 --epochs 200 --batch-size 24 --eval-batch-size 32 --num-points 1024 --points-per-shape 10000 --workers 4 --eval-votes 1 --final-votes 10 --seed 2026 --split-seed 2026 --save-every 10 --top-k-checkpoints 5
```

第二阶段微调：

```powershell
conda run -n pointnet python -m pointnet_final.train --data-root data/modelnet40 --train-root dataset/train --cache-dir cache/modelnet40_clean --run-dir runs/clean_stage2_balanced_seed2026 --epochs 80 --batch-size 24 --eval-batch-size 32 --num-points 1024 --points-per-shape 10000 --workers 4 --eval-votes 1 --final-votes 10 --seed 2027 --split-seed 2026 --split-from runs/clean_stage1_seed2026 --save-every 10 --top-k-checkpoints 5 --resume runs/clean_stage1_seed2026/best.pt --resume-model-only --lr 0.02 --min-lr 0.00001 --label-smoothing 0.1 --balanced-sampler --class-weight-power 0.3
```

训练会保存 `best.pt`、`best_class.pt`、`best_balanced.pt`、`last.pt`、周期 checkpoint 以及综合指标较好的 checkpoint。

## 5. 提交材料

建议提交：

- `pointnet_final/`
- `runs/clean_stage1_seed2026/`
- `runs/clean_stage2_balanced_seed2026/`
- `run_inference.ps1`
- `requirements.txt`
- `README.md`
- `docs/brief_design.md`
- `docs/run_instructions.md`
- 按要求命名的预测 CSV
