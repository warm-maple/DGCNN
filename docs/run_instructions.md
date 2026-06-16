# 运行说明

环境：使用已配置好的 Conda 环境 `pointnet`。

## 数据目录

训练数据目录：

```text
F:\Python Project\pointnet\dataset\train
```

训练代码会从该目录中按类别分层划分训练子集和验证子集。

## 训练模型

第一阶段训练 DGCNN 基线：

```powershell
conda run -n pointnet python -m pointnet_final.train --cache-dir cache/modelnet40_clean --run-dir runs/clean_stage1_seed2026 --epochs 200 --batch-size 24 --eval-batch-size 32 --num-points 1024 --points-per-shape 10000 --workers 4 --eval-votes 1 --final-votes 10 --seed 2026 --split-seed 2026 --save-every 10 --top-k-checkpoints 5
```

第二阶段类别均衡微调，复用第一阶段完全相同的训练/验证划分：

```powershell
conda run -n pointnet python -m pointnet_final.train --cache-dir cache/modelnet40_clean --run-dir runs/clean_stage2_balanced_seed2026 --epochs 80 --batch-size 24 --eval-batch-size 32 --num-points 1024 --points-per-shape 10000 --workers 4 --eval-votes 1 --final-votes 10 --seed 2027 --split-seed 2026 --split-from runs/clean_stage1_seed2026 --save-every 10 --top-k-checkpoints 5 --resume runs/clean_stage1_seed2026/best.pt --resume-model-only --lr 0.02 --min-lr 0.00001 --label-smoothing 0.1 --balanced-sampler --class-weight-power 0.3
```

训练会保留：

- `best.pt`
- `best_class.pt`
- `best_balanced.pt`
- `last.pt`
- 每 10 轮 `epoch_XXX.pt`
- 验证集综合分前 5 的 `top_balanced_epoch_XXX.pt`

## 现场推理

推荐使用统一入口脚本：

```powershell
.\run_inference.ps1 -TestRoot "<测试集目录>" -Output "<赛道1-组员1姓名学号-组员2姓名学号-组员3姓名学号.csv>" -Mode max
```

可选模式如下：

- `max`：两阶段权重 0.5/0.5 加权集成，7 票随机采样，推荐用于最终提交
- `stable`：第一阶段单模型 3 票，运行时间更短
- `fast`：第一阶段单模型 1 票，用于快速检查输入和输出
- `legacy`：第二阶段单模型 3 票，用于对比回退

完整命令为：

```powershell
conda run -n pointnet python -m pointnet_final.predict_advanced --test-root <测试集目录> --cache-name onsite_test_advanced --checkpoints runs/clean_stage1_seed2026/best.pt runs/clean_stage2_balanced_seed2026/best.pt --model-weights 0.5 0.5 --output <赛道1-组员1姓名学号-组员2姓名学号-组员3姓名学号.csv> --votes 7 --sampling random --seed 2026 --batch-size 24 --workers 4 --force-cache
```

测试目录既可以是直接存放 `.txt` 文件的目录，也可以是按 40 个类别划分子目录的目录。每个点云文件支持英文逗号或空白分隔，读取前六列 `x,y,z,nx,ny,nz`。

## 提交文件

- 完整代码：`pointnet_final/`
- 一页设计思路：`docs/brief_design.md`
- 运行说明：`docs/run_instructions.md`
- 推荐权重：`runs/clean_stage1_seed2026/best.pt`、`runs/clean_stage2_balanced_seed2026/best.pt`
- 预测结果：按验收要求命名的 `.csv`
