# 运行说明

环境：使用已配置好的 conda 环境 `pointnet`。

## 构建缓存

```powershell
conda run -n pointnet python -m pointnet_final.prepare_cache
```

## 训练模型

第一阶段训练 DGCNN 基线：

```powershell
conda run -n pointnet python -m pointnet_final.train --run-dir runs/dgcnn_normals_seed1 --epochs 250 --batch-size 24 --eval-batch-size 32 --num-points 1024 --workers 4 --eval-votes 1 --final-votes 10 --seed 1
```

第二阶段类别均衡微调：

```powershell
conda run -n pointnet python -m pointnet_final.train --run-dir runs/dgcnn_normals_balanced_ft_seed1 --epochs 80 --batch-size 24 --eval-batch-size 32 --num-points 1024 --workers 4 --eval-votes 1 --final-votes 10 --seed 2 --resume runs/dgcnn_normals_seed1/best.pt --resume-model-only --lr 0.02 --min-lr 0.00001 --label-smoothing 0.1 --balanced-sampler --class-weight-power 0.3
```

## 现场推理

推荐使用统一入口脚本。最高准确率模式会加载四个权重进行加权集成，并执行 3 次随机采样投票：

```powershell
.\run_inference.ps1 -TestRoot "<测试集目录>" -Output "<赛道1-组员1姓名学号-组员2姓名学号-组员3姓名学号.csv>" -Mode max
```

可选模式如下：

- `max`：四权重加权集成，推荐用于最终提交
- `stable`：三权重集成，运行时间更短
- `fast`：单模型单次采样，用于快速检查输入和输出
- `legacy`：原始单模型多票方案，用于回退

最高准确率模式的完整命令为：

```powershell
conda run -n pointnet python -m pointnet_final.predict_advanced --test-root <测试集目录> --cache-name onsite_test_advanced --checkpoints runs/dgcnn_normals_seed1/best.pt runs/dgcnn_normals_balanced_ft_seed1/best.pt runs/dgcnn_refine_seed3/best.pt runs/dgcnn_refine_seed3/last.pt --model-weights 0.42 0.38 0.10 0.10 --output <赛道1-组员1姓名学号-组员2姓名学号-组员3姓名学号.csv> --votes 3 --sampling random --batch-size 20 --workers 4 --force-cache
```

测试目录既可以是直接存放 `.txt` 文件的目录，也可以是按 40 个类别划分子目录的目录。每个点云文件支持英文逗号或空白分隔，读取前六列 `x,y,z,nx,ny,nz`。

## 提交文件

- 完整代码：`pointnet_final/`
- 一页设计思路：`docs/brief_design.md`
- 运行说明：`docs/run_instructions.md`
- 推荐权重：`runs/dgcnn_normals_seed1/best.pt`、`runs/dgcnn_normals_balanced_ft_seed1/best.pt`、`runs/dgcnn_refine_seed3/best.pt`、`runs/dgcnn_refine_seed3/last.pt`
- 预测结果：按验收要求命名的 `.csv`
