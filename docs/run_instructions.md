# 运行说明

环境：使用已配置好的 conda 环境 `pointnet`。

构建缓存：

```powershell
conda run -n pointnet python -m pointnet_final.prepare_cache
```

训练模型：

```powershell
conda run -n pointnet python -m pointnet_final.train --run-dir runs/dgcnn_normals_seed1 --epochs 250 --batch-size 24 --eval-batch-size 32 --num-points 1024 --workers 4 --eval-votes 1 --final-votes 10 --seed 1
```

类别均衡微调：

```powershell
conda run -n pointnet python -m pointnet_final.train --run-dir runs/dgcnn_normals_balanced_ft_seed1 --epochs 80 --batch-size 24 --eval-batch-size 32 --num-points 1024 --workers 4 --eval-votes 1 --final-votes 10 --seed 2 --resume runs/dgcnn_normals_seed1/best.pt --resume-model-only --lr 0.02 --min-lr 0.00001 --label-smoothing 0.1 --balanced-sampler --class-weight-power 0.3
```

预演评估并生成 CSV：

```powershell
conda run -n pointnet python -m pointnet_final.predict --test-list modelnet40_normal_resampled/modelnet40_test.txt --cache-name rehearsal_test --checkpoints runs/dgcnn_normals_balanced_ft_seed1/best_class.pt --output runs/rehearsal_submission.csv --votes 20
```

当前预演结果：Instance Accuracy `92.22%`，Class Accuracy `91.10%`。

若现场推理时间充足，也可使用三权重集成，当前预演结果为 Instance Accuracy `92.83%`，Class Accuracy `90.56%`：

```powershell
conda run -n pointnet python -m pointnet_final.predict --test-root <测试集目录> --cache-name onsite_test --checkpoints runs/dgcnn_normals_seed1/best.pt runs/dgcnn_normals_balanced_ft_seed1/best.pt runs/dgcnn_normals_balanced_ft_seed1/best_class.pt --output <赛道1-组员姓名学号.csv> --votes 20 --batch-size 24 --workers 4 --force-cache
```

现场验收时，如果老师下发的是测试目录：

```powershell
conda run -n pointnet python -m pointnet_final.predict --test-root <测试集目录> --cache-name onsite_test --checkpoints runs/dgcnn_normals_balanced_ft_seed1/best_class.pt --output <赛道1-组员姓名学号.csv> --votes 20 --force-cache
```

如果老师下发的是测试 id 列表，并且点云仍按 ModelNet40 类别目录存放：

```powershell
conda run -n pointnet python -m pointnet_final.predict --test-list <测试id列表.txt> --cache-name onsite_test --checkpoints runs/dgcnn_normals_balanced_ft_seed1/best_class.pt --output <赛道1-组员姓名学号.csv> --votes 20 --force-cache
```

主要提交文件：

- 完整代码：`pointnet_final/`
- 设计思路：`docs/design.md`
- 运行说明：`docs/run_instructions.md`
- 训练权重：`runs/dgcnn_normals_balanced_ft_seed1/best_class.pt`
- 预测结果：按验收要求命名的 `.csv`
