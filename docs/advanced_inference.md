# 高级推理说明

高级推理代码位于 `pointnet_final/predict_advanced.py`。它是新增文件，不会修改或替换原有 `pointnet_final/predict.py`。

## 最高准确率方案

使用四个权重进行加权集成，并对每个样本随机采样 3 次：

```powershell
conda run -n pointnet python -m pointnet_final.predict_advanced --test-root <测试集目录> --cache-name onsite_test_advanced --checkpoints runs/dgcnn_normals_seed1/best.pt runs/dgcnn_normals_balanced_ft_seed1/best.pt runs/dgcnn_refine_seed3/best.pt runs/dgcnn_refine_seed3/last.pt --model-weights 0.42 0.38 0.10 0.10 --output <赛道1-组员姓名学号.csv> --votes 3 --sampling random --batch-size 20 --workers 4 --force-cache
```

本地实验统计：

- Instance Accuracy：93.40%
- Class Accuracy：90.82%
- 2468 个样本推理时间：约 137 秒

## 三模型备用方案

```powershell
conda run -n pointnet python -m pointnet_final.predict_advanced --test-root <测试集目录> --cache-name onsite_test_advanced --checkpoints runs/dgcnn_normals_seed1/best.pt runs/dgcnn_normals_balanced_ft_seed1/best.pt runs/dgcnn_refine_seed3/best.pt --model-weights 0.34 0.36 0.30 --output <赛道1-组员姓名学号.csv> --votes 3 --sampling random --batch-size 24 --workers 4 --force-cache
```

本地实验统计：

- Instance Accuracy：93.35%
- Class Accuracy：90.88%
- 2468 个样本推理时间：约 105 秒

## 快速单模型方案

只加载精炼模型：

```powershell
conda run -n pointnet python -m pointnet_final.predict_advanced --test-root <测试集目录> --cache-name onsite_test_advanced --checkpoints runs/dgcnn_refine_seed3/best.pt --output <赛道1-组员姓名学号.csv> --votes 1 --sampling random --batch-size 64 --workers 4 --force-cache
```

该方案依赖最少、速度最快，适合作为现场故障回退。

## 回退保障

- 原推理代码：`pointnet_final/predict.py`
- 原推荐权重：`runs/dgcnn_normals_balanced_ft_seed1/best_class.pt`
- Git 保护分支：`baseline-passing-20260615`
- Git 保护标签：`v3-baseline-locked-20260615`
- 本地完整快照：`snapshots/baseline_passing_20260615.tar.gz`

