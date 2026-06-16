# 高级推理说明

高级推理代码位于 `pointnet_final/predict_advanced.py`。它支持多 checkpoint 加权集成、多次采样投票、固定随机 seed、CUDA 混合精度以及推理指标记录。

## 推荐方案

推荐现场使用两阶段验证集最优权重 0.5/0.5 加权集成，并固定 seed `2026` 进行 7 次随机采样投票：

```powershell
conda run -n pointnet python -m pointnet_final.predict_advanced --test-root <测试集目录> --cache-name onsite_test_advanced --checkpoints runs/clean_stage1_seed2026/best.pt runs/clean_stage2_balanced_seed2026/best.pt --model-weights 0.5 0.5 --output <赛道1-组员姓名学号.csv> --votes 7 --sampling random --seed 2026 --batch-size 24 --workers 4 --force-cache
```

本方案对应 `run_inference.ps1` 的 `max` 模式：

```powershell
.\run_inference.ps1 -TestRoot "<测试集目录>" -Output "<赛道1-组员姓名学号.csv>" -Mode max
```

## 备用方案

第一阶段单模型 3 票：

```powershell
conda run -n pointnet python -m pointnet_final.predict_advanced --test-root <测试集目录> --cache-name onsite_test_stage1 --checkpoints runs/clean_stage1_seed2026/best.pt --output <赛道1-组员姓名学号.csv> --votes 3 --sampling random --seed 2026 --batch-size 32 --workers 4 --force-cache
```

第一阶段单模型 1 票快速检查：

```powershell
conda run -n pointnet python -m pointnet_final.predict_advanced --test-root <测试集目录> --cache-name onsite_test_fast --checkpoints runs/clean_stage1_seed2026/best.pt --output <赛道1-组员姓名学号.csv> --votes 1 --sampling random --seed 2026 --batch-size 64 --workers 4 --force-cache
```

第二阶段单模型 3 票对比回退：

```powershell
conda run -n pointnet python -m pointnet_final.predict_advanced --test-root <测试集目录> --cache-name onsite_test_stage2 --checkpoints runs/clean_stage2_balanced_seed2026/best.pt --output <赛道1-组员姓名学号.csv> --votes 3 --sampling random --seed 2026 --batch-size 32 --workers 4 --force-cache
```

## 结果记录

推荐方案在训练集内部验证集上为 Instance Accuracy 93.81%、Class Accuracy 91.81%。固定配置后在官方测试集上评估为 Instance Accuracy 92.59%、Class Accuracy 90.62%。官方测试集不参与训练或 checkpoint 选择。
