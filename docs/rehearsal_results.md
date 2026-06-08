# 预演结果

预演集：`modelnet40_normal_resampled/modelnet40_test.txt`，共 2468 个样本。老师训练集已核对为官方 `modelnet40_train.txt`，与该预演集无重叠。

| 方案 | 权重 | Votes | Instance Accuracy | Class Accuracy |
| --- | --- | ---: | ---: | ---: |
| 推荐提交 | `runs/dgcnn_normals_balanced_ft_seed1/best_class.pt` | 20 | 92.22% | 91.10% |
| 均衡微调整体最优 | `runs/dgcnn_normals_balanced_ft_seed1/best.pt` | 10 | 92.14% | 90.02% |
| 均衡微调类均值最优 | `runs/dgcnn_normals_balanced_ft_seed1/best_class.pt` | 10 | 92.14% | 90.85% |
| 三权重集成 | `runs/dgcnn_normals_seed1/best.pt` + balanced `best.pt` + balanced `best_class.pt` | 10 | 92.67% | 90.45% |
| 三权重集成 | `runs/dgcnn_normals_seed1/best.pt` + balanced `best.pt` + balanced `best_class.pt` | 20 | 92.83% | 90.56% |

推荐 CSV：`runs/rehearsal_submission_fullscore.csv`。

最高整体准确率 CSV：`runs/rehearsal_submission_v2_ensemble3_20votes.csv`。
