# 实验结果记录

本文件记录本地实验中保留的几种推理方案，便于选择现场验收时的默认命令。

| 方案 | 权重 | Votes | Instance Accuracy | Class Accuracy |
| --- | --- | ---: | ---: | ---: |
| 单模型多票 | `runs/dgcnn_normals_balanced_ft_seed1/best_class.pt` | 20 | 92.22% | 91.10% |
| 均衡微调整体最优 | `runs/dgcnn_normals_balanced_ft_seed1/best.pt` | 10 | 92.14% | 90.02% |
| 均衡微调类均值最优 | `runs/dgcnn_normals_balanced_ft_seed1/best_class.pt` | 10 | 92.14% | 90.85% |
| 三权重集成 | `runs/dgcnn_normals_seed1/best.pt` + balanced `best.pt` + balanced `best_class.pt` | 10 | 92.67% | 90.45% |
| 三权重集成 | `runs/dgcnn_normals_seed1/best.pt` + balanced `best.pt` + balanced `best_class.pt` | 20 | 92.83% | 90.56% |
| 精炼单模型 | `runs/dgcnn_refine_seed3/best.pt` | 1 | 93.11% | 90.66% |
| 高级三模型加权 | baseline `best.pt` + balanced `best.pt` + refine `best.pt` | 3 | 93.35% | 90.88% |
| 最高准确率四模型加权 | baseline `best.pt` + balanced `best.pt` + refine `best.pt` + refine `last.pt` | 3 | 93.40% | 90.82% |

推荐优先级：

1. 默认推荐：最高准确率四模型加权，约 137 秒。
2. 稳健备用：高级三模型加权，约 105 秒。
3. 最小依赖：精炼单模型，单次推理即可超过两项加分线。
4. 原版回退：旧单模型多票和旧三权重集成仍完整保留。
