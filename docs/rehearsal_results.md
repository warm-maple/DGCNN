# 实验结果记录

本文件记录本地实验中保留的几种推理方案，便于选择现场验收时的默认命令。

| 方案 | 权重 | Votes | Instance Accuracy | Class Accuracy |
| --- | --- | ---: | ---: | ---: |
| 单模型多票 | `runs/dgcnn_normals_balanced_ft_seed1/best_class.pt` | 20 | 92.22% | 91.10% |
| 均衡微调整体最优 | `runs/dgcnn_normals_balanced_ft_seed1/best.pt` | 10 | 92.14% | 90.02% |
| 均衡微调类均值最优 | `runs/dgcnn_normals_balanced_ft_seed1/best_class.pt` | 10 | 92.14% | 90.85% |
| 三权重集成 | `runs/dgcnn_normals_seed1/best.pt` + balanced `best.pt` + balanced `best_class.pt` | 10 | 92.67% | 90.45% |
| 三权重集成 | `runs/dgcnn_normals_seed1/best.pt` + balanced `best.pt` + balanced `best_class.pt` | 20 | 92.83% | 90.56% |

推荐优先级：

1. 时间紧：使用单模型多票，速度更快，类别平均准确率更稳。
2. 时间充足：使用三权重集成，整体准确率更高。

