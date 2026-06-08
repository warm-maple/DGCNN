# 赛道一设计思路

本方案面向 ModelNet40 点云分类任务，输入为每个点的 `x,y,z,nx,ny,nz` 六维信息。数据集采用老师提供的 `dataset/train` 作为训练集；经核对，它与官方 `modelnet40_train.txt` 完全一致，因此使用官方 `modelnet40_test.txt` 作为预演评估集，避免训练和评估样本泄漏。

预处理阶段先将每个点云缓存为 `.npy`，减少训练时反复读取 txt 的磁盘开销。每次取样固定采样 1024 个点，对坐标做中心化和单位球归一化，对法向量重新归一化。训练时加入随机点 dropout、随机缩放、随机平移和小幅 jitter，增强模型对采样扰动和噪声的鲁棒性；验证和推理阶段不加随机增强，但通过多次采样投票提高稳定性。

模型采用 DGCNN 分类网络。DGCNN 在特征空间中动态构建 k 近邻图，使用 EdgeConv 提取局部几何关系，再通过全局最大池化和平均池化形成物体级特征。相比基础 PointNet，DGCNN 能显式建模点与邻域点之间的相对结构；相比过小模型，它在 ModelNet40 上更容易达到 92% 以上整体准确率和 90% 左右类均值准确率。本实现使用 `k=20`、四层 EdgeConv、1024 维全局嵌入，并同时使用坐标和法向量。

训练使用 PyTorch + CUDA，优化器为 SGD，初始学习率 0.1，momentum 0.9，weight decay 1e-4，cosine 学习率衰减，交叉熵使用 label smoothing。第一阶段训练普通 DGCNN 基线；第二阶段从基线最佳权重出发，使用类别均衡采样和轻量类别权重进行微调，以提升小样本类别的 Class Accuracy。训练过程中每轮在预演集评估 Instance Accuracy 和 Class Accuracy，并分别保存整体最优、类均值最优和综合最优权重。最终推理时加载 `best_class.pt`，对同一样本进行 20 次采样投票，输出规定格式的 `id,类别` CSV。预演集最佳结果为 Instance Accuracy 92.22%，Class Accuracy 91.10%。
