from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x: torch.Tensor, k: int) -> torch.Tensor:
    """Return k nearest neighbor indices for features shaped [B, C, N]."""
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    return pairwise_distance.topk(k=k, dim=-1)[1]


def get_graph_feature(
    x: torch.Tensor,
    k: int = 20,
    idx: torch.Tensor | None = None,
    knn_features: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build DGCNN edge features [neighbor - center, center]."""
    batch_size, num_dims, num_points = x.size()
    if idx is None:
        idx = knn(knn_features if knn_features is not None else x, k=k)

    device = x.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = (idx + idx_base).reshape(-1)

    x_t = x.transpose(2, 1).contiguous()
    feature = x_t.reshape(batch_size * num_points, num_dims)[idx, :]
    feature = feature.reshape(batch_size, num_points, k, num_dims)
    center = x_t.reshape(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((feature - center, center), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature


class DGCNNClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 40,
        input_dims: int = 6,
        k: int = 20,
        emb_dims: int = 1024,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.input_dims = input_dims
        self.k = k
        self.emb_dims = emb_dims

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_dims * 2, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(512, emb_dims, kernel_size=1, bias=False),
            nn.BatchNorm1d(emb_dims),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.linear1 = nn.Linear(emb_dims * 2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=dropout)
        self.linear3 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, N]. Use xyz for the first spatial graph if normals are present.
        coord_features = x[:, :3, :] if x.size(1) >= 3 else x

        x0 = get_graph_feature(x, k=self.k, knn_features=coord_features)
        x1 = self.conv1(x0).max(dim=-1)[0]

        x2 = get_graph_feature(x1, k=self.k)
        x2 = self.conv2(x2).max(dim=-1)[0]

        x3 = get_graph_feature(x2, k=self.k)
        x3 = self.conv3(x3).max(dim=-1)[0]

        x4 = get_graph_feature(x3, k=self.k)
        x4 = self.conv4(x4).max(dim=-1)[0]

        x_cat = torch.cat((x1, x2, x3, x4), dim=1)
        x_emb = self.conv5(x_cat)
        x_max = F.adaptive_max_pool1d(x_emb, 1).view(x.size(0), -1)
        x_avg = F.adaptive_avg_pool1d(x_emb, 1).view(x.size(0), -1)
        out = torch.cat((x_max, x_avg), dim=1)

        out = F.leaky_relu(self.bn6(self.linear1(out)), negative_slope=0.2)
        out = self.dp1(out)
        out = F.leaky_relu(self.bn7(self.linear2(out)), negative_slope=0.2)
        out = self.dp2(out)
        return self.linear3(out)

