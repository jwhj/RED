import torch
from torch import nn


class LangMixer(nn.Module):
    def __init__(self, army_dim: int, inst_dim: int, out_dim: int):
        super().__init__()
        self.army_k = nn.Linear(army_dim, out_dim)
        self.army_v = nn.Linear(army_dim, army_dim + inst_dim)
        self.inst_k = nn.Linear(inst_dim, out_dim)
        self.inst_v = nn.Linear(inst_dim, army_dim + inst_dim)
        self.Q = nn.Linear(out_dim, 1, bias=False)

    def forward(self, army_feat: torch.Tensor, inst_feat: torch.Tensor):
        army_k = self.army_k(army_feat)
        army_v = self.army_v(army_feat)
        inst_k = self.inst_k(inst_feat)
        inst_v = self.inst_v(inst_feat)

        attn_score = torch.cat([self.Q(army_k), self.Q(inst_k)], dim=2)
        attn_score = attn_score.softmax(dim=2)
        self.cache = attn_score.clone()

        return army_v * attn_score[:, :, 0, None] + inst_v * attn_score[:, :, 1, None]
