import pickle
from typing import Dict, List
import json
import torch
import torch.distributions
from torch import nn
from torch.functional import F
from torch.nn.utils import weight_norm
from gym_minirts.utils import num_types, action_dim, to_tensors


class Config:
    hidden_dim = 128
    num_blocks = 2
    num_actions = 7
    feat_dim = 2048
    num_army_type = 16

    def __init__(self, path: str = None):
        if path is not None:
            tmp = json.load(path)
            for key in tmp:
                setattr(self, key, tmp[key])


def conv(
    in_channels: int,
    out_channels: int = None,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = None,
):
    if out_channels is None:
        out_channels = in_channels
    if padding is None:
        padding = kernel_size // 2
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)


class Residual(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = None, stride: int = 1):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        self.conv1 = conv(in_channels, out_channels, 3)
        self.conv2 = conv(out_channels, out_channels, 3, stride)
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1), nn.AvgPool2d(stride)
            )
        else:
            self.skip = None

    def forward(self, x0: torch.Tensor):
        x = F.relu(self.conv1(x0))
        x = self.conv2(x)
        if self.skip is not None:
            x0 = self.skip(x0)
        return F.relu(x + x0)


class ConvNet(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.entry = nn.Conv2d(42, config.hidden_dim)
        modules = []
        for _ in range(config.num_blocks):
            modules.append(Residual(config.hidden_dim))
        self.body = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor):
        x = self.entry(x)
        x = self.body(x)
        return x


class Attention(nn.Module):
    def __init__(self, emb_dim: int = 128):
        super().__init__()
        self.emb_dim = emb_dim
        self.q_proj = nn.Linear(emb_dim, emb_dim)
        self.k_proj = nn.Linear(emb_dim, emb_dim)

    def forward(self, q: torch.Tensor, k: torch.Tensor, mask: torch.Tensor = None):
        # [bsz, L1, emb_dim] [bsz, L2, emb_dim] [bsz, L1, L2]
        e = torch.bmm(q, k.transpose(1, 2))
        if mask is not None:
            e = e + mask
        e = e.softmax(dim=-1)
        return e


# class Policy(nn.Module):
#     def __init__(self, config: Config):
#         self.config = config
#         self.conv_net = ConvNet(config)
#         self.feat_proj = nn.Sequential(
#             nn.AdaptiveAvgPool2d(2),
#             nn.Linear(4 * config.hidden_dim, config.feat_dim),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#         )
#         self.action_proj = nn.Linear(config.feat_dim, action_dim)
#         self.value_proj = nn.Linear(config.feat_dim, 1)
#         self.glob_cont_proj = nn.Linear(config.feat_dim, 2)
#         self.building_type_proj = nn.Linear(config.feat_dim, num_types)

#     @staticmethod
#     def get_feat_by_loc(feat: torch.Tensor, x: torch.Tensor, y: torch.Tensor):
#         _, nc, h, w = feat.size()
#         feat = feat.view(-1, nc, h * w)  # [bsz, nc, h*w]
#         loc = y * w + x  # [bsz, num_units]
#         loc = loc.unsqueeze(1).expand(1, nc, 1)  # [bsz, nc, num_units]
#         result = feat.gather(2, loc)
#         result = result.transpose(1, 2).contiguous()  # [bsz, num_units, nc]
#         return result

#     def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
#         spatial_feat = self.conv_net(obs['map'])
#         x = self.feat_proj(spatial_feat)
#         return x

#     def act(self, obs: Dict[str, torch.Tensor]):
#         x = self.forward(obs)
#         value = self.value_proj(x)
#         bsz = value.size(0)
#         device = value.device


class PolicySimple(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        emb_dim = 128
        self.spatial_encoder = nn.Sequential(
            conv(42, 32, 3),
            nn.ReLU(),
            Residual(32, 64, 2),
            Residual(64, 128, 2),
            Residual(128, 256, 2),
            Residual(256, 512, 2),
            nn.Flatten(),
            nn.Linear(2048, 128),
            nn.ReLU(),
            nn.Linear(128, emb_dim),
        )
        self.money_emb = nn.Linear(11, emb_dim)

        self.value_proj = nn.Linear(emb_dim, 1)
        self.glob_cont_proj = nn.Linear(emb_dim, 2)

        self.hp_emb = nn.Embedding(11, emb_dim)
        self.type_emb = nn.Embedding(16, emb_dim)
        self.loc_emb = nn.Embedding(1024, emb_dim)

        self.units_feat_proj = nn.Linear(emb_dim, emb_dim)
        self.gather_idx_attn = Attention(emb_dim)
        self.attack_idx_attn = Attention(emb_dim)

        self.action_type_proj = nn.Linear(emb_dim, 7)
        self.unit_type_proj = nn.Linear(emb_dim, 16)
        self.building_type_proj = nn.Linear(emb_dim, 16)
        self.building_loc_proj = nn.Linear(emb_dim, 1024)
        self.move_loc_proj = nn.Linear(emb_dim, 1024)

        if pretrained:
            self.load_state_dict(
                torch.load(
                    '/mnt/shared/minirts/scripts/behavior_clone/saved_models/policy_simple/best_checkpoint.pt'
                )
            )

    @property
    def is_recurrent(self):
        return False

    @property
    def recurrent_hidden_state_size(self):
        return 1

    def get_units_feat(self, obs: Dict[str, torch.Tensor], prefix: str):
        loc = obs[f'{prefix}_y'] * 32 + obs[f'{prefix}_x']
        return F.relu(
            self.hp_emb((obs[f'{prefix}_hp'] * 10).long())
            + self.type_emb(obs[f'{prefix}_type'])
            + self.loc_emb(loc)
        )

    def forward(self, obs: Dict[str, torch.Tensor]):
        result = {}

        f = self.spatial_encoder(obs['map'])
        f = F.relu(f + self.money_emb(obs['resource_bin']))
        bsz = obs['map'].size(0)
        device = f.device

        value = self.value_proj(f)
        result['glob_cont_prob'] = F.softmax(self.glob_cont_proj(f), dim=-1)

        units_feat = self.get_units_feat(obs, 'army')
        enemy_feat = self.get_units_feat(obs, 'enemy')
        resource_feat = self.get_units_feat(obs, 'resource')

        tmp = torch.tensor(range(50), device=device).unsqueeze(0).repeat(bsz, 1)
        enemy_mask = (
            torch.zeros((bsz, 50), device=device)
            .masked_fill_(tmp >= obs['num_enemy'], float('-inf'))
            .unsqueeze(1)
        )
        enemy_mask[:, :, 0] = 0
        resources_mask = (
            torch.zeros((bsz, 50), device=device)
            .masked_fill_(tmp >= obs['num_resource'], float('-inf'))
            .unsqueeze(1)
        )
        resources_mask[:, :, 0] = 0

        tmp = self.units_feat_proj(units_feat) + f.unsqueeze(1)
        result['cmd_type_prob'] = F.softmax(self.action_type_proj(tmp), dim=-1)
        result['unit_type_prob'] = F.softmax(self.unit_type_proj(tmp), dim=-1)
        result['move_loc_prob'] = F.softmax(self.move_loc_proj(tmp), dim=-1)
        result['building_type_prob'] = F.softmax(self.building_type_proj(tmp), dim=-1)
        result['building_loc_prob'] = F.softmax(self.building_loc_proj(tmp), dim=-1)

        result['gather_idx_prob'] = self.gather_idx_attn(
            tmp, resource_feat, resources_mask
        )
        result['attack_idx_prob'] = self.attack_idx_attn(tmp, enemy_feat, enemy_mask)

        return value, to_tensors(result)

    def get_log_prob(
        self, action: torch.Tensor, action_pi: List[torch.Tensor]
    ) -> torch.Tensor:
        bsz = action.size(0)
        result = torch.zeros((bsz, 1), device=action.device)
        for i, pi in enumerate(action_pi):
            # [bsz, nu?] [bsz, nu?, L]
            lprob = pi.gather(-1, action[:, i, None]).log()
            if len(lprob.size()) > 2:
                lprob = lprob.sum(1)
            result = result + lprob
        return result

    def act(self, obs: Dict[str, torch.Tensor], rnn_hxs, *unused):
        value, action_pi = self.forward(obs)
        action = torch.stack(
            [torch.distributions.Categorical(tmp).sample() for tmp in action_pi], dim=1
        )
        return value, action, self.get_log_prob(action, action_pi), rnn_hxs

    def get_value(self, obs: Dict[str, torch.Tensor], *unused):
        value, _ = self.forward(obs)
        return value

    def evaluate_actions(
        self, obs: Dict[str, torch.Tensor], rnn_hxs, masks, action: torch.Tensor
    ):
        value, action_pi = self.forward(obs)
        entropy = sum(
            torch.distributions.Categorical(tmp).entropy().mean() for tmp in action_pi
        )
        return value, self.get_log_prob(action, action_pi), entropy, rnn_hxs
