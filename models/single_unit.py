from typing import Dict, List
import torch
from torch import nn, distributions
from torch._C import Value
from torch.functional import F
from torch.nn.utils import weight_norm
from models.simple import Attention
from gym_minirts.utils import num_types


def conv(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = None,
):
    if padding is None:
        padding = kernel_size // 2
    return weight_norm(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
    )


def feat_proj(d, emb_dim):
    return nn.Sequential(nn.Linear(d, emb_dim), nn.ReLU(), nn.Linear(emb_dim, emb_dim))


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        d = 128
        self.conv_encoder = nn.Sequential(
            conv(42, 64, 3),
            nn.ReLU(),
            conv(64, 128, 3),
            nn.ReLU(),
            conv(128, d, 3),
            nn.ReLU(),
        )

        emb_dim = 128
        self.hp_embedding = nn.Embedding(11, emb_dim)
        self.type_embedding = nn.Embedding(16, emb_dim)
        self.units_feat_proj = feat_proj(d, emb_dim)
        self.enemy_feat_proj = feat_proj(d, emb_dim)
        self.resource_feat_proj = feat_proj(d, emb_dim)
        self.feat_proj = {
            'army': self.units_feat_proj,
            'enemy': self.enemy_feat_proj,
            'resource': self.resource_feat_proj,
        }
        self.glob_feat_proj = nn.Sequential(
            conv(d, emb_dim, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(2),
            nn.Flatten(),
            nn.Linear(emb_dim * 4, emb_dim),
        )
        self.money_embedding = nn.Sequential(
            nn.Linear(11, 128), nn.ReLU(), nn.Linear(128, emb_dim)
        )

        self.select_attn = Attention(emb_dim)
        self.value_proj = nn.Linear(emb_dim, 1)
        self.glob_cont_proj = nn.Linear(emb_dim, 2)
        self.action_type_proj = nn.Linear(emb_dim * 2, 7)
        self.gather_idx_attn = Attention(emb_dim)
        self.attack_idx_attn = Attention(emb_dim)
        self.unit_type_proj = nn.Linear(emb_dim, num_types)
        self.building_type_proj = nn.Linear(emb_dim, num_types)
        self.building_loc_attn = Attention(emb_dim)
        self.move_loc_attn = Attention(emb_dim)

    @property
    def recurrent_hidden_state_size(self):
        return 1

    @property
    def is_recurrent(self):
        return False

    def get_units_feat(
        self, obs: Dict[str, torch.Tensor], spatial_feat: torch.Tensor, prefix
    ):
        loc = obs[f'{prefix}_y'] * 32 + obs[f'{prefix}_x']
        bsz, nc = spatial_feat.size()[:2]
        loc = loc.unsqueeze(2).repeat(1, 1, nc)
        spatial_feat = spatial_feat.view(bsz, nc, -1).transpose(1, 2).gather(1, loc)
        spatial_feat = self.feat_proj[prefix](spatial_feat)
        hp_feat = self.hp_embedding((obs[f'{prefix}_hp'] * 10).long())
        type_feat = self.type_embedding(obs[f'{prefix}_type'])
        return F.relu(spatial_feat + hp_feat + type_feat)

    def forward(self, obs: Dict[str, torch.Tensor], units_id: torch.LongTensor = None):
        device = obs['map'].device
        bsz = obs['map'].size(0)

        spatial_feat = self.conv_encoder(obs['map'])

        glob_feat = self.glob_feat_proj(spatial_feat)
        glob_feat = F.relu(glob_feat + self.money_embedding(obs['resource_bin']))

        units_feat = self.get_units_feat(obs, spatial_feat, 'army')
        enemy_feat = self.get_units_feat(obs, spatial_feat, 'enemy')
        resource_feat = self.get_units_feat(obs, spatial_feat, 'resource')

        num_units = obs['army_type'].size(1)
        num_enemy = obs['enemy_type'].size(1)
        num_resource = obs['resource_type'].size(1)

        tmp = torch.tensor(range(num_units), device=device).unsqueeze(0).repeat(bsz, 1)
        units_mask = (
            torch.zeros((bsz, num_units), device=device)
            .masked_fill_(tmp >= obs['num_army'], float('-inf'))
            .unsqueeze(1)
        )
        units_mask[:, :, 0] = 0
        tmp = torch.tensor(range(num_enemy), device=device).unsqueeze(0).repeat(bsz, 1)
        enemy_mask = (
            torch.zeros((bsz, num_enemy), device=device)
            .masked_fill_(tmp >= obs['num_enemy'], float('-inf'))
            .unsqueeze(1)
        )
        enemy_mask[:, :, 0] = 0
        tmp = (
            torch.tensor(range(num_resource), device=device).unsqueeze(0).repeat(bsz, 1)
        )
        resource_mask = (
            torch.zeros((bsz, num_resource), device=device)
            .masked_fill_(tmp >= obs['num_resource'], float('-inf'))
            .unsqueeze(1)
        )
        resource_mask[:, :, 0] = 0

        value = self.value_proj(glob_feat)
        action_pi = []
        glob_cont_prob = F.softmax(self.glob_cont_proj(glob_feat), dim=-1)
        action_pi.append(glob_cont_prob)

        units_pi = self.select_attn(glob_feat.unsqueeze(1), units_feat, units_mask)
        # [bsz, 1, nu]
        action_pi.append(units_pi.squeeze(1))
        if units_id is None:
            units_id = distributions.Categorical(units_pi).sample()
        units_feat = units_feat.gather(
            1, units_id.unsqueeze(2).repeat(1, 1, units_feat.size(2))
        )

        cmd_type_prob = F.softmax(
            self.action_type_proj(torch.cat((glob_feat, units_feat.squeeze(1)), dim=1)),
            dim=-1,
        )
        action_pi.append(cmd_type_prob)
        gather_idx_prob = self.gather_idx_attn(
            units_feat, resource_feat, resource_mask
        ).squeeze(1)
        action_pi.append(gather_idx_prob)
        attack_idx_prob = self.attack_idx_attn(
            units_feat, enemy_feat, enemy_mask
        ).squeeze(1)
        action_pi.append(attack_idx_prob)
        unit_type_prob = F.softmax(self.unit_type_proj(units_feat).squeeze(1), dim=-1)
        action_pi.append(unit_type_prob)
        building_type_prob = F.softmax(
            self.building_type_proj(units_feat).squeeze(1), dim=-1
        )
        action_pi.append(building_type_prob)
        spatial_feat = spatial_feat.flatten(start_dim=2).transpose(1, 2)
        building_loc_prob = self.building_loc_attn(units_feat, spatial_feat).squeeze(1)
        action_pi.append(building_loc_prob)
        move_loc_prob = self.move_loc_attn(units_feat, spatial_feat).squeeze(1)
        action_pi.append(move_loc_prob)
        return value, action_pi

    def compute_loss(self, obs: Dict[str, torch.Tensor]):
        mask: torch.Tensor = (obs['current_cmds']['cmd_type'] != 0) * (
            obs['current_cmds']['cmd_type'] != 6
        )
        eps = 1e-6
        action_unit_id = distributions.Categorical(mask + eps).sample().unsqueeze(1)

        _obs = {'map': obs['map'], 'resource_bin': obs['resource_bin']}
        for k1, k2 in [
            ('my_units', 'army'),
            ('enemy_units', 'enemy'),
            ('resource_units', 'resource'),
        ]:
            _obs[f'{k2}_type'] = obs[k1]['types']
            _obs[f'{k2}_x'] = obs[k1]['xs']
            _obs[f'{k2}_y'] = obs[k1]['ys']
            _obs[f'{k2}_hp'] = obs[k1]['hps']
            _obs[f'num_{k2}'] = obs[k1]['num_units'].unsqueeze(1)
        _, action_pi = self.forward(_obs, action_unit_id)

        glob_cont_loss = F.nll_loss((action_pi[0]).log(), obs['glob_cont'])

        action_unit_loss = -(action_pi[1].gather(1, action_unit_id) + eps).log()
        cmd_type = obs['current_cmds']['cmd_type'].gather(1, action_unit_id)
        cmd_type_loss = -(action_pi[2].gather(1, cmd_type)).log()
        gather_idx_loss = -(
            action_pi[3].gather(
                1, obs['current_cmds']['target_gather_idx'].gather(1, action_unit_id)
            )
            + eps
        ).log()
        attack_idx_loss = -(
            action_pi[4].gather(
                1, obs['current_cmds']['target_attack_idx'].gather(1, action_unit_id)
            )
            + eps
        ).log()
        unit_type_loss = -(
            action_pi[5].gather(
                1, obs['current_cmds']['target_type'].gather(1, action_unit_id)
            )
            + eps
        ).log()
        building_type_loss = -(
            action_pi[6].gather(
                1, obs['current_cmds']['target_type'].gather(1, action_unit_id)
            )
            + eps
        ).log()
        loc = obs['target_cmds']['target_y'] * 32 + obs['target_cmds']['target_x']
        loc = loc.gather(1, action_unit_id)
        building_loc_loss = -(action_pi[7].gather(1, loc) + eps).log()
        move_loc_loss = -(action_pi[8].gather(1, loc) + eps).log()

        action_output_losses = torch.cat(
            [
                torch.zeros_like(gather_idx_loss),
                gather_idx_loss,
                attack_idx_loss,
                building_type_loss + building_loc_loss,
                unit_type_loss,
                move_loc_loss,
                torch.zeros_like(gather_idx_loss),
            ],
            dim=1,
        )
        action_loss = (
            action_unit_loss + cmd_type_loss + action_output_losses.gather(1, cmd_type)
        )

        total_loss = (
            glob_cont_loss
            + (
                (1 - obs['glob_cont'].unsqueeze(1))
                * (cmd_type != 0)
                * (cmd_type != 6)
                * action_loss
            ).mean()
        )
        return total_loss, {'loss': total_loss.detach()}

    def get_log_prob(self, action: torch.LongTensor, action_pi: List[torch.Tensor]):
        bsz = action.size(0)
        result = torch.zeros((bsz, 1), device=action.device)
        for i, pi in enumerate(action_pi):
            lprob = (pi.gather(-1, action[:, i, None]) + 1e-6).log()
            result = result + lprob
        return result

    def act(self, obs: Dict[str, torch.Tensor], rnn_hxs, *unused):
        value, action_pi = self.forward(obs)
        # __import__('ipdb').set_trace()s
        action = torch.stack(
            [distributions.Categorical(tmp).sample() for tmp in action_pi], dim=1
        )
        return value, action, self.get_log_prob(action, action_pi), rnn_hxs

    def get_value(self, obs: Dict[str, torch.Tensor], *unused):
        value, _ = self.forward(obs)
        return value

    def evaluate_actions(
        self, obs: Dict[str, torch.Tensor], rnn_hxs, masks, action: torch.Tensor
    ):
        value, action_pi = self.forward(obs, action[:, 1, None])
        entropy = sum(
            distributions.Categorical(tmp).entropy().mean() for tmp in action_pi
        )
        return value, self.get_log_prob(action, action_pi), entropy, rnn_hxs

    def save(self, model_file):
        torch.save(self, model_file)
