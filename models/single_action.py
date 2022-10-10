from typing import Dict, List
import torch
from torch import nn, distributions
from torch.functional import F
from torch.nn.utils import weight_norm
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


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        emb_dim = 256
        self.conv_encoder = nn.Sequential(
            conv(42, 64, 3),
            nn.ReLU(),
            conv(64, 128, 3),
            nn.ReLU(),
            conv(128, 256, 3),
            nn.ReLU(),
            conv(256, emb_dim, 3),
            nn.ReLU(),
        )
        self.glob_feat_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(2),
            nn.Flatten(),
            nn.Linear(emb_dim * 4, emb_dim),
        )
        self.money_emb = nn.Sequential(
            nn.Linear(11, emb_dim), nn.ReLU(), nn.Linear(emb_dim, emb_dim)
        )
        self.value_proj = nn.Linear(emb_dim, 1)
        self.glob_cont_proj = nn.Linear(emb_dim, 2)
        self.action_type_proj = nn.Linear(emb_dim, 7)

        self.unit_feat_proj = nn.Linear(emb_dim, 2)
        self.unit_hp_emb = nn.Embedding(11, 2)
        self.unit_type_emb = nn.Embedding(num_types, 2)

        self.enemy_feat_proj = nn.Linear(emb_dim, 1)
        self.enemy_hp_emb = nn.Embedding(11, 1)
        self.enemy_type_emb = nn.Embedding(num_types, 1)

        self.resource_feat_proj = nn.Linear(emb_dim, 1)
        self.resource_hp_emb = nn.Embedding(11, 1)
        self.resource_type_emb = nn.Embedding(num_types, 1)

        self.unit_type_proj = nn.Linear(emb_dim, num_types)
        self.building_type_proj = nn.Linear(emb_dim, num_types)
        self.building_loc_proj = nn.Linear(emb_dim, 1)
        self.move_loc_proj = nn.Linear(emb_dim, 1)

    @property
    def recurrent_hidden_state_size(self):
        return 1

    @property
    def is_recurrent(self):
        return False

    def get_units_feat(
        self, spatial_feat: torch.Tensor, obs: Dict[str, torch.Tensor], prefix: str
    ):
        loc = obs[f'{prefix}_y'] * 32 + obs[f'{prefix}_x']
        bsz, nc = spatial_feat.size()[:2]
        loc = loc.unsqueeze(2).repeat(1, 1, nc)
        x = spatial_feat.view(bsz, nc, -1).transpose(1, 2).gather(1, loc)
        return x

    def forward(self, obs: Dict[str, torch.Tensor]):
        bsz = obs['map'].size(0)
        device = obs['map'].device

        spatial_feat = self.conv_encoder(obs['map'])
        glob_feat = self.glob_feat_proj(spatial_feat)
        glob_feat = F.relu(glob_feat + self.money_emb(obs['resource_bin']))
        value = self.value_proj(glob_feat)
        glob_cont_prob = F.softmax(self.glob_cont_proj(glob_feat), dim=-1)
        cmd_type_prob = F.softmax(self.action_type_proj(glob_feat), dim=-1)

        num_units = obs['army_type'].size(1)
        tmp = torch.tensor(range(num_units), device=device).unsqueeze(0).repeat(bsz, 1)

        units_feat = self.get_units_feat(spatial_feat, obs, 'army')
        select_unit_prob = F.softmax(
            self.unit_feat_proj(units_feat)
            + self.unit_hp_emb((obs['army_hp'] * 10).long())
            + self.unit_type_emb(obs['army_type']),
            dim=-1,
        )

        num_resource = obs['resource_type'].size(1)
        tmp = (
            torch.tensor(range(num_resource), device=device).unsqueeze(0).repeat(bsz, 1)
        )
        resource_mask = (
            torch.zeros((bsz, num_resource), device=device)
            .masked_fill_(tmp >= obs['num_resource'], float('-inf'))
            .unsqueeze(2)
        )
        resource_mask[:, 0, :] = 0
        resource_feat = self.get_units_feat(spatial_feat, obs, 'resource')
        gather_idx_prob = F.softmax(
            self.resource_feat_proj(resource_feat)
            + self.resource_hp_emb((obs['resource_hp'] * 10).long())
            + self.resource_type_emb(obs['resource_type'])
            + resource_mask,
            dim=-2,
        ).squeeze(-1)

        num_enemy = obs['enemy_type'].size(1)
        tmp = torch.tensor(range(num_enemy), device=device).unsqueeze(0).repeat(bsz, 1)
        enemy_mask = (
            torch.zeros((bsz, num_enemy), device=device)
            .masked_fill_(tmp >= obs['num_enemy'], float('-inf'))
            .unsqueeze(2)
        )
        enemy_mask[:, 0, :] = 0
        enemy_feat = self.get_units_feat(spatial_feat, obs, 'enemy')
        attack_idx_prob = F.softmax(
            self.enemy_feat_proj(enemy_feat)
            + self.enemy_hp_emb((obs['enemy_hp'] * 10).long())
            + self.enemy_type_emb(obs['enemy_type'])
            + enemy_mask,
            dim=-2,
        ).squeeze(-1)

        unit_type_prob = F.softmax(self.unit_type_proj(glob_feat), dim=-1)
        building_type_prob = F.softmax(self.building_type_proj(glob_feat), dim=-1)
        bsz, nc = spatial_feat.size()[:2]
        tmp = spatial_feat.view(bsz, nc, -1).transpose(1, 2)
        building_loc_prob = F.softmax(self.building_loc_proj(tmp).squeeze(2), dim=-1)
        move_loc_prob = F.softmax(self.move_loc_proj(tmp).squeeze(2), dim=-1)

        return value, [
            glob_cont_prob,
            cmd_type_prob,
            gather_idx_prob,
            attack_idx_prob,
            unit_type_prob,
            building_type_prob,
            building_loc_prob,
            move_loc_prob,
            *[select_unit_prob[:, i] for i in range(num_units)],
        ]

    def compute_loss(self, obs: Dict[str, torch.Tensor]):
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
        _, action_pi = self.forward(_obs)

        glob_cont_loss = F.nll_loss((action_pi[0]).log(), obs['glob_cont'])
        mask: torch.Tensor = (obs['current_cmds']['cmd_type'] != 0) * (
            obs['current_cmds']['cmd_type'] != 6
        )
        eps = 1e-6
        action_unit_id = distributions.Categorical(mask + eps).sample().unsqueeze(1)

        eps = 0
        cmd_type = obs['current_cmds']['cmd_type'].gather(1, action_unit_id)
        cmd_type_loss = -(action_pi[1].gather(1, cmd_type)).log()
        gather_idx_loss = -(
            action_pi[2].gather(
                1, obs['current_cmds']['target_gather_idx'].gather(1, action_unit_id)
            )
            + eps
        ).log()
        attack_idx_loss = -(
            action_pi[3].gather(
                1, obs['current_cmds']['target_attack_idx'].gather(1, action_unit_id)
            )
            + eps
        ).log()
        unit_type_loss = -(
            action_pi[4].gather(
                1, obs['current_cmds']['target_type'].gather(1, action_unit_id)
            )
            + eps
        ).log()
        building_type_loss = -(
            action_pi[5].gather(
                1, obs['current_cmds']['target_type'].gather(1, action_unit_id)
            )
            + eps
        ).log()
        loc = obs['target_cmds']['target_y'] * 32 + obs['target_cmds']['target_x']
        loc = loc.gather(1, action_unit_id)
        building_loc_loss = -(action_pi[6].gather(1, loc) + eps).log()
        move_loc_loss = -(action_pi[7].gather(1, loc) + eps).log()

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
        action_loss = cmd_type_loss + action_output_losses.gather(1, cmd_type)

        select_loss = -torch.stack(action_pi[8:], dim=1).log().mean()

        total_loss = (
            glob_cont_loss
            + (
                (1 - obs['glob_cont'].unsqueeze(1))
                * (cmd_type != 0)
                * (cmd_type != 6)
                * action_loss
            ).mean()
            + select_loss
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
        value, action_pi = self.forward(obs)
        entropy = sum(
            distributions.Categorical(tmp).entropy().mean() for tmp in action_pi
        )
        return value, self.get_log_prob(action, action_pi), entropy, rnn_hxs

    def save(self, model_file):
        torch.save(self, model_file)
