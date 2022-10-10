# from scripts.behavior_clone.common_utils.assert_utils import assert_eq
from torch._C import device
from a2c_ppo_acktr import model
import pickle
import sys
import os
import torch
from torch import nn, distributions
import torch.nn.functional as F
import numpy as np
from typing import Dict, List
from a2c_ppo_acktr.utils import init
sys.path.append(os.path.join(os.environ['MINIRTS_ROOT'], 'scripts/behavior_clone'))
from executor import Executor, create_loss_masks, gc, DotGatherHead, log_sum_exp
from cmd_heads import assert_eq

class UnitSelectHead(DotGatherHead):

    def compute_loss(self, ufeat, efeat, globfeat, num_enemy, target_cmd_type):
        """loss, averaged by sum(mask)

        ufeat: [batch, padded_num_unit, ufeat_dim]
        efeat: [batch, padded_num_enemy, efeat_dim]
        globfeat: [batch, padded_num_unit, globfeat_dim]
        num_enemy: [batch]
        target_idx: [batch, padded_num_unit]
          target_idx[i, j] is real target idx iff unit_{i,j} is attacking
        mask: [batch, padded_num_unit]
          mask[i] = 1 iff unit i is true unit and its cmd_type == ATTACK
        """
        target_idxs = ((target_cmd_type != gc.CmdTypes.CONT.value) & (target_cmd_type != 0)).float()
        mask = target_idxs.sum(1) != 0
        target_idxs[~mask] = 1e-6
        target_prob = target_idxs / target_idxs.sum(1, keepdim=True)

        prob = self.forward(ufeat, efeat, globfeat, num_enemy).squeeze(1)
        # prob [batch, pnum_unit, pnum_enemy]
        # prob = prob.gather(2, target_idx.unsqueeze(2)).squeeze(2)
        kl = F.kl_div((prob + 1e-6).log(), target_prob, reduction='none')
        # prob [batch, pnum_unit]
        # logp = (prob + 1e-6).log()

        # loss = -(logp * mask).sum(1)
        loss = (kl * mask.unsqueeze(1)).sum(1)
        return loss

class ExecutorSingle(Executor):

    def __init__(self, *args, **kwargs):
        if 'args' in kwargs:
            kwargs['args'].inst_dict_path = os.path.join(os.environ['MINIRTS_ROOT'], 'data/dataset/dict.pt')
        super(ExecutorSingle, self).__init__(*args, **kwargs)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.unit_cls = UnitSelectHead(
            self.conv_encoder.glob_dim,
            self.conv_encoder.army_dim + self.conv_encoder.inst_dim,
            0
        )
        init_(self.unit_cls.net[0])
        self.value_proj = init_(nn.Linear(self.conv_encoder.glob_dim, 1))

    @classmethod
    def load(cls, model_file, pretrained=True):
        params = pickle.load(open(model_file + '.params', 'rb'))
        print(params)
        model = cls(**params)
        if pretrained:
            model.load_state_dict(torch.load(model_file), strict=False)
        return model
    
    def forward(self, batch, *, unit_ids=None):
        bsz = batch['map'].size(0)
        device = batch['map'].device
        inst = torch.ones([bsz, 15]).fill_(927).long().to(device=device)
        inst_len = torch.zeros(bsz).to(device=device)
        inst_cont = torch.zeros([bsz, 1]).to(device=device)
        executor_input = self.format_executor_input(
            batch, inst, inst_len, inst_cont
        )

        features = self.conv_encoder(executor_input)
        # army_feat, enemy_feat, resource_feat, glob_feat, map_feat = self.conv_encoder(batch)

        num_army = executor_input['my_units']['num_units']
        num_enemy = executor_input['enemy_units']['num_units']
        num_resource = executor_input['resource_units']['num_units']

        glob_feat = torch.cat([features['sum_army'],
                               features['sum_enemy'],
                               features['sum_resource'],
                               features['money_feat'],
                               features['sum_inst']], 1)

        value = self.value_proj(glob_feat)
        
        # prob is omitted in variable names
        glob_cont = self.glob_cont_cls.compute_prob(glob_feat)
        # glob_cont = glob_cont[:, 1]

        army_inst = torch.cat([features['army_feat'], features['inst_feat']], 2)
        ctype_context = torch.cat([features['sum_army'],
                                   features['sum_enemy'],
                                   features['sum_resource'],
                                   features['money_feat']], 1)

        # select units
        bsz = num_army.size(0)
        # device = army_inst.device
        # num_units = executor_input['my_units']['num_units'].masked_fill(executor_input['my_units']['num_units'] == 0, 1)
        
        # # selected_unit = torch.from_numpy(np.random.randint(0, num_units.view(-1, 1).cpu())).to(device=device)  # random select one unit
        # selected_unit = torch.arange(5).unsqueeze(0).repeat([bsz, 1]).to(device=device)
        
        # total_unit = army_inst.size(1)
        # selected_army_inst = torch.gather(army_inst, 1, selected_unit.unsqueeze(2).repeat(1, 1, army_inst.size(-1)))

        select_unit_prob = self.unit_cls.compute_prob(glob_feat.unsqueeze(1), army_inst, None, num_army)
        if unit_ids is None:
            selected_unit = distributions.Categorical(probs=select_unit_prob).sample()
        else:
            selected_unit = unit_ids
        selected_army_inst = torch.gather(army_inst, 1, selected_unit.unsqueeze(2).repeat(1, 1, army_inst.size(-1)))

        cmd_type = self.cmd_type_cls.compute_prob(selected_army_inst, ctype_context)
        gather_idx = self.gather_cls.compute_prob(
            selected_army_inst, features['resource_feat'], None, num_resource)
        attack_idx = self.attack_cls.compute_prob(
            selected_army_inst, features['enemy_feat'], None, num_enemy)
        unit_type = self.build_unit_cls.compute_prob(selected_army_inst, None)
        building_type, building_loc = self.build_building_cls.compute_prob(
            selected_army_inst, features['map_feat'], None)
        move_loc = self.move_cls.compute_prob(selected_army_inst, features['map_feat'], None)
    
        probs = {
            'selected_unit': select_unit_prob.squeeze(1),
            'glob_cont_prob': glob_cont.squeeze(1),
            'cmd_type_prob': cmd_type.squeeze(1),
            'gather_idx_prob': gather_idx.squeeze(1),
            'attack_idx_prob': attack_idx.squeeze(1),
            'unit_type_prob': unit_type.squeeze(1),
            'building_type_prob': building_type.squeeze(1),
            'building_loc_prob': building_loc.squeeze(1),
            'move_loc_prob': move_loc.squeeze(1),
        }

        return value, probs, selected_unit.squeeze(1)
    
    # For evaluation in match2.py
    def compute_prob(self, batch):
        features = self.conv_encoder(batch)
        # army_feat, enemy_feat, resource_feat, glob_feat, map_feat = self.conv_encoder(batch)

        num_army = batch['my_units']['num_units']
        num_enemy = batch['enemy_units']['num_units']
        num_resource = batch['resource_units']['num_units']

        glob_feat = torch.cat([features['sum_army'],
                               features['sum_enemy'],
                               features['sum_resource'],
                               features['money_feat'],
                               features['sum_inst']], 1)
        # prob is omitted in variable names
        glob_cont = self.glob_cont_cls.compute_prob(glob_feat)
        # glob_cont = glob_cont[:, 1]

        army_inst = torch.cat([features['army_feat'], features['inst_feat']], 2)
        ctype_context = torch.cat([features['sum_army'],
                                   features['sum_enemy'],
                                   features['sum_resource'],
                                   features['money_feat']], 1)

        # select units
        bsz = num_army.size(0)
        device = army_inst.device
        num_units = batch['my_units']['num_units'].masked_fill(batch['my_units']['num_units'] == 0, 1)
        
        # selected_unit = torch.from_numpy(np.random.randint(0, num_units.view(-1, 1).cpu())).to(device=device)  # random select one unit
        # selected_unit = torch.arange(5).unsqueeze(0).repeat([bsz, 1]).to(device=device)
        select_unit_prob = self.unit_cls.compute_prob(glob_feat.unsqueeze(1), army_inst, None, num_army)
        selected_unit = distributions.Categorical(probs=select_unit_prob).sample()
        
        total_unit = army_inst.size(1)
        selected_army_inst = torch.gather(army_inst, 1, selected_unit.unsqueeze(2).repeat(1, 1, army_inst.size(-1)))

        # init
        cmd_type = torch.zeros([bsz, total_unit, self.cmd_type_cls.out_dim], device=device)
        cmd_type[:, :, gc.CmdTypes.CONT.value] = 1
        gather_idx = torch.zeros([bsz, total_unit, features['resource_feat'].size(1)], device=device)
        attack_idx = torch.zeros([bsz, total_unit, features['enemy_feat'].size(1)], device=device)
        unit_type = torch.zeros([bsz, total_unit, self.build_unit_cls.out_dim], device=device)
        building_type = torch.zeros([bsz, total_unit, self.build_building_cls.type_net.out_dim], device=device)
        building_loc = torch.zeros([bsz, total_unit, features['map_feat'].size(1)], device=device)
        move_loc = torch.zeros([bsz, total_unit, features['map_feat'].size(1)], device=device)

        cmd_type.scatter_(
            1, selected_unit.unsqueeze(2).repeat(1, 1, cmd_type.size(-1)),
            self.cmd_type_cls.compute_prob(selected_army_inst, ctype_context)
        )
        gather_idx.scatter_(
            1, selected_unit.unsqueeze(2).repeat(1, 1, gather_idx.size(-1)),
            self.gather_cls.compute_prob(selected_army_inst, features['resource_feat'], None, num_resource)
        )
        attack_idx.scatter_(
            1, selected_unit.unsqueeze(2).repeat(1, 1, attack_idx.size(-1)),
            self.attack_cls.compute_prob(selected_army_inst, features['enemy_feat'], None, num_enemy)
        )
        unit_type.scatter_(
            1, selected_unit.unsqueeze(2).repeat(1, 1, unit_type.size(-1)),
            self.build_unit_cls.compute_prob(selected_army_inst, None)
        )
        tmp_building_type, tmp_building_loc = self.build_building_cls.compute_prob(selected_army_inst, features['map_feat'], None)
        building_type.scatter_(
            1, selected_unit.unsqueeze(2).repeat(1, 1, building_type.size(-1)),
            tmp_building_type
        )
        building_loc.scatter_(
            1, selected_unit.unsqueeze(2).repeat(1, 1, building_loc.size(-1)),
            tmp_building_loc
        )
        move_loc.scatter_(
            1, selected_unit.unsqueeze(2).repeat(1, 1, move_loc.size(-1)),
            self.move_cls.compute_prob(selected_army_inst, features['map_feat'], None)
        )

        # cmd_type = self.cmd_type_cls.compute_prob(army_inst, ctype_context)
        # gather_idx = self.gather_cls.compute_prob(
        #     army_inst, features['resource_feat'], None, num_resource)
        # attack_idx = self.attack_cls.compute_prob(
        #     army_inst, features['enemy_feat'], None, num_enemy)
        # unit_type = self.build_unit_cls.compute_prob(army_inst, None)
        # building_type, building_loc = self.build_building_cls.compute_prob(
        #     army_inst, features['map_feat'], None)
        # move_loc = self.move_cls.compute_prob(army_inst, features['map_feat'], None)

        probs = {
            'glob_cont_prob': glob_cont,
            'cmd_type_prob': cmd_type,
            'gather_idx_prob': gather_idx,
            'attack_idx_prob': attack_idx,
            'unit_type_prob': unit_type,
            'building_type_prob': building_type,
            'building_loc_prob': building_loc,
            'move_loc_prob': move_loc,
        }
        return probs

    def compute_loss(self, batch, *, mean=True):
        # army_feat, enemy_feat, resource_feat, glob_feat, map_feat = self.conv_encoder(batch)
        features = self.conv_encoder(batch)

        real_unit_mask, cmd_type_mask = create_loss_masks(
            batch['my_units']['num_units'],
            batch['target_cmds']['cmd_type'],
            self.num_cmd_type
        )

        # global continue classfier
        glob_feat = torch.cat([features['sum_army'],
                               features['sum_enemy'],
                               features['sum_resource'],
                               features['money_feat'],
                               features['sum_inst']], 1)
        glob_cont_loss = self.glob_cont_cls.compute_loss(
            glob_feat, batch['glob_cont']
        )

        army_inst = torch.cat([features['army_feat'], features['inst_feat']], 2)
        # Unit Selector Loss
        unit_select_loss = self.unit_cls.compute_loss(
            glob_feat.unsqueeze(1), army_inst, None, batch['my_units']['num_units'],
            batch['target_cmds']['cmd_type']
        )

        # action-arg classifiers
        gather_loss = self.gather_cls.compute_loss(
            army_inst,
            features['resource_feat'],
            None,
            batch['resource_units']['num_units'],
            batch['target_cmds']['target_gather_idx'],
            cmd_type_mask[:, :, gc.CmdTypes.GATHER.value]
        )
        attack_loss = self.attack_cls.compute_loss(
            army_inst,
            features['enemy_feat'],
            None,            # glob_feat,
            batch['enemy_units']['num_units'],
            batch['target_cmds']['target_attack_idx'],
            cmd_type_mask[:, :, gc.CmdTypes.ATTACK.value]
        )
        build_building_loss = self.build_building_cls.compute_loss(
            army_inst,
            features['map_feat'],
            None,            # glob_feat,
            batch['target_cmds']['target_type'],
            batch['target_cmds']['target_x'],
            batch['target_cmds']['target_y'],
            cmd_type_mask[:, :, gc.CmdTypes.BUILD_BUILDING.value]
        )
        build_unit_loss, nil_build_logp = self.build_unit_cls.compute_loss(
            army_inst,
            None,            # glob_feat,
            batch['target_cmds']['target_type'],
            cmd_type_mask[:, :, gc.CmdTypes.BUILD_UNIT.value],
            include_nil=True
        )
        move_loss = self.move_cls.compute_loss(
            army_inst,
            features['map_feat'],
            None,            # glob_feat,
            batch['target_cmds']['target_x'],
            batch['target_cmds']['target_y'],
            cmd_type_mask[:, :, gc.CmdTypes.MOVE.value]
        )

        # type loss
        ctype_context = torch.cat([features['sum_army'],
                                   features['sum_enemy'],
                                   features['sum_resource'],
                                   features['money_feat']], 1)
        cmd_type_logp = self.cmd_type_cls.compute_prob(army_inst, ctype_context, logp=True)
        # cmd_type_logp: [batch, num_unit, num_cmd_type]

        # extra continue
        # cmd_type_prob = self.cmd_type_cls.compute_prob(army_inst, ctype_context)
        # cont_type_prob = cmd_type_prob[:, :, gc.CmdTypes.CONT.value].clamp(max=1-1e-6)
        build_unit_type_logp = cmd_type_logp[:, :, gc.CmdTypes.BUILD_UNIT.value]
        extra_cont_logp = build_unit_type_logp + nil_build_logp
        # extra_cont_logp: [batch, num_unit]
        # print('extra cont logp size:', extra_cont_logp.size())
        # the following hack only works if CONT is the last one
        assert gc.CmdTypes.CONT.value == len(gc.CmdTypes) - 1
        assert extra_cont_logp.size() == cmd_type_logp[:, :, gc.CmdTypes.CONT.value].size()
        cont_logp = log_sum_exp(
            cmd_type_logp[:, :, gc.CmdTypes.CONT.value], extra_cont_logp)
        # cont_logp: [batch, num_unit]
        cmd_type_logp = torch.cat(
            [cmd_type_logp[:, :, :gc.CmdTypes.CONT.value], cont_logp.unsqueeze(2)], 2)
        # cmd_type_logp: [batch, num_unit, num_cmd_type]
        cmd_type_logp = cmd_type_logp.gather(
            2, batch['target_cmds']['cmd_type'].unsqueeze(2)).squeeze(2)
        # cmd_type_logp: [batch, num_unit]
        cmd_type_loss = -(cmd_type_logp * real_unit_mask).sum(1)

        # aggregate losses
        num_my_units_size = batch['my_units']['num_units'].size()
        assert_eq(glob_cont_loss.size(), num_my_units_size)
        assert_eq(cmd_type_loss.size(), num_my_units_size)
        assert_eq(move_loss.size(), num_my_units_size)
        assert_eq(attack_loss.size(), num_my_units_size)
        assert_eq(gather_loss.size(), num_my_units_size)
        assert_eq(build_unit_loss.size(), num_my_units_size)
        assert_eq(build_building_loss.size(), num_my_units_size)
        unit_loss = (cmd_type_loss
                     + move_loss
                     + unit_select_loss
                     + attack_loss
                     + gather_loss
                     + build_unit_loss
                     + build_building_loss)

        unit_loss = (1 - batch['glob_cont'].float()) * unit_loss
        loss = glob_cont_loss + unit_loss

        all_loss = {
            'loss': loss.detach(),
            'unit_loss': unit_loss.detach(),
            'cmd_type_loss': cmd_type_loss.detach(),
            'unit_select_loss': unit_select_loss.detach(),
            'move_loss': move_loss.detach(),
            'attack_loss': attack_loss.detach(),
            'gather_loss': gather_loss.detach(),
            'build_unit_loss': build_unit_loss.detach(),
            'build_building_loss': build_building_loss.detach(),
            'glob_cont_loss': glob_cont_loss.detach()
        }

        if mean:
            for k, v in all_losses.items():
                all_losses[k] = v.mean()
            loss = loss.mean()

        return loss, all_loss

class Policy(nn.Module):

    ACTION_KEYS = ['glob_cont_prob', 'selected_unit', 'cmd_type_prob', 'gather_idx_prob', 'attack_idx_prob', 
        'unit_type_prob', 'building_type_prob', 'building_loc_prob', 'move_loc_prob']

    def __init__(self, pretrained=True):
        super(Policy, self).__init__()
        # model_path = os.path.join(os.environ['MINIRTS_ROOT'], 'pretrained_models/executor_zero.pt')
        # model_path = os.path.join(os.environ['MINIRTS_ROOT'], 'scripts/behavior_clone/save_models/executor_zero/best_checkpoint.pt')
        model_path = os.path.join(os.environ['MINIRTS_ROOT'], 'scripts/behavior_clone/saved_models/executor_zero_with_selector/best_checkpoint.pt')
        self.base = ExecutorSingle.load(model_path, pretrained=pretrained).to('cuda')

    def act(self, inputs, rnn_hxs, masks=None, deterministic=False):

        value, actor_features, selected_unit = self.base(inputs)
        dists = dict([[k, distributions.Categorical(probs=v)] for k, v in actor_features.items()])

        if deterministic:
            action = dict([[k, v.mode()] for k, v in dists.items() if k != "selected_unit"])
        else:
            action = action = dict([[k, v.sample()] for k, v in dists.items() if k != "selected_unit"])
        action['selected_unit'] = selected_unit

        action_log_probs = sum(
            dists[k].log_prob(action[k]) for k in dists
        ).unsqueeze(1)
        return value, self.Dict2Tensor(action), action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        
        value, _, _ = self.base(inputs)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):

        action = self.Tensor2Dict(action)
        value, actor_features, _ = self.base(inputs, unit_ids=action['selected_unit'])
        dists = dict([[k, distributions.Categorical(probs=v)] for k, v in actor_features.items()])

        action_log_probs = sum(
            dists[k].log_prob(action[k]) for k in dists
        ).unsqueeze(1)

        dist_entropy = sum(
            dists[k].entropy().mean() for k in dists
        )

        return value, action_log_probs, dist_entropy, rnn_hxs
    
    def Dict2Tensor(self, d):
 
        return torch.stack([d[k] for k in self.ACTION_KEYS], dim=1)

    def Tensor2Dict(self, T):
        d = {}
        for i, k in enumerate(self.ACTION_KEYS):
            d[k] = T[:, i, None]
        return d

    @property
    def recurrent_hidden_state_size(self):
        return 1

    @property
    def is_recurrent(self):
        return False