# from scripts.behavior_clone.common_utils.assert_utils import assert_eq
from torch._C import device
from torch.nn.modules.activation import Sigmoid
from a2c_ppo_acktr import model
import sys
import os
import pickle
import torch
from torch import nn, distributions
import torch.nn.functional as F
import numpy as np
from typing import Dict, List
from a2c_ppo_acktr.utils import init
sys.path.append(os.path.join(os.environ['MINIRTS_ROOT'], 'scripts/behavior_clone'))
from gym_minirts.utils import format_single_action
from executor import Executor, create_loss_masks, gc, weight_norm, log_sum_exp, create_real_unit_mask
from cmd_heads import assert_eq
from models.lang_mixer import LangMixer

class UnitContHead(nn.Module):
    def __init__(self, ufeat_dim, globfeat_dim, hid_dim):
        super().__init__()
        self.ufeat_dim = ufeat_dim
        self.globfeat_dim = globfeat_dim
        self.in_dim = ufeat_dim + globfeat_dim
        self.hid_dim = hid_dim

        self.net = nn.Sequential(
            weight_norm(nn.Linear(self.in_dim, self.hid_dim), dim=None),
            nn.ReLU(),
            nn.Linear(self.hid_dim, 1, bias=False)
        )

    def forward(self, ufeat, globfeat):
        if globfeat is None:
            assert self.globfeat_dim == 0
            infeat = ufeat
        else:
            assert globfeat.dim() == 2
            globfeat = globfeat.unsqueeze(1).repeat(1, ufeat.size(1), 1)
            infeat = torch.cat([ufeat, globfeat], 2)

        logit = self.net(infeat)
        return logit

    def compute_loss(self, ufeat, globfeat, label, mask):
        """loss, averaged by sum(mask)

        ufeat: [batch, padded_num_unit, ufeat_dim]
        globfeat: [batch, padded_num_unit, globfeat_dim]
        target_type: [batch, padded_num_unit]
          target_type[i, j] is real unit_type iff unit_{i,j} is build unit
        mask: [batch, padded_num_unit]
          mask[i, j] = 1 iff the unit is true unit and its cmd_type == BUILD_UNIT
        """
        batch, pnum_unit, _  = ufeat.size()
        # assert_eq(target_type.size(), (batch, pnum_unit))
        assert_eq(mask.size(), (batch, pnum_unit))

        logit = self.forward(ufeat, globfeat)

        loss = F.binary_cross_entropy_with_logits(logit.squeeze(2), label.squeeze(2).float(), reduction='none', weight=mask)

        return loss.sum(1)

    def compute_prob(self, ufeat, globfeat, *, logp=False):
        logit = self.forward(ufeat, globfeat)
        # logit [batch, pnum_unit, num_unit_type]
        if logp:
            logp = torch.logsigmoid(logit)
            return logp
        else:
            prob = torch.sigmoid(logit)
            return prob

class ExecutorSingle(Executor):

    def __init__(self, *args, use_lang_mixer=False, **kwargs):
        if 'args' in kwargs:
            kwargs['args'].inst_dict_path = os.path.join(os.environ['MINIRTS_ROOT'], 'data/dataset/dict.pt')
            kwargs['args'].conv_dropout = 0
            kwargs['args'].word_emb_dropout = 0
        super(ExecutorSingle, self).__init__(*args, **kwargs)
        self.use_lang_mixer = use_lang_mixer
        if use_lang_mixer:
            self.lang_mixer = LangMixer(self.conv_encoder.army_dim,
                                        self.conv_encoder.inst_dim,
                                        self.conv_encoder.args.other_out_dim)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        # self.unit_cls = UnitSelectHead(
        #     self.conv_encoder.glob_dim,
        #     self.conv_encoder.army_dim + self.conv_encoder.inst_dim,
        #     0
        # )
        # self.unit_cont = nn.Sequential(
        #     init_(nn.Linear(self.conv_encoder.army_dim + self.conv_encoder.inst_dim, 1)), 
        #     nn.Sigmoid()) # 1 for  continue
        self.unit_cont = UnitContHead(
            self.conv_encoder.army_dim + self.conv_encoder.inst_dim,
            self.conv_encoder.glob_dim - self.conv_encoder.inst_dim,
            self.target_emb_dim
        )
        self.value_proj = init_(nn.Linear(self.conv_encoder.glob_dim, 1))
    
    def _preprocess(self, batch):
        bsz = batch['map'].size(0)
        device = batch['map'].device
        inst = torch.ones([bsz, 15]).fill_(927).long().to(device=device)
        inst_len = torch.zeros(bsz).to(device=device)
        inst_cont = torch.zeros([bsz, 1]).to(device=device)
        executor_input = self.format_executor_input(
            batch, inst, inst_len, inst_cont
        )
        return executor_input
    
    def forward(self, batch, *, unit_is_cont=None, need_preprocess=True):
        if need_preprocess:
            executor_input = self._preprocess(batch)
        else:
            executor_input = batch

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

        if self.use_lang_mixer:
            army_inst = self.lang_mixer(features['army_feat'], features['inst_feat'])
        else:
            army_inst = torch.cat([features['army_feat'], features['inst_feat']], 2)

        ctype_context = torch.cat([features['sum_army'],
                                   features['sum_enemy'],
                                   features['sum_resource'],
                                   features['money_feat']], 1)

        # select units
        bsz = num_army.size(0)
        raw_unit_mask = create_real_unit_mask(executor_input['my_units']['num_units'], army_inst.size(1)).unsqueeze(2)
        device = army_inst.device
        # num_units = executor_input['my_units']['num_units'].masked_fill(executor_input['my_units']['num_units'] == 0, 1)

        # is_cont = self.unit_cont.compute_prob(army_inst, ctype_context) >= 0.5
        unit_cont_prob =  self.unit_cont.compute_prob(army_inst, ctype_context)
        if unit_is_cont is None:
            is_cont = torch.rand_like(raw_unit_mask) < unit_cont_prob
        else:
            is_cont = unit_is_cont.unsqueeze(-1)
        is_cont.masked_fill_(raw_unit_mask == 0, True)
        unit_mask = raw_unit_mask.masked_fill(is_cont.bool(), 0)
        num_units = unit_mask.sum(1).masked_fill(unit_mask.sum(1) == 0, 1)
        
        selected_army_inst = (army_inst * unit_mask).sum(1, keepdim=True) / num_units.view(-1, 1, 1)

        total_army = army_inst.size(1)
        cmd_type = torch.zeros([bsz, total_army, self.cmd_type_cls.out_dim], device=device)
        cmd_type[:, :, gc.CmdTypes.CONT.value] = 1

        cmd_type = torch.where(
            unit_mask.repeat(1, 1, 7).bool(),
            self.cmd_type_cls.compute_prob(selected_army_inst, ctype_context).repeat(1, total_army, 1),
            cmd_type)

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
            'unit_cont_prob': unit_cont_prob.squeeze(2),
            'glob_cont_prob': glob_cont.squeeze(1),
            'cmd_type_prob': cmd_type.squeeze(1),
            'gather_idx_prob': gather_idx.squeeze(1),
            'attack_idx_prob': attack_idx.squeeze(1),
            'unit_type_prob': unit_type.squeeze(1),
            'building_type_prob': building_type.squeeze(1),
            'building_loc_prob': building_loc.squeeze(1),
            'move_loc_prob': move_loc.squeeze(1),
        }

        return value, probs, is_cont.squeeze(2), raw_unit_mask.bool().squeeze(2)
    
    def compute_prob(self, executor_input, deterministic=False):

        def Dict2Tensor(d):
            ACTION_KEYS = ['glob_cont_prob', 'cmd_type_prob', 'gather_idx_prob', 'attack_idx_prob', 
                            'unit_type_prob', 'building_type_prob', 'building_loc_prob', 'move_loc_prob', 'unit_cont_prob']
            return torch.cat([d[k].view(d[k].size(0), -1) for k in ACTION_KEYS], dim=1)

        value, actor_features, unit_cont, raw_unit_mask = self.forward(executor_input, need_preprocess=False)
        dists = dict([[k, distributions.Categorical(probs=v)] for k, v in actor_features.items() if k != 'unit_cont_prob'])
        dists['unit_cont_prob'] = distributions.Bernoulli(probs=actor_features['unit_cont_prob'])

        if deterministic:
            action = dict([[k, v.mode()] for k, v in dists.items() if k != "unit_cont_prob"])
        else:
            action = action = dict([[k, v.sample()] for k, v in dists.items() if k != "unit_cont_prob"])
        action['unit_cont_prob'] = unit_cont.float()

        actions = Dict2Tensor(action)
        return format_single_action(actions)


    # For evaluation in match2.py
    def compute_prob_old(self, executor_input):
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
        # prob is omitted in variable names
        glob_cont = self.glob_cont_cls.compute_prob(glob_feat)
        # glob_cont = glob_cont[:, 1]

        if self.use_lang_mixer:
            army_inst = self.lang_mixer(features['army_feat'], features['inst_feat'])
        else:
            army_inst = torch.cat([features['army_feat'], features['inst_feat']], 2)
        ctype_context = torch.cat([features['sum_army'],
                                   features['sum_enemy'],
                                   features['sum_resource'],
                                   features['money_feat']], 1)

        # select units
        bsz = num_army.size(0)
        raw_unit_mask = create_real_unit_mask(executor_input['my_units']['num_units'], army_inst.size(1)).unsqueeze(2)
        device = army_inst.device
        # num_units = executor_input['my_units']['num_units'].masked_fill(executor_input['my_units']['num_units'] == 0, 1)

        # is_cont = self.unit_cont.compute_prob(army_inst, ctype_context) >= 0.5
        unit_cont_prob =  self.unit_cont.compute_prob(army_inst, ctype_context)

        is_cont = torch.rand_like(raw_unit_mask) < unit_cont_prob

        is_cont.masked_fill_(raw_unit_mask == 0, True)
        unit_mask = raw_unit_mask.masked_fill(is_cont.bool(), 0)
        num_units = unit_mask.sum(1).masked_fill(unit_mask.sum(1) == 0, 1)
        
        selected_army_inst = (army_inst * unit_mask).sum(1, keepdim=True) / num_units.view(-1, 1, 1)

        total_army = army_inst.size(1)
        cmd_type = torch.zeros([bsz, total_army, self.cmd_type_cls.out_dim], device=device)
        cmd_type[:, :, gc.CmdTypes.CONT.value] = 1

        cmd_type = torch.where(
            unit_mask.repeat(1, 1, 7).bool(),
            self.cmd_type_cls.compute_prob(selected_army_inst, ctype_context).repeat(1, total_army, 1),
            cmd_type)
        # cmd_type[unit_mask.repeat(1, 1, 7).bool()] = \
        #     self.cmd_type_cls.compute_prob(selected_army_inst, ctype_context).repeat(1, total_army, 1)[unit_mask.repeat(1, 1, 7).bool()]

        gather_idx = self.gather_cls.compute_prob(
            selected_army_inst, features['resource_feat'], None, num_resource).repeat(1, total_army, 1)
        attack_idx = self.attack_cls.compute_prob(
            selected_army_inst, features['enemy_feat'], None, num_enemy).repeat(1, total_army, 1)
        unit_type = self.build_unit_cls.compute_prob(selected_army_inst, None).repeat(1, total_army, 1)
        building_type, building_loc = self.build_building_cls.compute_prob(
            selected_army_inst, features['map_feat'], None)
        building_type = building_type.repeat(1, total_army, 1)
        building_loc= building_loc.repeat(1, total_army, 1)
        move_loc = self.move_cls.compute_prob(selected_army_inst, features['map_feat'], None).repeat(1, total_army, 1)

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
        ctype_context = torch.cat([features['sum_army'],
                                   features['sum_enemy'],
                                   features['sum_resource'],
                                   features['money_feat']], 1)
        glob_cont_loss = self.glob_cont_cls.compute_loss(
            glob_feat, batch['glob_cont']
        )

        if self.use_lang_mixer:
            army_inst = self.lang_mixer(features['army_feat'], features['inst_feat'])
        else:
            army_inst = torch.cat([features['army_feat'], features['inst_feat']], 2)
        # Unit Selector Loss
        # unit_select_loss = self.unit_cls.compute_loss(
        #     glob_feat.unsqueeze(1), army_inst, None, batch['my_units']['num_units'],
        #     batch['target_cmds']['cmd_type']
        # )

        # is_cont = self.unit_cont.compute_prob(army_inst, ctype_context) >= 0.5
        total_army = real_unit_mask.size(1)

        common_action = torch.argmax(cmd_type_mask.sum(1)[:, : -1], dim=1)
        is_cont = (cmd_type_mask.gather(2, common_action.view(-1, 1, 1).repeat(1, total_army, 1)) != 1)
        # is_cont = (cmd_type_mask[:, :, gc.CmdTypes.CONT.value] == 1).unsqueeze(2)

        unit_cont_loss = self.unit_cont.compute_loss(
            army_inst, ctype_context,
            is_cont,
            real_unit_mask
        )

        non_cont_mask = real_unit_mask.unsqueeze(2).masked_fill(is_cont, 0)
        bsz_mask = (non_cont_mask.sum(1) > 0).squeeze(1)
        num_units = non_cont_mask.sum(1).masked_fill(non_cont_mask.sum(1) == 0, 1)
        
        army_inst = (army_inst * non_cont_mask).sum(1, keepdim=True) / num_units.view(-1, 1, 1)
        
        army_inst = army_inst.repeat(1, total_army, 1)

        cmd_type_mask = cmd_type_mask * non_cont_mask
        # action-arg classifiers
        gather_loss = self.gather_cls.compute_loss(
            army_inst,
            features['resource_feat'],
            None,
            batch['resource_units']['num_units'],
            batch['target_cmds']['target_gather_idx'],
            cmd_type_mask[:, :, gc.CmdTypes.GATHER.value]
        ) * bsz_mask
        attack_loss = self.attack_cls.compute_loss(
            army_inst,
            features['enemy_feat'],
            None,            # glob_feat,
            batch['enemy_units']['num_units'],
            batch['target_cmds']['target_attack_idx'],
            cmd_type_mask[:, :, gc.CmdTypes.ATTACK.value]
        ) * bsz_mask
        build_building_loss = self.build_building_cls.compute_loss(
            army_inst,
            features['map_feat'],
            None,            # glob_feat,
            batch['target_cmds']['target_type'],
            batch['target_cmds']['target_x'],
            batch['target_cmds']['target_y'],
            cmd_type_mask[:, :, gc.CmdTypes.BUILD_BUILDING.value]
        ) * bsz_mask
        build_unit_loss, nil_build_logp = self.build_unit_cls.compute_loss(
            army_inst,
            None,            # glob_feat,
            batch['target_cmds']['target_type'],
            cmd_type_mask[:, :, gc.CmdTypes.BUILD_UNIT.value],
            include_nil=True
        )
        build_unit_loss = build_unit_loss * bsz_mask
        move_loss = self.move_cls.compute_loss(
            army_inst,
            features['map_feat'],
            None,            # glob_feat,
            batch['target_cmds']['target_x'],
            batch['target_cmds']['target_y'],
            cmd_type_mask[:, :, gc.CmdTypes.MOVE.value]
        ) * bsz_mask

        # type loss

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
        cmd_type_loss = -(cmd_type_logp * non_cont_mask.squeeze(-1)).sum(1) * bsz_mask

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
                     + unit_cont_loss
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
            'unit_cont_loss': unit_cont_loss.detach(),
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

def smoothed_Bernoulli_KL(p, q)->torch.Tensor:
    a = p.probs.clamp(1e-6, 1-1e-6)
    b = q.probs.clamp(1e-6, 1-1e-6)
    return distributions.kl_divergence(distributions.Bernoulli(a), distributions.Bernoulli(b))

class Policy(nn.Module):

    ACTION_KEYS = ['glob_cont_prob', 'cmd_type_prob', 'gather_idx_prob', 'attack_idx_prob', 
        'unit_type_prob', 'building_type_prob', 'building_loc_prob', 'move_loc_prob', 'unit_cont_prob']

    def __init__(self, pretrained:bool=True):
        super(Policy, self).__init__()
        model_path = os.path.join(os.environ['MINIRTS_ROOT'], 'scripts/behavior_clone/saved_models/executor_zero_with_cont2/best_checkpoint.pt')
        if pretrained:
            self.base = ExecutorSingle.load(model_path).to('cuda')
        else:
            with open(model_path+'.params', 'rb') as f:
                params = pickle.load(f)
            self.base = ExecutorSingle(**params)

    def act(self, inputs, rnn_hxs, masks=None, deterministic=False):

        value, actor_features, unit_cont, raw_unit_mask = self.base(inputs)
        dists = dict([[k, distributions.Categorical(probs=v)] for k, v in actor_features.items() if k != 'unit_cont_prob'])
        dists['unit_cont_prob'] = distributions.Bernoulli(probs=actor_features['unit_cont_prob'])

        if deterministic:
            action = dict([[k, v.mode()] for k, v in dists.items() if k != "unit_cont_prob"])
        else:
            action = action = dict([[k, v.sample()] for k, v in dists.items() if k != "unit_cont_prob"])
        action['unit_cont_prob'] = unit_cont.float()

        action_log_probs = sum(
            dists[k].log_prob(action[k]) if k != "unit_cont_prob" else dists[k].log_prob(action[k]).masked_fill(~raw_unit_mask, 0).sum(1) for k in dists
        ).unsqueeze(1)
        return value, self.Dict2Tensor(action), action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs=None, masks=None):
        
        value, _, _, _ = self.base(inputs)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action, reference_dists=None, return_entropy_dist=False):

        action = self.Tensor2Dict(action)
        value, actor_features, _, raw_unit_mask = self.base(inputs, unit_is_cont=action['unit_cont_prob'])
        dists = dict([[k, distributions.Categorical(probs=v)] for k, v in actor_features.items() if k != 'unit_cont_prob'])
        dists['unit_cont_prob'] = distributions.Bernoulli(probs=actor_features['unit_cont_prob'])

        action_log_probs = sum(
            dists[k].log_prob(action[k]) if k != "unit_cont_prob" else dists[k].log_prob(action[k]).masked_fill(~raw_unit_mask, 0).sum(1) for k in dists
        ).unsqueeze(1)

        dist_entropy = {
            k: dists[k].entropy().mean() if k != 'unit_cont_prob' else dists[k].entropy().masked_fill(~raw_unit_mask, 0).sum() / (~raw_unit_mask).sum() for k in dists
        }
        if not return_entropy_dist:
            dist_entropy = sum(dist_entropy.values())

        # if reference_dists is not None:
        #     KL = {
        #         k: distributions.kl_divergence(reference_dists[k], dists[k]).mean()
        #             if k != 'unit_cont_prob'
        #             else smoothed_Bernoulli_KL(reference_dists[k], dists[k]).masked_fill(~raw_unit_mask, 0).sum() / (~raw_unit_mask).sum()
        #         for k in dists
        #     }
        #     return value, action_log_probs, dist_entropy, rnn_hxs, dists, KL
        if reference_dists is not None:
            mask = inputs['inst_len'] > 0
            device = inputs['inst_len'].device
            if not mask.any():
                KL = {k: torch.tensor(0., device=device) for k in dists}
                return value, action_log_probs, dist_entropy, rnn_hxs, dists, KL
            KL = {}
            for k in dists:
                if k != 'unit_cont_prob':
                    tmp = distributions.kl_divergence(reference_dists[k], dists[k])
                    KL[k] = (tmp * mask).sum() / mask.sum()
                else:
                    raw_unit_mask.masked_fill_(~mask.unsqueeze(1), 0)
                    KL[k] = smoothed_Bernoulli_KL(reference_dists[k], dists[k]).masked_fill(~raw_unit_mask, 0).sum() / raw_unit_mask.sum()
            return value, action_log_probs, dist_entropy, rnn_hxs, dists, KL
                    

        return value, action_log_probs, dist_entropy, rnn_hxs, dists
    
    def Dict2Tensor(self, d):
 
        return torch.cat([d[k].view(d[k].size(0), -1) for k in self.ACTION_KEYS], dim=1)

    def Tensor2Dict(self, T):
        d = {}
        for i, k in enumerate(self.ACTION_KEYS):
            if k != 'unit_cont_prob':
                d[k] = T[:, i]
            else:
                d[k] = T[:, i:]
        return d

    @property
    def recurrent_hidden_state_size(self):
        return 1

    @property
    def is_recurrent(self):
        return False
