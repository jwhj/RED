from typing import List, Dict
import os
import uuid
import torch
import gym
from gym.spaces import MultiDiscrete, MultiBinary
from gym_minirts.set_path import append_sys_path
from gym_minirts.utils import format_single_action, format_single_unit, num_units, action_dim, sizes

append_sys_path()
import tube
from pytube import DataChannelManager
import minirts


class MiniRTSEnv(gym.Env):
    def __init__(
        self, idx, context, game_option, ai1_option, ai2_option, level: str = "medium", *, single_action=False, kind_enemy=5,
    ):
        self.idx = idx
        self.context = context
        self.game_option = game_option
        self.ai1_option = ai1_option
        self.ai2_option = ai2_option
        self.num_steps = 0
        self.num_army = None
        self.coach_reply = {}
        self.action_space_ = None
        self.current_obs = None
        self.is_inst_complete = False
        self.single_action = single_action
        
        self.prev_glob_feats = []
        self.prev_glob_score = []

        special_id = str(uuid.uuid4())
        self.act_dc = tube.DataChannel(f'act{self.idx}' + special_id, 1, 1)

        idx2utype = [
            minirts.UnitType.SPEARMAN,
            minirts.UnitType.SWORDMAN,
            minirts.UnitType.CAVALRY,
            minirts.UnitType.DRAGON,
            minirts.UnitType.ARCHER,
            # minirts.UnitType.CATAPULT
        ]
        if kind_enemy == 6:
            idx2utype.append(minirts.UnitType.CATAPULT)

        g_option = minirts.RTSGameOption(self.game_option)
        g_option.seed = self.game_option.seed + self.idx
        if self.game_option.save_replay_prefix:
            g_option.save_replay_prefix = self.game_option.save_replay_prefix + str(
                self.idx
            )

        g = minirts.RTSGame(g_option)
        bot1 = minirts.CheatExecutorAI(self.ai1_option, 0, None, self.act_dc)
        utype = idx2utype[self.idx % len(idx2utype)]
        if level == 'simple':
            bot2 = minirts.SimpleAI(self.ai2_option, 0, utype)
        elif level == 'medium':
            bot2 = minirts.MediumAI(self.ai2_option, 0, None, utype, False)
        elif level == 'strong':
            bot2 = minirts.StrongAI(self.ai2_option, 0)
        else:
            raise RuntimeError('level must be one of [simple, medium, strong]')
        g.add_bot(bot1)
        g.add_bot(bot2)
        self.context.push_env_thread(g)

    @property
    def action_space(self):
        if self.action_space_ is None:
            if not self.single_action:
                lst = [2, num_units]
                for key in sizes:
                    lst.append(sizes[key])
                self.action_space_ = MultiDiscrete(lst)
            else:
                self.action_space_ = [
                    MultiDiscrete([2] + [v for v in sizes.values()]),
                    MultiBinary(num_units)
            ]
        return self.action_space_

    def get_input(self):
        data = self.act_dc.get_input()
        reward = data['reward'].item()
        done = reward != 0
        self.num_army = data['num_army']
        info = {
            "frame_idx": self.num_steps * self.ai1_option.fs
        }
        if done:
            if self.num_steps * self.ai1_option.fs == self.game_option.max_tick:
                reward = 0
            info['episode'] = {'r': reward}
            info['num_steps'] = self.num_steps
            self.num_steps = 0
        return data, reward, done, info

    def make_them_happy(self, reply: Dict[str, torch.Tensor]):
        bsz = reply['cmd_type_prob'].size(0)
        reply['cont'] = torch.zeros((bsz, 1), dtype=torch.long)
        reply['cont_pi'] = torch.zeros((bsz, 2))
        reply['inst'] = torch.zeros((bsz, 1), dtype=torch.long)
        reply['inst_pi'] = torch.zeros((bsz, self.ai1_option.num_instructions))
        reply['raw_inst'] = torch.zeros((bsz, 200), dtype=torch.long)
        reply['num_unit'] = self.num_army

    def set_coach_reply(self, coach_reply: Dict[str, torch.Tensor]):
        for key in coach_reply:
            self.coach_reply[key] = coach_reply[key].clone()
    
    def set_prev_glob_feat(self, glob_feat: torch.Tensor, is_complete: bool, score=None):
        self.is_inst_complete = is_complete
        assert self.coach_reply
        if self.coach_reply['inst'].item() == self.current_obs['prev_inst'].item():
            self.prev_glob_feats.append(glob_feat)
            if score is not None:
                self.prev_glob_score.append(score)
                raw_inst_len = (self.coach_reply['raw_inst'] != -1).sum()
                raw_inst_len = min(193, raw_inst_len)
                str_score = list((" " + str(round(score.item(), 4))).encode())
                self.coach_reply['raw_inst'][:, raw_inst_len: raw_inst_len + len(str_score)] = torch.LongTensor(str_score).to(self.coach_reply['raw_inst'].device)
        else:
            self.prev_glob_feats = []
            self.prev_glob_score = []
            self.is_inst_complete = False
    
    def get_prev_glob_feat(self):
        return self.prev_glob_feats[-19:]
    
    def get_prev_score(self):
        if self.prev_glob_score:
            return max(self.prev_glob_score)
        else:
            return 0
    
    def get_is_inst_complete(self):
        return self.is_inst_complete

    def step(self, action: torch.Tensor):
        # __import__('ipdb').set_trace()
        self.num_steps += 1
        reply = format_single_unit(action) if not self.single_action else format_single_action(action)
        if len(self.coach_reply)==0:
            self.make_them_happy(reply)
        else:
            for key in self.coach_reply:
                reply[key] = self.coach_reply[key]
            reply['num_unit'] = self.num_army
            self.coach_reply = {}
        self.act_dc.set_reply(reply)
        data, reward, done, info  = self.get_input()
        self.current_obs = data
        return data, reward, done, info

    def reset(self, need_obs=True):
        assert self.num_steps == 0, 'resetting in the middle of a game is not supported'
        self.num_steps = 0
        self.is_inst_complete = False
        self.prev_glob_feats = []
        self.prev_glob_score = []

        if self.num_army is not None:
            action = torch.zeros((1, action_dim + 1 if not self.single_action else action_dim + num_units), dtype=torch.long)
            reply = format_single_unit(action) if not self.single_action else format_single_action(action)
            self.make_them_happy(reply)
            self.act_dc.set_reply(reply)
        if need_obs:
            self.current_obs, _, _, _ = self.get_input()
            return self.current_obs
        else:
            return self.current_obs

    def render(self):
        raise NotImplemented

    def close(self):
        self.act_dc.terminate()
        raise NotImplemented
