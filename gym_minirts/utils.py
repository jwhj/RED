from typing import Dict, List
import os
import torch
from torch.functional import F

from gym_minirts.set_path import append_sys_path

append_sys_path()
import tube
import minirts

num_units = 50
num_types = 16
num_loc = 1024
sizes = {
    'cmd_type_prob': 7,
    'gather_idx_prob': num_units,
    'attack_idx_prob': num_units,
    'unit_type_prob': num_types,
    'building_type_prob': num_types,
    'building_loc_prob': num_loc,
    'move_loc_prob': num_loc,
}
action_dim = 1 + len(sizes)


class Args:
    seed = 1
    num_thread = 1
    game_per_thread = -1
    lua_files: str
    frame_skip = 50
    fow = 1
    use_moving_avg = 1
    moving_avg_decay = 0.98
    num_resource_bins = 11
    resource_bin_size = 50
    max_num_units = 50
    num_prev_cmds = 25
    max_raw_chars = 200
    verbose = False
    max_tick = int(2e5)
    no_terrain = False
    resource = 500
    resource_dist = 4
    fair = 0
    save_replay_freq = 0
    save_replay_per_games = 1
    save_dir = 'matches/dev'
    cheat = 0
    level = 'medium'
    kind_enemy = 5
    single_action = False

    def __init__(self, **kwargs):
        # root = '/workspace/minirts'
        root = os.environ['MINIRTS_ROOT']
        self.lua_files = os.path.join(root, 'game/game_MC/lua')
        for key in kwargs:
            setattr(self, key, kwargs[key])


def get_obs(obs, idx):
    return {key: obs[key][idx] for key in obs}


def make_batch(obs_list):
    if isinstance(obs_list[0], torch.Tensor):
        return torch.cat(obs_list, dim=0)
    elif isinstance(obs_list[0], dict):
        return {key: make_batch([obs[key] for obs in obs_list]) for key in obs_list[0]}
    else:
        assert False, f'unsupported type: {type(obs_list[0])}'


def to_tensors(reply: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
    raise NotImplemented
    result = [reply['glob_cont_prob']]
    for key in sizes:
        assert num_units == reply[key].size(1)
        result.extend(reply[key][:, i] for i in range(reply[key].size(1)))
    return result


@torch.no_grad()
def format_old(
    # glob_cont: torch.LongTensor, x: torch.LongTensor, building_type: torch.LongTensor
    action: torch.LongTensor,
) -> Dict[str, torch.Tensor]:
    # # action: [bsz, 1+nu+nu+...]
    # reply = {}
    # reply['glob_cont_prob'] = F.one_hot(action[:, 0], 2).float()
    # i = 1
    # for key in sizes:
    #     reply[key] = F.one_hot(action[:, i : i + num_units], sizes[key]).float()
    #     i += num_units
    # return reply

    # action: [bsz, 1+1+len(sizes)]
    action = action.cpu()
    bsz = action.size(0)
    reply: Dict[str, torch.Tensor] = {}
    reply['glob_cont_prob'] = F.one_hot(action[:, 0], 2).float()
    for key in sizes:
        reply[key] = torch.zeros((bsz, num_units, sizes[key]))
    reply['cmd_type_prob'][:, :, 6].fill_(1)
    reply['cmd_type_prob'][:, :, 6].scatter_(1, action[:, 1].unsqueeze(1), 0)
    for i, key in enumerate(sizes):
        idx = action[:, 1] * sizes[key] + action[:, 2 + i]
        reply[key].view(bsz, -1).scatter_(1, idx.unsqueeze(1), 1)
        assert (reply[key].sum(-1) >= 0).all().item(), (key, reply[key])
    return reply

@torch.no_grad()
def format_single_action(
    action: torch.LongTensor
) -> Dict[str, torch.Tensor]:

    ACTION_KEYS = ['cmd_type_prob', 'gather_idx_prob', 'attack_idx_prob', 
        'unit_type_prob', 'building_type_prob', 'building_loc_prob', 'move_loc_prob']

    action = action.long().cpu()
    bsz = action.size(0)
    reply: Dict[str, torch.Tensor] = {}
    reply['glob_cont_prob'] = F.one_hot(action[:, 0], 2).float()

    for i, key in enumerate(ACTION_KEYS):

        cur_reply = F.one_hot(action[:, 1 + i], sizes[key]).float()

        if key == 'cmd_type_prob':
            reply[key] = torch.zeros((bsz, num_units, sizes[key]))
            reply['cmd_type_prob'][:, :, 6].fill_(1)
            
            unit_is_cont = action[:, -num_units:].unsqueeze(-1)

            reply['cmd_type_prob'] = torch.where(
                unit_is_cont.repeat(1, 1, 7).bool(),
                reply['cmd_type_prob'],
                cur_reply.repeat(1, num_units, 1),
                )
        else:
            reply[key] = cur_reply.unsqueeze(1).repeat(1, num_units, 1)

    return reply

@torch.no_grad()
def format_single_unit(
    action: torch.LongTensor,
) -> Dict[str, torch.Tensor]:
    action = action.cpu()
    bsz = action.size(0)
    reply: Dict[str, torch.Tensor] = {}
    reply['glob_cont_prob'] = F.one_hot(action[:, 0], 2).float()
    for i, key in enumerate(sizes):
        reply[key] = torch.zeros((bsz, num_units, sizes[key]))

        if key == 'cmd_type_prob':
            reply['cmd_type_prob'][:, :, 6].fill_(1)
        else:
            reply[key] += 1e-5

        cur_reply = F.one_hot(action[:, 2 + i], sizes[key]).float()
        reply[key].scatter_(
            1, action[:, 1].view(-1, 1, 1).repeat(1, 1, sizes[key]),
            cur_reply.unsqueeze(1)
        )    
        assert (reply[key].sum(-1) >= 0).all().item(), (key, reply[key])

    return reply


def get_game_option(args):
    game_option = minirts.RTSGameOption()
    game_option.seed = args.seed
    game_option.max_tick = args.max_tick
    game_option.no_terrain = args.no_terrain
    game_option.resource = args.resource
    game_option.resource_dist = args.resource_dist
    game_option.fair = args.fair
    game_option.save_replay_freq = args.save_replay_freq
    game_option.save_replay_per_games = args.save_replay_per_games
    game_option.lua_files = args.lua_files
    game_option.num_games_per_thread = args.game_per_thread
    # !!! this is important
    game_option.max_num_units_per_player = args.max_num_units

    if args.save_dir is not None:
        save_dir = os.path.abspath(args.save_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        game_option.save_replay_prefix = save_dir + '/'
    else:
        game_option.save_replay_prefix = ""

    return game_option


def get_ai_options(args, num_instructions):
    options = []
    for i in range(2):
        ai_option = minirts.AIOption()
        ai_option.t_len = 1
        ai_option.fs = args.frame_skip
        ai_option.fow = args.fow
        ai_option.use_moving_avg = args.use_moving_avg
        ai_option.moving_avg_decay = args.moving_avg_decay
        ai_option.num_resource_bins = args.num_resource_bins
        ai_option.resource_bin_size = args.resource_bin_size
        ai_option.max_num_units = args.max_num_units
        ai_option.num_prev_cmds = args.num_prev_cmds
        ai_option.num_instructions = num_instructions
        ai_option.max_raw_chars = args.max_raw_chars
        ai_option.verbose = args.verbose
        options.append(ai_option)

    return options[0], options[1]
