# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import os
import pickle
import pprint
import requests
from gym_minirts.utils import Args, get_game_option, get_ai_options
from gym_minirts.wrappers import make_envs
from models.lang import ExecutorRNNSingle, Policy
from models.rule import RuleCoach

import torch
import tube
import pytube
import minirts
from executor_wrapper import ExecutorWrapper
from executor import Executor
from common_utils import to_device


def create_game(ai1_option, ai2_option, game_option, unit_type, *, act_name='act', port=8002):
    print('ai1 option:')
    print(ai1_option.info())
    print('ai2 option:')
    print(ai2_option.info())
    print('game option:')
    print(game_option.info())
    
    idx2utype = [
        minirts.UnitType.SPEARMAN,
        minirts.UnitType.SWORDMAN,
        minirts.UnitType.CAVALRY,
        minirts.UnitType.DRAGON,
        minirts.UnitType.ARCHER,
        minirts.UnitType.CATAPULT
    ]

    act_dc = tube.DataChannel(act_name, 1, -1)
    context = tube.Context()
    g = minirts.RTSGame(game_option)
    bot1 = minirts.CheatExecutorAI(ai1_option, 0, None, act_dc)
    bot2 = minirts.MediumAI(ai2_option, 0, None, idx2utype[unit_type], False)
    g.add_bot(bot1)
    g.add_bot(bot2)
    g.add_default_spectator(port)

    context.push_env_thread(g)
    return context, act_dc


def parse_args():
    parser = argparse.ArgumentParser(description='human coach')
    # parser.add_argument('--save_dir', type=str, default='')
    parser.add_argument('--seed', type=int, default=1)
    # parser.add_argument('--deterministic', action='store_true')
    # parser.add_argument('--num_thread', type=int, default=1)
    # parser.add_argument('--batchsize', type=int, default=1)
    parser.add_argument('--gpu', type=int, default=0)
    # parser.add_argument('--update_per_epoch', type=int, default=200)
    # parser.add_argument('--num_epoch', type=int, default=400)

    default_lua = os.path.join(os.environ['MINIRTS_ROOT'], 'game/game_MC/lua')
    # print(default_lua)
    # assert False
    parser.add_argument('--lua_files', type=str, default=default_lua)

    # ai1 option
    parser.add_argument('--frame_skip', type=int, default=50)
    parser.add_argument('--fow', type=int, default=1)
    # parser.add_argument('--t_len', type=int, default=10)
    parser.add_argument('--use_moving_avg', type=int, default=1)
    parser.add_argument('--moving_avg_decay', type=float, default=0.98)
    parser.add_argument('--num_resource_bins', type=int, default=11)
    parser.add_argument('--resource_bin_size', type=int, default=50)
    parser.add_argument('--max_num_units', type=int, default=50)
    parser.add_argument('--num_prev_cmds', type=int, default=25)
    parser.add_argument('--num_instructions', type=int, default=1,
                        help="not used in human coach, > 0 is sufficient")
    parser.add_argument('--max_raw_chars', type=int, default=200)
    parser.add_argument('--verbose', action='store_true')

    # game option
    parser.add_argument('--max_tick', type=int, default=int(2e5))
    parser.add_argument('--no_terrain', action='store_true')
    parser.add_argument('--resource', type=int, default=500)
    parser.add_argument('--resource_dist', type=int, default=4)
    parser.add_argument('--fair', type=int, default=0)
    parser.add_argument('--save_replay_freq', type=int, default=0)
    parser.add_argument('--save_replay_per_games', type=int, default=1)

    # model
    parser.add_argument(
        '--model_path',
        type=str,
        default='../../pretrained_models/executor_rnn.pt'
    )

    # used in game interface
    parser.add_argument('--port', type=int, default=8002)
    parser.add_argument('--unit-type', type=int)
    parser.add_argument('--save-dir', type=str)

    # a match that should be recorded
    parser.add_argument('--username', type=str)
    parser.add_argument('--model-type', type=str)
    parser.add_argument('--match-id', type=int)
    parser.add_argument('--bc-path', type=str)

    args = parser.parse_args()

    args.model_path = os.path.abspath(args.model_path)
    # if not os.path.exists(args.model_path):
    #     print('cannot find model at:', args.model_path)
    #     assert False

    return args


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
    game_option.num_games_per_thread = 1
    # !!! this is important
    game_option.max_num_units_per_player = args.max_num_units

    if args.save_dir is not None:
        save_dir = os.path.abspath(args.save_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        game_option.save_replay_prefix = save_dir + '/'
    return game_option


def get_ai_option(args):
    ai1_option = minirts.AIOption()
    ai1_option.t_len = 1; 
    ai1_option.fs = args.frame_skip
    ai1_option.fow = args.fow
    ai1_option.use_moving_avg = args.use_moving_avg
    ai1_option.moving_avg_decay = args.moving_avg_decay
    ai1_option.num_resource_bins = args.num_resource_bins
    ai1_option.resource_bin_size = args.resource_bin_size
    ai1_option.max_num_units = args.max_num_units
    ai1_option.num_prev_cmds = args.num_prev_cmds
    ai1_option.num_instructions = args.num_instructions
    ai1_option.max_raw_chars = args.max_raw_chars
    ai1_option.verbose = args.verbose

    ai2_option = minirts.AIOption()
    ai2_option.fs = args.frame_skip
    ai2_option.fow = args.fow
    return ai1_option, ai2_option
    

if __name__ == '__main__':
    args = parse_args()
    print('args:')
    pprint.pprint(vars(args))

    os.environ['LUA_PATH'] = os.path.join(args.lua_files, '?.lua')
    print('lua path:', os.environ['LUA_PATH'])

    device = torch.device('cuda:%d' % args.gpu)

    model = Policy()
    model.to(device)

    executor = model.base
    # exper_name = 'distributed_minirts_iter/no-coach-with-distl-hist-inst-none-v10'
    # distributed_para = torch.load(f'/home/xss/distributed_minirts_results/{exper_name}/default/4001', map_location=device)
    # executor.load_state_dict(distributed_para['state_dict'])
    params = torch.load(args.model_path, map_location=device)
    executor.load_state_dict(params)
    model.eval()

    if args.bc_path is not None:
        with open(args.bc_path + '.params', 'rb') as f:
            hparams = pickle.load(f)
        executor_bc = ExecutorRNNSingle(**hparams).to(device)
        params = torch.load(args.bc_path, map_location=device)
        executor_bc.load_state_dict(params)
        executor_bc.eval()
    else:
        executor_bc = None

    # print('top 500 insts')
    # for inst  in executor.inst_dict._idx2inst[:500]:
    #     print(inst)
    executor_wrapper = ExecutorWrapper(
        None, executor, args.num_instructions, args.max_raw_chars, False)
    executor_wrapper.train(False)

    game_option = get_game_option(args)
    ai1_option, ai2_option = get_ai_option(args)
    context, act_dc = create_game(ai1_option, ai2_option, game_option, unit_type=args.unit_type, port=args.port)
    context.start()
    dc = pytube.DataChannelManager([act_dc])
    while not context.terminated():
        data = dc.get_input(max_timeout_s=1)
        if len(data) == 0:
            continue
        data = data['act']
        reward = data['reward'].item()
        if reward != 0 and args.match_id is not None:
            assert args.model_type is not None
            assert args.match_id is not None
            requests.get(f'http://localhost:8000/update_match_result?username={args.username}&model_type={args.model_type}&match_id={args.match_id}&result={str(reward)}')
        data = to_device(data, device)
        # import IPython
        # IPython.embed()
        with torch.no_grad():
            reply = executor_wrapper.forward(data, executor_bc)

        # reply = {key : reply[key].detach().cpu() for key in reply}
        dc.set_reply('act', reply)
        print('===end of a step===')
    dc.terminate()
    del act_dc
    del dc
    del context

    # import IPython
    # IPython.embed()
