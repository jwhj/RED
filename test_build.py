from typing import Dict
import argparse
import os
import sys
import torch
import numpy as np
from tqdm import tqdm
import lmdb
from gym_minirts.utils import Args, get_game_option, get_ai_options
from gym_minirts.wrappers import make_envs

sys.path.append(os.path.join(os.environ['MINIRTS_ROOT'], 'scripts/behavior_clone'))
from utils import convert_to_raw_instruction
import common_utils.global_consts as gc

from models.lang import Policy
from models.rule import RuleCoach


def format_instruction(inst: str, inst_dict, device):
    inst_idx = torch.zeros((1,), device=device).long()
    inst_idx[0] = inst_dict.get_inst_idx(inst)
    inst_cont = torch.zeros((1,), device=device).long()

    raw_inst = convert_to_raw_instruction(inst, 200)
    inst, inst_len = inst_dict.parse(inst, True)
    inst = torch.tensor(inst, dtype=torch.long, device=device).unsqueeze(0)
    inst_len = torch.tensor([inst_len], dtype=torch.long, device=device)
    raw_inst = torch.tensor([raw_inst], dtype=torch.long, device=device)

    reply = {
        'inst': inst_idx.unsqueeze(1),
        'inst_pi': torch.ones(1, 500),
        'cont': inst_cont.unsqueeze(1),
        'cont_pi': torch.ones(1, 2, device=device) / 2,
        'raw_inst': raw_inst
    }
    return inst, inst_len, inst_cont.unsqueeze(1), reply

def to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, dict):
        return {key: to_device(batch[key], device) for key in batch}
    else:
        assert False, f'unsupported type {type(batch)}'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exper-name', type=str, required=True)
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cmd-idx', type=int, default=4)
    parser.add_argument('--save-to-db', type=str)
    args = parser.parse_args()
    return args

def main(args):
    device = 'cuda'
    model = Policy()
    model.to(device)
    model.coach.max_raw_chars = 200

    model.base.inst_dict.init_inst_cache()
    model.eval()
    if 'switch' not in args.exper_name:
        distributed_para = torch.load(args.model_path)
        model.base.load_state_dict(distributed_para)
        rl_model = None
    else:
        rl_model = Policy()
        rl_model.to(device)
        rl_model.coach.max_raw_chars = 200
        rl_model.base.inst_dict.init_inst_cache()
        distributed_para = torch.load(args.model_path)
        rl_model.base.load_state_dict(distributed_para)
        rl_model.eval()


    # if 'gail' in exper_name or 'airl' in exper_name:
    #     distributed_para['state_dict'] = distributed_para['state_dict_net']

    # model.base.load_state_dict(distributed_para['state_dict'])

    # exper_name='lang_mixer_pretrain'
    # exper_name = 'attn-iter-bc'
    # path = '/mnt/shared/minirts/scripts/behavior_clone/saved_models/lang_mixer/best_checkpoint.pt'
    # with open(path + '.params', 'rb') as f:
    #     params = pickle.load(f)

    # model.base = ExecutorRNNSingle(**params, use_lang_mixer=True).to(device)
    # model.base.load_state_dict(torch.load('/mnt/shared/attn-iter-bc', map_location=device)['state_dict'])
    dropout = 0

    # model.coach = RuleCoach(
    #     os.path.join(os.environ['MINIRTS_ROOT'], 'data/dataset/dict.pt')
    # )
    # model.coach.inst_dict.init_inst_cache()

    # print(f'matches/{exper_name}.{dropout}')
    n_steps = 35
    N = 600
    m = 2
    game_args = Args(
        num_thread=1,
        seed=args.seed,
        game_per_thread=m,
        max_tick=50*n_steps,
        level='medium',
        frame_skip=50,
        single_action=True,
        kind_enemy=6,
        save_dir='matches/test',
    )
    os.environ['LUA_PATH'] = os.path.join(game_args.lua_files, '?.lua')
    envs = make_envs(N, game_args)

    cmds = [
        ('build a barrack', 'build a spearman', gc.UnitTypes.SPEARMAN.value),
        ('build a blacksmith','build a swordman', gc.UnitTypes.SWORDMAN.value),
        ('build a stable', 'build a cavalry', gc.UnitTypes.CAVALRY.value),
        ('build a workshop', 'build a dragon', gc.UnitTypes.DRAGON.value),
        ('build a workshop', 'build an archer', gc.UnitTypes.ARCHER.value),
        ('build a workshop', 'build a catapult', gc.UnitTypes.CATAPULT.value),
        ('construire un atelier', 'construire un archer', gc.UnitTypes.ARCHER.value),
        ('построить мастерскую', 'построить лучника', gc.UnitTypes.ARCHER.value),
        ('建造车间', '造弓箭手', gc.UnitTypes.ARCHER.value),
        ('we need more towers', 'we need more towers', gc.UnitTypes.GUARD_TOWER.value)
    ]
    # build_type = [np.random.randint(0, len(cmds)) for _ in range(N)]
    build_type = [args.cmd_idx] * N
    # inst, unit_type = [
    #     (26, gc.UnitTypes.BLACKSMITH.value),
    #     (5, gc.UnitTypes.STABLE.value),
    #     (15, gc.UnitTypes.BARRACK.value),
    # ][0]
    obs = envs.reset()
    if hasattr(model, 'generate_instructions'):
        with torch.no_grad():
            # model.generate_instructions(obs, envs, dropout=dropout)
            insts = []
            inst_lens = []
            inst_conts = []
            for i, env in enumerate(envs.envs):
                inst, inst_len, inst_cont, reply = format_instruction('all peasant mine', model.base.inst_dict, device)
                insts.append(inst)
                inst_lens.append(inst_len)
                inst_conts.append(inst_cont)
                env.set_coach_reply(reply)
            obs['inst'] = torch.cat(insts, dim=0)
            obs['inst_len'] = torch.cat(inst_lens, dim=0)
            obs['inst_cont'] = torch.cat(inst_conts, dim=0)
    pbar = tqdm()
    army_count = [0] * N
    results = []
    while True:
        # if pbar.n < 5:
        #     inst = 'build a blacksmith'
        # elif pbar.n <30:
        #     inst = 'build a swordman'
        # else:
        #     inst = 'go around the map'
        with torch.no_grad():
            (
                value,
                action,
                action_log_prob,
                recurrent_hidden_states,
            ) = model.act(to_device(obs, device), None)
            if 'switch' in args.exper_name:
                (
                    _,
                    action_rl,
                    _,
                    _,
                ) = rl_model.act(to_device(obs, device), None)
                action = torch.where(obs['inst_len'].unsqueeze(1) == 0, action_rl, action)
        obs, reward, done, infos = envs.step(action)
        pbar.update()
        if done.all():
            m -= 1
            results.extend(army_count)
            pbar.close()
            pbar = tqdm()
            if m > 0:
                obs = envs.reset()
                if hasattr(model, 'generate_instructions'):
                    with torch.no_grad():
                        # model.generate_instructions(obs, envs, dropout=dropout)
                        insts = []
                        inst_lens = []
                        inst_conts = []
                        for i, env in enumerate(envs.envs):
                            inst, inst_len, inst_cont, reply = format_instruction('all peasant mine', model.base.inst_dict, device)
                            insts.append(inst)
                            inst_lens.append(inst_len)
                            inst_conts.append(inst_cont)
                            env.set_coach_reply(reply)
                        obs['inst'] = torch.cat(insts, dim=0)
                        obs['inst_len'] = torch.cat(inst_lens, dim=0)
                        obs['inst_cont'] = torch.cat(inst_conts, dim=0)
                continue
            else:
                envs.reset(need_obs=False)
                break
        if hasattr(model, 'generate_instructions'):
            with torch.no_grad():
                insts = []
                inst_lens = []
                inst_conts = []
                for i, env in enumerate(envs.envs):
                    if pbar.n < 10:
                        cmd = cmds[build_type[i]][0]
                    elif pbar.n < 30:
                        cmd = cmds[build_type[i]][1]
                    else:
                        cmd = ''
                    if pbar.n == 30:
                        army_count[i] = torch.sum(obs['army_type'][i] == cmds[build_type[i]][2]).item()
                    inst, inst_len, inst_cont, reply = format_instruction(cmd, model.base.inst_dict, device)
                    insts.append(inst)
                    inst_lens.append(inst_len)
                    inst_conts.append(inst_cont)
                    env.set_coach_reply(reply)
                obs['inst'] = torch.cat(insts, dim=0)
                obs['inst_len'] = torch.cat(inst_lens, dim=0)
                obs['inst_cont'] = torch.cat(inst_conts, dim=0)
    pbar.close()
    results = np.array(results)
    rate = np.mean(results > 0)
    if args.save_to_db is not None:
        env = lmdb.open(args.save_to_db)
        with env.begin(write=True) as txn:
            txn.put('{} {} {}'.format(args.exper_name, args.cmd_idx, args.seed).encode(), str(rate).encode())
    print('{} {} seed={} {}'.format(args.exper_name, args.cmd_idx, args.seed, rate))


if __name__ == '__main__':
    args = get_args()
    main(args)
