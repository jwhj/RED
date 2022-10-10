import os
import torch
from gym_minirts.utils import Args
from gym_minirts.wrappers import make_envs
from models.lang import Policy, ConvRnnCoachRL
from models.rule import RuleCoach
from models.random_coach import RandomCoach

import argparse


def to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, dict):
        return {key: to_device(batch[key], device) for key in batch}
    else:
        assert False, f'unsupported type {type(batch)}'


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--exper-name', type=str, required=True)
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--coach-name', default=None, choices=['None', 'rule-based', 'random'])
    parser.add_argument('--adv-coach', action='store_true')
    parser.add_argument('--dropout', default=0, type=int)
    parser.add_argument('--p', default=0, type=float)
    parser.add_argument('--save-to-db', type=str)
    args = parser.parse_args()

    device = 'cuda'
    # model = Policy(pretrained=True)
    # model.coach.max_raw_chars = 200
    model = Policy()
    model.to(device)
    N = 120
    m = 10

    exper_name = args.exper_name
    model_path = args.model_path
    coach_name = args.coach_name
    dropout = args.dropout
    save_to_db = args.save_to_db

    # dropout = 0
    level = ['medium', 'strong'][0]

    # if exper_name is not None and exper_name != 'switch' and exper_name != "None":
    #     model.base.inst_dict.init_inst_cache()
    #     distributed_para = torch.load(f'{model_dir}/{exper_name}/default/4001', map_location=device)
        
    #     if 'gail' in exper_name:
    #         distributed_para['state_dict'] = distributed_para['state_dict_net']

    #     model.base.load_state_dict(distributed_para['state_dict'])
    
    # else:
    #     if exper_name == 'switch':
    #         import copy
    #         rl_model = Policy()
    #         rl_model.to(device)

    #         rl_model.base.inst_dict.init_inst_cache()
    #         distributed_para = torch.load(f'{model_dir}/minirts-rl-pretrained/kind-enemy-6-v2-trail-3/default/4001', map_location=device)
    #         rl_model.base.load_state_dict(distributed_para['state_dict'])
    #     if exper_name == "None":
    #         pass
    if 'switch' in exper_name:
        rl_model = Policy()
        rl_model.to(device)
        rl_model.base.inst_dict.init_inst_cache()
        params = torch.load(args.model_path, map_location=device)
        rl_model.base.load_state_dict(params)
    else:
        model.base.inst_dict.init_inst_cache()
        params = torch.load(model_path, map_location=device)
        model.base.load_state_dict(params)



    if coach_name == 'rule-based':
        model.coach = RuleCoach(os.path.join(os.environ['MINIRTS_ROOT'], 'data/dataset/dict.pt'), num_games=N, kind_enemy=6)
        model.coach.inst_dict.init_inst_cache()
        model.coach.rule_type = 'r3' if not args.adv_coach else 'adv'
    elif coach_name == 'random':
        model.coach = RandomCoach(os.path.join(os.environ['MINIRTS_ROOT'], 'data/dataset/dict.pt'), p=args.p, max_raw_chars=200)
        model.coach.inst_dict.init_inst_cache()
        coach_name = coach_name + str(args.p)
    elif coach_name is not None:
        # coach_para = torch.load(f'/home/xss/distributed_minirts_results/{coach_name}/default/4001', map_location=device)
        # model.coach.load_state_dict(coach_para['state_dict'])

        model.coach = ConvRnnCoachRL.load(is_rnn=True).to(device)
        # coach_para = torch.load(f'/home/xss/distributed_minirts_results/{coach_name}/default/4001', map_location=device)
        # model.coach.load_state_dict(coach_para['state_dict'])

    model.coach.max_raw_chars = 200
    model.eval()


    print(f'matches_{level}/{exper_name}/{coach_name}.{dropout}.{model.coach.rule_type if isinstance(model.coach, RuleCoach) else False}')
    args = Args(
        num_thread=1, seed=42, game_per_thread=m, max_tick=int(12800), level=level, kind_enemy=6,
        frame_skip=50, single_action=True, save_dir=f'matches_{level}/{exper_name}/{coach_name}.{dropout}.{model.coach.rule_type if isinstance(model.coach, RuleCoach) else False}'
    )
    os.environ['LUA_PATH'] = os.path.join(args.lua_files, '?.lua')
    envs = make_envs(N, args)
    obs = envs.reset()
    if hasattr(model, 'generate_instructions'):
        with torch.no_grad():
            rnn_policy_state, inst = model.generate_instructions(obs, envs, dropout=dropout)

    a = 0
    b = 0
    c = 0
    utypes = [0] * 6
    # pbar = tqdm(total=N * m)
    win_list = []
    draw_list = []
    while a + b + c < N * m:
        with torch.no_grad():
            (
                value,
                action,
                action_log_prob,
                recurrent_hidden_states,
            ) = model.act(to_device(obs, device), None)
            if 'switch' in exper_name:
                (
                    _,
                    action_rl,
                    _,
                    _,
                ) = rl_model.act(to_device(obs, device), None)
                action = torch.where(inst==-1, action_rl, action)
        obs, reward, done, infos = envs.step(action)
        if hasattr(model, 'generate_instructions'):
            with torch.no_grad():
                rnn_policy_state, inst = model.generate_instructions(obs, envs, infos, dropout=dropout, policy_state=rnn_policy_state)
        for i, info in enumerate(infos):
            if 'episode' in info.keys():
                r = info['episode']['r']
                if r > 0:
                    a += 1
                    utypes[i % 6] += 1
                    win_list.append(i)
                elif r < 0:
                    b += 1
                else:
                    c += 1
                    draw_list.append(i)
                # pbar.update(1)
                # pbar.set_description(f'win: {a} loss: {b}, draw: {c}, win list: {win_list}, draw list: {draw_list}')
                # print(f'win: {a} loss: {b}, draw: {c}')
                if (a + b + c > 0 and (a + b + c) % N == 0):
                    if not (a + b + c == N * m):
                        obs = envs.reset()
                        if hasattr(model, 'generate_instructions'):
                            if isinstance(model.coach, RuleCoach):
                                model.coach.hist_enemy = [None for _ in range(N)]
                            with torch.no_grad():
                                rnn_policy_state, inst = model.generate_instructions(obs, envs, dropout=dropout, policy_state=rnn_policy_state)
                    else:
                        obs = envs.reset(need_obs=False)
                    print(f'win: {a} loss: {b}, draw: {c}')
                    print(sorted(win_list))
                    print(utypes)
                    # print(draw_list)
                    win_list = []
                    draw_list = []
                    print('reset')
    rate = a/(a + b + c)*100
    print(f'{rate:.2f}%')
    if save_to_db is not None:
        import lmdb
        env = lmdb.open(save_to_db)
        with env.begin(write=True) as txn:
            key = f'matches_{level}/{exper_name}/{coach_name}.{dropout}.{model.coach.rule_type if isinstance(model.coach, RuleCoach) else False}'
            txn.put(key.encode(), str(rate).encode())


if __name__ == '__main__':
    main()
