import os
import sys
import argparse
import pickle
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from gym_minirts.utils import Args, get_game_option, get_ai_options
from gym_minirts.wrappers import make_envs

# from minirts_models.policy import Policy
# from models.single_action_inherit import Policy
from models.lang import ExecutorRNNSingle, Policy
from models.rule import RuleCoach

plt.style.use('ggplot')
matplotlib.rcParams.update({'font.size': 12})

MINITS_SCRIPTS = os.path.join(os.environ['MINIRTS_ROOT'], 'scripts/behavior_clone')
if MINITS_SCRIPTS not in sys.path:
    sys.path.append(MINITS_SCRIPTS)
from dataset import BehaviorCloneDataset
import common_utils.global_consts as gc

data_path = 'categorized_acc.pt'


def to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, dict):
        return {key: to_device(batch[key], device) for key in batch}
    else:
        assert False, f'unsupported type {type(batch)}'


def load_checkpoint(path: str, device):
    model = Policy()
    model.to(device)
    model.coach.max_raw_chars = 200
    model.base.inst_dict.init_inst_cache()

    params = torch.load(path, map_location=device)
    if 'state_dict' in params:
        params = params['state_dict']
    elif 'state_dict_net' in params:
        params = params['state_dict_net']

    model.base.load_state_dict(params)
    return model


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--compute', action='store_true')
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    return args


def compute_acc(executor, loader, keywords, device, use_tqdm=True):
    results = [[] for _ in range(len(keywords))]
    val_acc = []
    pbar = tqdm(loader) if use_tqdm else loader
    for batch in pbar:
        batch = to_device(batch, device)
        insts = executor.inst_dict.deparse(batch['inst'])
        cmd_type = batch['target_cmds']['cmd_type']
        bsz, nu = cmd_type.shape[:2]
        unit_mask = torch.tensor(range(nu), device=device).unsqueeze(0) < batch[
            'my_units'
        ]['num_units'].unsqueeze(1)
        # non_cont_mask = cmd_type != gc.CmdTypes.CONT.value
        # mask = unit_mask * non_cont_mask
        _, probs, _, _ = executor(
            batch,
            unit_is_cont=(cmd_type == gc.CmdTypes.CONT.value),
            need_preprocess=False,
        )

        glob_cont_eq = (
            batch['glob_cont'] == torch.argmax(probs['glob_cont_prob'], dim=1)
        ).unsqueeze(1)

        _cmd_type = torch.argmax(probs['cmd_type_prob'], dim=1, keepdim=True)

        cmd_type_eq = cmd_type == _cmd_type

        gather_idx_eq = batch['target_cmds']['target_gather_idx'] == torch.argmax(
            probs['gather_idx_prob'], dim=1
        ).unsqueeze(1)
        attack_idx_eq = batch['target_cmds']['target_attack_idx'] == torch.argmax(
            probs['attack_idx_prob'], dim=1
        ).unsqueeze(1)

        building_type_eq = batch['target_cmds']['target_type'] == torch.argmax(
            probs['building_type_prob'], dim=1
        ).unsqueeze(1)
        building_loc_eq = (
            batch['target_cmds']['target_y'] * gc.MAP_X
            + batch['target_cmds']['target_x']
        ) == torch.argmax(probs['building_loc_prob'], dim=1).unsqueeze(1)
        building_eq = building_type_eq * building_loc_eq

        unit_eq = batch['target_cmds']['target_type'] == torch.argmax(
            probs['unit_type_prob'], dim=1
        ).unsqueeze(1)

        move_eq = (
            batch['target_cmds']['target_y'] * gc.MAP_X
            + batch['target_cmds']['target_x']
        ) == torch.argmax(probs['move_loc_prob'], dim=1).unsqueeze(1)

        ones = torch.ones_like(gather_idx_eq)
        action_output_eq = torch.gather(
            torch.stack(
                [
                    ones,
                    gather_idx_eq,
                    attack_idx_eq,
                    building_eq,
                    unit_eq,
                    move_eq,
                    ones,
                ],
                dim=0,
            ),
            0,
            cmd_type.unsqueeze(0),
        ).squeeze(0)

        eq = glob_cont_eq * (
            batch['glob_cont'].unsqueeze(1)
            + (1 - batch['glob_cont'].unsqueeze(1)) * cmd_type_eq * action_output_eq
        )
        eq = eq.masked_fill(~unit_mask, False).any(dim=1).float()
        # all_cont_mask = torch.logical_or(
        #     cmd_type == gc.CmdTypes.CONT.value, ~unit_mask
        # ).all(dim=1)
        # eq.masked_fill_(all_cont_mask, -1)
        val_acc.extend(filter(lambda x: x >= 0, eq.tolist()))
        for i in range(bsz):
            inst = insts[i]
            # if all_cont_mask[i].item():
            #     continue
            for j, keyword_list in enumerate(keywords):
                for keyword in keyword_list:
                    if keyword in inst:
                        results[j].append(eq[i].item())
                        break
    return val_acc, results


def compute():
    device = 'cuda'
    model_best = load_checkpoint('ckpt/red-1.pt', device)
    model_bc = load_checkpoint(
        'ckpt/executor_rnn_with_cont_bsz128/best_checkpoint.pt',
        device,
    )

    dataset = BehaviorCloneDataset(
        os.path.join(os.environ['MINIRTS_ROOT'], 'data/dataset/val.json'),
        11,
        50,
        25,
        inst_dict=model_best.base.inst_dict,
        word_based=True,
    )
    loader = DataLoader(dataset, 16, False, num_workers=20)

    keywords = [
        ['attack', 'kill'],
        ['defend'],
        ['build', 'create', 'make'],
        ['mine', 'mining', 'mineral'],
        ['stop'],
    ]
    data = [{}, {}]
    for model_id, model in enumerate([model_best, model_bc]):
        executor = model.base

        val_acc, results = compute_acc(executor, loader, keywords, device)

        # print(sum(val_acc) / len(val_acc))
        # for j in range(len(results)):
        #     if len(results[j]) == 0:
        #         continue
        #     print(keywords[j][0])
        #     print(sum(results[j]) / len(results[j]))
        data[model_id]['average'] = sum(val_acc) / len(val_acc)
        for j in range(len(results)):
            tmp = results[j]
            if len(tmp) == 0:
                continue
            data[model_id][' '.join(keywords[j])] = sum(tmp) / len(tmp)
        print(data[model_id])
    with open(data_path, 'wb') as f:
        pickle.dump(data, f)


def plot_figure():
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    color = sns.hls_palette(10, l=.4)[1:]
    plt.figure(figsize=(4, 4))
    cnt = 0
    for key in data[0].keys():
        if key != 'average':
            plt.bar(
                key.split()[0], (data[0][key] - data[1][key]) * 100, color=color[cnt]
            )
            cnt += 1
    plt.axhline(
        (data[0]['average'] - data[1]['average']) * 100, linestyle='--', label='average'
    )
    plt.ylabel('$\Delta$ Acc (%)')
    plt.ylim(top=0.5)
    plt.gca().invert_yaxis()
    plt.legend()
    plt.savefig('categorized_acc.png', bbox_inches='tight', pad_inches=0)


if __name__ == '__main__':
    args = get_args()
    if args.compute:
        compute()
    if args.plot:
        plot_figure()
