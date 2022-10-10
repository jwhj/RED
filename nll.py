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

data_path = 'categorized_losses.pt'


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


def compute_losses():
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
    nlls = [{}, {}]
    for model_id, model in enumerate([model_best, model_bc]):
        executor = model.base

        results = [[] for _ in range(len(keywords))]
        test_nll = []
        pbar = tqdm(loader)
        for batch in pbar:
            batch = to_device(batch, device)
            cmd_type = batch['target_cmds']['cmd_type']
            bsz, nu = cmd_type.shape[:2]
            unit_mask = torch.tensor(range(nu), device=device).unsqueeze(0) < batch[
                'my_units'
            ]['num_units'].unsqueeze(1)
            non_cont_mask = cmd_type != 6
            mask = unit_mask * non_cont_mask
            tmp = executor.compute_loss(batch, mean=False)
            insts = executor.inst_dict.deparse(batch['inst'])
            for i in range(bsz):
                inst = insts[i]
                for j, keyword_list in enumerate(keywords):
                    for keyword in keyword_list:
                        if keyword in inst:
                            results[j].append(tmp[0][i].item())
                            break
                test_nll.extend(tmp[0].tolist())
            # tmp = executor.compute_prob_old(batch)
            # t1 = torch.masked_fill(
            #     tmp['cmd_type_prob'].argmax(dim=2) == cmd_type, ~mask, 0
            # ).sum(dim=1) / mask.sum(dim=1).clamp(1)
            # results.extend(
            #     filter(lambda x: x >= 0, t1.masked_fill(mask.sum(dim=1) == 0, -1).tolist())
            # )
        pbar.close()
        nlls[model_id]['average'] = sum(test_nll) / len(test_nll)
        for j in range(len(results)):
            tmp = results[j]
            nlls[model_id][' '.join(keywords[j])] = sum(tmp) / len(tmp)
        print(nlls[model_id])
    with open(data_path, 'wb') as f:
        pickle.dump(nlls, f)


def plot_figure():
    with open(data_path, 'rb') as f:
        nlls = pickle.load(f)
    color = sns.hls_palette(10, l=.4)[1:]
    plt.figure(figsize=(4, 4))
    cnt = 0
    for key in nlls[0].keys():
        if key != 'average':
            plt.bar(key.split()[0], nlls[0][key] - nlls[1][key], color=color[cnt])
            cnt += 1
    plt.axhline(
        nlls[0]['average'] - nlls[1]['average'], linestyle='--', label='average'
    )
    plt.ylabel('$\Delta$ NLL')
    plt.legend()
    plt.savefig('categorized_losses.png', bbox_inches='tight', pad_inches=0)


if __name__ == '__main__':
    args = get_args()
    if args.compute:
        compute_losses()
    if args.plot:
        plot_figure()
