import torch
from torch import nn
import pickle
import numpy as np
from utils import convert_to_raw_instruction
import common_utils.global_consts as gc


class RandomCoach(nn.Module):

    good_insts = [
    'attack',
    'mine with all idle peasant',
    'send all peasant to mine',
    'build a workshop',
    'build a stable',
    'build a dragon',
    'build a guard tower',
    'build a catapult',
    'build a blacksmith',
    'build peasant',
    'build a barrack',
    'build archer',
    'build cavalry',
    'build a spearman',
    'build another swordman',
    'keep scouting',
    'send one peasant to scout',
    ]

    def __init__(self, inst_dict_path, p=1, small_action=False, **kwargs) -> None:
        super().__init__()
        inst_dict = pickle.load(open(inst_dict_path, 'rb'))
        inst_dict.set_max_sentence_length(15)
        self.inst_dict = inst_dict

        if small_action:
            self.actions = [-1] + [self.inst_dict.get_inst_idx(x) for x in self.good_insts]
        else:
            self.actions = list(np.arange(500))

        self.max_raw_chars = kwargs.get('max_raw_chars', 200)
        self.p = p

    def format_coach_input(self, batch):
        return batch
    
    def sample(self, coach_input, word_based=True, **kwargs):

        bsz = coach_input['map'].shape[0]
        device = coach_input['map'].device
        coach_reply = {}
        coach_reply['cont'] = torch.zeros((bsz, 1), dtype=torch.long)
        coach_reply['cont_pi'] = torch.zeros((bsz, 2))
        coach_reply['inst'] = torch.zeros((bsz, 1), dtype=torch.long).fill_(-1)
        coach_reply['inst_pi'] = torch.zeros((bsz, 500))
        coach_reply['raw_inst'] = torch.zeros((bsz, self.max_raw_chars), dtype=torch.long)

        samples = []
        lengths = []
        raws = []
        sampled_inst = np.random.choice(self.actions, bsz)
        for i, inst_idx in enumerate(sampled_inst):
            if np.random.rand() >= self.p:
                inst_idx = -1
            coach_reply['inst'][i] = inst_idx

            inst_ = self.inst_dict.get_inst(inst_idx)
            tokens, length = self.inst_dict.parse(inst_, True)
            samples.append(tokens)
            if inst_idx >= 0:
                lengths.append(length)
                raws.append(convert_to_raw_instruction(inst_, self.max_raw_chars))
            else:
                lengths.append(0)
                raws.append([-1] * self.max_raw_chars)

        if word_based:
            inst = torch.LongTensor(samples).to(device)
        else:
            inst = coach_reply['inst']
        inst_len = torch.LongTensor(lengths).to(device)
        coach_reply['raw_inst'] = torch.LongTensor(raws).to(device)       
        coach_reply = {k : v.to(device) for k, v in coach_reply.items()}    

        return inst, inst_len, coach_reply['cont'], coach_reply
