from typing import Dict
import pickle
import torch
from torch import nn
import numpy as np
from utils import convert_to_raw_instruction
import common_utils.global_consts as gc


class RuleCoach(nn.Module):
    def __init__(self, inst_dict_path, num_games, kind_enemy=5, *unused, **kwargs) -> None:
        super().__init__()
        print('loading cmd dict from: ', inst_dict_path)
        if inst_dict_path is None or inst_dict_path == '':
            return None

        self.kind_enemy = kind_enemy
        inst_dict = pickle.load(open(inst_dict_path, 'rb'))
        inst_dict.set_max_sentence_length(15)
        self.inst_dict = inst_dict
        self.max_raw_chars = 200
        self.placeholder = nn.Linear(1, 1)
        self.rule_type = 'r1'
        self.hist_enemy = [None for _ in range(num_games)]

    def format_coach_input(self, batch):
        return batch

    def sample(self, batch: Dict[str, torch.Tensor], *args, infos=None, word_based=True, **kwargs):
        if self.rule_type == 'r2':
            return self.sample2(batch, infos=infos, word_based=word_based, **kwargs)
        elif self.rule_type == 'r3':
            return self.sample3(batch, infos=infos, word_based=word_based, **kwargs)
        elif self.rule_type == 'adv':
            return self.sample_adv(batch, infos=infos, word_based=word_based, **kwargs)

        device = batch['map'].device
        bsz = batch['map'].shape[0]
        cont = torch.zeros((bsz, 1), dtype=torch.long, device=device)
        cont_pi = torch.zeros((bsz, 2), device=device)
        cont_pi[:, 0] = 1
        inst = torch.empty((bsz, 1), dtype=torch.long, device=device).fill_(63)
        inst_pi = torch.zeros((bsz, 500), device=device)
        inst_pi[:, 63] = 1
        reply = {'cont': cont, 'cont_pi': cont_pi, 'inst': inst, 'inst_pi': inst_pi}

        samples = []
        lengths = []
        raws = []
        for i in range(bsz):
            idx = -1
            if infos is None or 'bad_transition' in infos[i] or infos[i]['frame_idx'] <= 100:
                # mine
                # idx = -1
                idx = 361
            elif batch['num_army'][i] < 7 and (batch['army_type'][i] == gc.UnitTypes.PEASANT.value).sum() < 6: # and infos[i]['frame_idx'] <= 50 * 10:
                # build peasants
                idx = 63
            # elif batch['num_enemy'][i] == 0:
            #     # scout
            #     idx = 386


            elif (batch['enemy_type'][i] == gc.UnitTypes.SPEARMAN.value).any() or self.hist_enemy[i] == gc.UnitTypes.SPEARMAN.value:
                self.hist_enemy[i] = gc.UnitTypes.SPEARMAN.value
                if not (batch['army_type'][i] == gc.UnitTypes.BLACKSMITH.value).any():
                    idx = 26
                elif (batch['army_type'][i] == gc.UnitTypes.SWORDMAN.value).sum() < 5:
                    idx = 337
            elif (batch['enemy_type'][i] == gc.UnitTypes.SWORDMAN.value).any() or self.hist_enemy[i] == gc.UnitTypes.SWORDMAN.value:
                self.hist_enemy[i] = gc.UnitTypes.SWORDMAN.value
                if not (batch['army_type'][i] == gc.UnitTypes.STABLE.value).any():
                    idx = 5
                elif (batch['army_type'][i] == gc.UnitTypes.CAVALRY.value).sum() < 5:
                    idx = 81
            elif (batch['enemy_type'][i] == gc.UnitTypes.CAVALRY.value).any() or self.hist_enemy[i] == gc.UnitTypes.CAVALRY.value:
                self.hist_enemy[i] = gc.UnitTypes.CAVALRY.value
                if not (batch['army_type'][i] == gc.UnitTypes.BARRACK.value).any():
                    idx = 15
                elif (batch['army_type'][i] == gc.UnitTypes.SPEARMAN.value).sum() < 5:
                    idx = 195
            elif (batch['enemy_type'][i] == gc.UnitTypes.DRAGON.value).any() or self.hist_enemy[i] == gc.UnitTypes.DRAGON.value:
                self.hist_enemy[i] = gc.UnitTypes.DRAGON.value
                if not (batch['army_type'][i] == gc.UnitTypes.WORKSHOP.value).any():
                    idx = 2
                elif (batch['army_type'][i] == gc.UnitTypes.ARCHER.value).sum() < 5:
                    idx = 55
            # elif (batch['current_cmd_type'][i][batch['army_type'][i] == gc.UnitTypes.PEASANT.value] == gc.CmdTypes.IDLE.value).sum() > 1:
            #     idx = 361
            # elif ((batch['army_type'][0] > 1) & (batch['army_type'][0] < 10)).sum() >= 5:
            #     idx = 0
            inst[i] = idx
            inst_ = self.inst_dict.get_inst(idx)
            tokens, length = self.inst_dict.parse(inst_, True)
            samples.append(tokens)
            if idx >= 0:
                lengths.append(length)
                raws.append(convert_to_raw_instruction(inst_, self.max_raw_chars))
            else:
                lengths.append(0)
                raws.append([-1] * self.max_raw_chars)

        if word_based:
            # #TODO
            # reply['inst'][reply['inst'] == -1] = 0
            inst = torch.LongTensor(samples).to(device)
        else:
            inst = reply['inst']
        inst_len = torch.LongTensor(lengths).to(device)
        reply['raw_inst'] = torch.LongTensor(raws).to(device)
        return inst, inst_len, reply['cont'], reply


    def sample2(self, batch: Dict[str, torch.Tensor], *args, infos=None, word_based=True, **kwargs):
        device = batch['map'].device
        bsz = batch['map'].shape[0]
        cont = torch.zeros((bsz, 1), dtype=torch.long, device=device)
        cont_pi = torch.zeros((bsz, 2), device=device)
        cont_pi[:, 0] = 1
        inst = torch.empty((bsz, 1), dtype=torch.long, device=device).fill_(63)
        inst_pi = torch.zeros((bsz, 500), device=device)
        inst_pi[:, 63] = 1
        reply = {'cont': cont, 'cont_pi': cont_pi, 'inst': inst, 'inst_pi': inst_pi}

        samples = []
        lengths = []
        raws = []
        for i in range(bsz):
            idx = -1
            if infos is None or 'bad_transition' in infos[i] or infos[i]['frame_idx'] <= 100:
                # mine
                # idx = -1
                idx = 361
            elif batch['num_army'][i] < 7 and (batch['army_type'][i] == gc.UnitTypes.PEASANT.value).sum() < 6: # and infos[i]['frame_idx'] <= 50 * 10:
                # build peasants
                idx = 63
            # elif batch['num_enemy'][i] == 0:
            #     # scout
            #     idx = 386

            elif not (batch['army_type'][i] == gc.UnitTypes.WORKSHOP.value).any():
                idx = 10   # build workshop
            elif (batch['enemy_type'][i] == gc.UnitTypes.DRAGON.value).any():
                if (batch['army_type'][i] == gc.UnitTypes.ARCHER.value).sum() < 5:
                    idx = 88 # make archer
            elif (batch['enemy_type'][i] == gc.UnitTypes.ARCHER.value).any():
                if (batch['army_type'][i] == gc.UnitTypes.STABLE.value).any() and (batch['army_type'][i] == gc.UnitTypes.CAVALRY.value).sum() < 5:
                    idx = 452 # make cavalry
                elif  (batch['army_type'][i] == gc.UnitTypes.BLACKSMITH.value).any() and (batch['army_type'][i] == gc.UnitTypes.SWORDMAN.value).sum() < 5:
                    idx = 337 # make swordman
                elif  (batch['army_type'][i] == gc.UnitTypes.BARRACK.value).any() and (batch['army_type'][i] == gc.UnitTypes.SPEARMAN.value).sum() < 5:
                    idx = 195
                elif (batch['army_type'][i] == gc.UnitTypes.ARCHER.value).sum() < 10:
                    idx = 88 # make more archer
            elif ((batch['enemy_type'][i] > 1) & (batch['enemy_type'][i] < 10)).any():

                # if (batch['enemy_type'][i] == gc.UnitTypes.SPEARMAN.value).any():
                #     if not (batch['army_type'][i] == gc.UnitTypes.BLACKSMITH.value).any():
                #         idx = 26
                #     elif (batch['army_type'][i] == gc.UnitTypes.SWORDMAN.value).sum() < 5:
                #         idx = 337
                # elif (batch['enemy_type'][i] == gc.UnitTypes.SWORDMAN.value).any():
                #     if not (batch['army_type'][i] == gc.UnitTypes.STABLE.value).any():
                #         idx = 5
                #     elif (batch['army_type'][i] == gc.UnitTypes.CAVALRY.value).sum() < 5:
                #         idx = 452
                # elif (batch['enemy_type'][i] == gc.UnitTypes.CAVALRY.value).any():
                #     if not (batch['army_type'][i] == gc.UnitTypes.BARRACK.value).any():
                #         idx = 53
                #     elif (batch['army_type'][i] == gc.UnitTypes.SPEARMAN.value).sum() < 5:
                #         idx = 195
                if (batch['army_type'][i] == gc.UnitTypes.DRAGON.value).sum() < 5:
                    idx = 20 # make dragon
            # elif (batch['army_type'][i] == gc.UnitTypes.GUARD_TOWER.value).sum() < 2:
            #     idx = 120

            inst[i] = idx
            inst_ = self.inst_dict.get_inst(idx)
            tokens, length = self.inst_dict.parse(inst_, True)
            samples.append(tokens)
            if idx >= 0:
                lengths.append(length)
                raws.append(convert_to_raw_instruction(inst_, self.max_raw_chars))
            else:
                lengths.append(0)
                raws.append([-1] * self.max_raw_chars)

        # # convert format needed by executor
        # samples = []
        # lengths = []
        # raws = []
        # for idx in reply['inst']:
        #     inst = self.inst_dict.get_inst(int(idx.item()))
        #     tokens, length = self.inst_dict.parse(inst, True)
        #     samples.append(tokens)
        #     lengths.append(length)
        #     raw = convert_to_raw_instruction(inst, self.max_raw_chars)
        #     raws.append(convert_to_raw_instruction(inst, self.max_raw_chars))

        if word_based:
            # #TODO
            # reply['inst'][reply['inst'] == -1] = 0
            inst = torch.LongTensor(samples).to(device)
        else:
            inst = reply['inst']
        inst_len = torch.LongTensor(lengths).to(device)
        reply['raw_inst'] = torch.LongTensor(raws).to(device)
        return inst, inst_len, reply['cont'], reply

        # inst_len = torch.zeros((bsz,), dtype=torch.long, device=device)
        # inst = torch.zeros((bsz, 15), dtype=torch.long, device=device)
        # reply['raw_inst'] = torch.empty((bsz, 200), dtype=torch.long, device=device).fill_(-1)
        # return inst, inst_len, reply['cont'], reply


    def sample3(self, batch: Dict[str, torch.Tensor], *args, infos=None, word_based=True, **kwargs):
        device = batch['map'].device
        bsz = batch['map'].shape[0]
        cont = torch.zeros((bsz, 1), dtype=torch.long, device=device)
        cont_pi = torch.zeros((bsz, 2), device=device)
        cont_pi[:, 0] = 1
        inst = torch.empty((bsz, 1), dtype=torch.long, device=device).fill_(63)
        inst_pi = torch.zeros((bsz, 500), device=device)
        inst_pi[:, 63] = 1
        reply = {'cont': cont, 'cont_pi': cont_pi, 'inst': inst, 'inst_pi': inst_pi}

        samples = []
        lengths = []
        raws = []
        for i in range(bsz):
            idx = -1
            if infos is None or 'bad_transition' in infos[i] or infos[i]['frame_idx'] <= 100:
                # mine
                # idx = -1
                idx = 361
            elif (self.kind_enemy != 6 or i % self.kind_enemy != 5) and (batch['num_army'][i] < 7 and (batch['army_type'][i] == gc.UnitTypes.PEASANT.value).sum() < 6):
                # if batch['num_army'][i] < 7 and (batch['army_type'][i] == gc.UnitTypes.PEASANT.value).sum() < 6: # and infos[i]['frame_idx'] <= 50 * 10:
                # build peasants
                idx = 63

            elif i % self.kind_enemy == 3: # enemy dragon
                if not (batch['army_type'][i] == gc.UnitTypes.WORKSHOP.value).any(): 
                    idx = 10   # build workshop
                elif (batch['army_type'][i] == gc.UnitTypes.ARCHER.value).sum() < 5:
                    idx = 88 # make archer
            elif i % self.kind_enemy == 0:   # enemy SPEARMAN
                if not (batch['army_type'][i] == gc.UnitTypes.BLACKSMITH.value).any():
                    idx = 26 # bulid blacksmith
                elif (batch['army_type'][i] == gc.UnitTypes.SWORDMAN.value).sum() < 5:
                    idx = 337
            elif i % self.kind_enemy == 1: # enemy SWORDMAN
                if not (batch['army_type'][i] == gc.UnitTypes.STABLE.value).any(): 
                    idx = 5 # build stable
                elif (batch['army_type'][i] == gc.UnitTypes.CAVALRY.value).sum() < 5:
                    idx = 452
            elif i % self.kind_enemy == 2: # enemy CAVALRY
                if not (batch['army_type'][i] == gc.UnitTypes.BARRACK.value).any(): 
                    idx = 53
                elif (batch['army_type'][i] == gc.UnitTypes.SPEARMAN.value).sum() < 5:
                    idx = 195

            elif i % self.kind_enemy == 4: # enemy ARCHER
                idx = -1
                if (i // self.kind_enemy) % 3 == 0:
                    if not (batch['army_type'][i] == gc.UnitTypes.BLACKSMITH.value).any():
                        idx = 26 # bulid blacksmith
                    elif (batch['army_type'][i] == gc.UnitTypes.SWORDMAN.value).sum() < 5:
                        idx = 337
                elif (i // self.kind_enemy) % 3 == 1:
                    if not (batch['army_type'][i] == gc.UnitTypes.STABLE.value).any():
                        idx = 5 # build stable
                    elif (batch['army_type'][i] == gc.UnitTypes.CAVALRY.value).sum() < 5:
                        idx = 452
                elif (i // self.kind_enemy) % 3 == 2:
                    if not (batch['army_type'][i] == gc.UnitTypes.BARRACK.value).any():
                        idx = 53
                    elif (batch['army_type'][i] == gc.UnitTypes.SPEARMAN.value).sum() < 5:
                        idx = 195

                if (batch['army_type'][i] == gc.UnitTypes.STABLE.value).any() and (batch['army_type'][i] == gc.UnitTypes.CAVALRY.value).sum() < 5:
                    idx = 452 # make cavalry
                elif  (batch['army_type'][i] == gc.UnitTypes.BLACKSMITH.value).any() and (batch['army_type'][i] == gc.UnitTypes.SWORDMAN.value).sum() < 5:
                    idx = 337 # make swordman
                elif  (batch['army_type'][i] == gc.UnitTypes.BARRACK.value).any() and (batch['army_type'][i] == gc.UnitTypes.SPEARMAN.value).sum() < 5:
                    idx = 195
            
            elif self.kind_enemy == 6 and i % self.kind_enemy == 5:
                idx = -1
                # if (i // self.kind_enemy) % 3 == 0:
                #     if not (batch['army_type'][i] == gc.UnitTypes.BLACKSMITH.value).any():
                #         idx = 26 # bulid blacksmith
                #     elif (batch['army_type'][i] == gc.UnitTypes.SWORDMAN.value).sum() < 5:
                #         idx = 337
                # elif (i // self.kind_enemy) % 3 == 1:
                #     if not (batch['army_type'][i] == gc.UnitTypes.STABLE.value).any():
                #         idx = 5 # build stable
                #     elif (batch['army_type'][i] == gc.UnitTypes.CAVALRY.value).sum() < 5:
                #         idx = 452
                # elif (i // self.kind_enemy) % 3 == 2:
                #     if not (batch['army_type'][i] == gc.UnitTypes.BARRACK.value).any():
                #         idx = 53
                #     elif (batch['army_type'][i] == gc.UnitTypes.SPEARMAN.value).sum() < 5:
                #         idx = 195

                # if ((batch['army_type'][i] > 1) & (batch['army_type'][i] < 8)).sum() >= 3:
                #     idx = 0


            # elif (batch['enemy_type'][i] == gc.UnitTypes.DRAGON.value).any():
            #     if (batch['army_type'][i] == gc.UnitTypes.ARCHER.value).sum() < 5:
            #         idx = 88 # make archer
            # elif (batch['enemy_type'][i] == gc.UnitTypes.ARCHER.value).any():
            #     # idx = -1
            #     if (batch['army_type'][i] == gc.UnitTypes.STABLE.value).any() and (batch['army_type'][i] == gc.UnitTypes.CAVALRY.value).sum() < 5:
            #         idx = 452 # make cavalry
            #     elif  (batch['army_type'][i] == gc.UnitTypes.BLACKSMITH.value).any() and (batch['army_type'][i] == gc.UnitTypes.SWORDMAN.value).sum() < 5:
            #         idx = 373 # make swordman
            #     elif  (batch['army_type'][i] == gc.UnitTypes.BARRACK.value).any() and (batch['army_type'][i] == gc.UnitTypes.SPEARMAN.value).sum() < 5:
            #         idx = 195
            #     elif (batch['army_type'][i] == gc.UnitTypes.ARCHER.value).sum() < 10:
            #         idx = 88 # make more archer
            # elif ((batch['enemy_type'][i] > 1) & (batch['enemy_type'][i] < 8)).any():

            #     if (batch['enemy_type'][i] == gc.UnitTypes.SPEARMAN.value).any():
            #         if not (batch['army_type'][i] == gc.UnitTypes.BLACKSMITH.value).any():
            #             idx = 26
            #         elif (batch['army_type'][i] == gc.UnitTypes.SWORDMAN.value).sum() < 5:
            #             idx = 337
            #     elif (batch['enemy_type'][i] == gc.UnitTypes.SWORDMAN.value).any():
            #         if not (batch['army_type'][i] == gc.UnitTypes.STABLE.value).any():
            #             idx = 5
            #         elif (batch['army_type'][i] == gc.UnitTypes.CAVALRY.value).sum() < 5:
            #             idx = 452
            #     elif (batch['enemy_type'][i] == gc.UnitTypes.CAVALRY.value).any():
            #         if not (batch['army_type'][i] == gc.UnitTypes.BARRACK.value).any():
            #             idx = 53
            #         elif (batch['army_type'][i] == gc.UnitTypes.SPEARMAN.value).sum() < 5:
            #             idx = 195

            inst[i] = idx
            inst_ = self.inst_dict.get_inst(idx)
            tokens, length = self.inst_dict.parse(inst_, True)
            samples.append(tokens)
            if idx >= 0:
                lengths.append(length)
                raws.append(convert_to_raw_instruction(inst_, self.max_raw_chars))
            else:
                lengths.append(0)
                raws.append([-1] * self.max_raw_chars)

        if word_based:
            # #TODO
            # reply['inst'][reply['inst'] == -1] = 0
            inst = torch.LongTensor(samples).to(device)
        else:
            inst = reply['inst']
        inst_len = torch.LongTensor(lengths).to(device)
        reply['raw_inst'] = torch.LongTensor(raws).to(device)
        return inst, inst_len, reply['cont'], reply

    def sample_adv(self, batch: Dict[str, torch.Tensor], *args, infos=None, word_based=True, **kwargs):
        device = batch['map'].device
        bsz = batch['map'].shape[0]
        cont = torch.zeros((bsz, 1), dtype=torch.long, device=device)
        cont_pi = torch.zeros((bsz, 2), device=device)
        cont_pi[:, 0] = 1
        inst = torch.empty((bsz, 1), dtype=torch.long, device=device).fill_(63)
        inst_pi = torch.zeros((bsz, 500), device=device)
        inst_pi[:, 63] = 1
        reply = {'cont': cont, 'cont_pi': cont_pi, 'inst': inst, 'inst_pi': inst_pi}

        samples = []
        lengths = []
        raws = []
        for i in range(bsz):
            idx = -1
            if infos is None or 'bad_transition' in infos[i] or infos[i]['frame_idx'] <= 100:
                # mine
                # idx = -1
                idx = 361
            elif (self.kind_enemy != 6 or i % self.kind_enemy != 5) and (batch['num_army'][i] < 7 and (batch['army_type'][i] == gc.UnitTypes.PEASANT.value).sum() < 6):
                # if batch['num_army'][i] < 7 and (batch['army_type'][i] == gc.UnitTypes.PEASANT.value).sum() < 6: # and infos[i]['frame_idx'] <= 50 * 10:
                # build peasants
                idx = 63

            elif i % self.kind_enemy == 3: # enemy dragon
                if not (batch['army_type'][i] == gc.UnitTypes.STABLE.value).any(): 
                    idx = 5   # build stable
                elif (batch['army_type'][i] == gc.UnitTypes.CAVALRY.value).sum() < 5:
                    idx = 452 # build another cavalry
            elif i % self.kind_enemy == 0:   # enemy SPEARMAN
                if not (batch['army_type'][i] == gc.UnitTypes.STABLE.value).any():
                    idx = 5 # bulid stable
                elif (batch['army_type'][i] == gc.UnitTypes.CAVALRY.value).sum() < 5:
                    idx = 452
            elif i % self.kind_enemy == 1: # enemy SWORDMAN
                if not (batch['army_type'][i] == gc.UnitTypes.BARRACK.value).any(): 
                    idx = 53 # build barrack
                elif (batch['army_type'][i] == gc.UnitTypes.SPEARMAN.value).sum() < 5:
                    idx = 195
            elif i % self.kind_enemy == 2: # enemy CAVALRY
                if not (batch['army_type'][i] == gc.UnitTypes.BLACKSMITH.value).any(): 
                    idx = 26
                elif (batch['army_type'][i] == gc.UnitTypes.SWORDMAN.value).sum() < 5:
                    idx = 337

            elif i % self.kind_enemy == 4: # enemy ARCHER
                if not (batch['army_type'][i] == gc.UnitTypes.WORKSHOP.value).any(): 
                    idx = 10   # build workshop
                elif (batch['army_type'][i] == gc.UnitTypes.ARCHER.value).sum() < 5:
                    idx = 12 # build dragon
            
            elif self.kind_enemy == 6 and i % self.kind_enemy == 5:
                idx = 75
                # if (i // self.kind_enemy) % 3 == 0:
                #     if not (batch['army_type'][i] == gc.UnitTypes.BLACKSMITH.value).any():
                #         idx = 26 # bulid blacksmith
                #     elif (batch['army_type'][i] == gc.UnitTypes.SWORDMAN.value).sum() < 5:
                #         idx = 337
                # elif (i // self.kind_enemy) % 3 == 1:
                #     if not (batch['army_type'][i] == gc.UnitTypes.STABLE.value).any():
                #         idx = 5 # build stable
                #     elif (batch['army_type'][i] == gc.UnitTypes.CAVALRY.value).sum() < 5:
                #         idx = 452
                # elif (i // self.kind_enemy) % 3 == 2:
                #     if not (batch['army_type'][i] == gc.UnitTypes.BARRACK.value).any():
                #         idx = 53
                #     elif (batch['army_type'][i] == gc.UnitTypes.SPEARMAN.value).sum() < 5:
                #         idx = 195

                # if ((batch['army_type'][i] > 1) & (batch['army_type'][i] < 8)).sum() >= 3:
                #     idx = 0


            # elif (batch['enemy_type'][i] == gc.UnitTypes.DRAGON.value).any():
            #     if (batch['army_type'][i] == gc.UnitTypes.ARCHER.value).sum() < 5:
            #         idx = 88 # make archer
            # elif (batch['enemy_type'][i] == gc.UnitTypes.ARCHER.value).any():
            #     # idx = -1
            #     if (batch['army_type'][i] == gc.UnitTypes.STABLE.value).any() and (batch['army_type'][i] == gc.UnitTypes.CAVALRY.value).sum() < 5:
            #         idx = 452 # make cavalry
            #     elif  (batch['army_type'][i] == gc.UnitTypes.BLACKSMITH.value).any() and (batch['army_type'][i] == gc.UnitTypes.SWORDMAN.value).sum() < 5:
            #         idx = 373 # make swordman
            #     elif  (batch['army_type'][i] == gc.UnitTypes.BARRACK.value).any() and (batch['army_type'][i] == gc.UnitTypes.SPEARMAN.value).sum() < 5:
            #         idx = 195
            #     elif (batch['army_type'][i] == gc.UnitTypes.ARCHER.value).sum() < 10:
            #         idx = 88 # make more archer
            # elif ((batch['enemy_type'][i] > 1) & (batch['enemy_type'][i] < 8)).any():

            #     if (batch['enemy_type'][i] == gc.UnitTypes.SPEARMAN.value).any():
            #         if not (batch['army_type'][i] == gc.UnitTypes.BLACKSMITH.value).any():
            #             idx = 26
            #         elif (batch['army_type'][i] == gc.UnitTypes.SWORDMAN.value).sum() < 5:
            #             idx = 337
            #     elif (batch['enemy_type'][i] == gc.UnitTypes.SWORDMAN.value).any():
            #         if not (batch['army_type'][i] == gc.UnitTypes.STABLE.value).any():
            #             idx = 5
            #         elif (batch['army_type'][i] == gc.UnitTypes.CAVALRY.value).sum() < 5:
            #             idx = 452
            #     elif (batch['enemy_type'][i] == gc.UnitTypes.CAVALRY.value).any():
            #         if not (batch['army_type'][i] == gc.UnitTypes.BARRACK.value).any():
            #             idx = 53
            #         elif (batch['army_type'][i] == gc.UnitTypes.SPEARMAN.value).sum() < 5:
            #             idx = 195

            inst[i] = idx
            inst_ = self.inst_dict.get_inst(idx)
            tokens, length = self.inst_dict.parse(inst_, True)
            samples.append(tokens)
            if idx >= 0:
                lengths.append(length)
                raws.append(convert_to_raw_instruction(inst_, self.max_raw_chars))
            else:
                lengths.append(0)
                raws.append([-1] * self.max_raw_chars)

        if word_based:
            # #TODO
            # reply['inst'][reply['inst'] == -1] = 0
            inst = torch.LongTensor(samples).to(device)
        else:
            inst = reply['inst']
        inst_len = torch.LongTensor(lengths).to(device)
        reply['raw_inst'] = torch.LongTensor(raws).to(device)
        return inst, inst_len, reply['cont'], reply