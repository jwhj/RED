from email import policy
import os
import sys
import pickle
from numpy import isin
from pyparsing import Word
import torch
from torch import nn, distributions

sys.path.append(os.path.join(os.environ['MINIRTS_ROOT'], 'scripts/behavior_clone'))
from executor import (
    Executor,
    create_loss_masks,
    gc,
    weight_norm,
    log_sum_exp,
    create_real_unit_mask,
    is_word_based,
)
from cmd_heads import assert_eq
from rnn_coach import ConvRnnCoach as ConvRnnCoach_, convert_to_raw_instruction
from models.rule import RuleCoach
from models.single_action_inherit import ExecutorSingle
from models.single_action_inherit import Policy as Policy_


class ConvRnnCoach(ConvRnnCoach_):
    def __init__(self, *args, **kwargs):
        if 'args' in kwargs:
            kwargs['args'].glob_dropout = 0
        super().__init__(*args, **kwargs)

    def mode(self, cont_probs, probs, prev_samples):

        assert_eq(cont_probs.size(1), 2)
        cont_samples = cont_probs.argmax(1)
        new_samples = probs.argmax(1)

        assert_eq(prev_samples.size(), new_samples.size())
        samples = cont_samples * prev_samples + (1 - cont_samples) * new_samples
        return {
            'cont': cont_samples,
            'inst': samples,
        }

    def sample(self, batch, word_based=True, deter=False):
        """used for actor in ELF and visually evaulating model

        return
            inst: [batch, max_sentence_len], even inst is one-hot
            inst_len: [batch]
        """
        output = self.rl_forward(batch)
        if not deter:
            samples = self.sampler.sample(
                output['cont_pi'], output['inst_pi'], batch['prev_inst_idx'])
        else: 
            samples = self.mode(
                output['cont_pi'], output['inst_pi'], batch['prev_inst_idx'])

        reply = {
            'cont': samples['cont'].unsqueeze(1),
            'cont_pi': output['cont_pi'],
            'inst': samples['inst'].unsqueeze(1),
            'inst_pi': output['inst_pi'],
        }

        # convert format needed by executor
        samples = []
        lengths = []
        raws = []
        for idx in reply['inst']:
            inst = self.inst_dict.get_inst(int(idx.item()))
            tokens, length = self.inst_dict.parse(inst, True)
            samples.append(tokens)
            lengths.append(length)
            raw = convert_to_raw_instruction(inst, self.max_raw_chars)
            raws.append(convert_to_raw_instruction(inst, self.max_raw_chars))

        device = reply['cont'].device
        if word_based:
            # for word based
            inst = torch.LongTensor(samples).to(device)
        else:
            inst = reply['inst']

        inst_len = torch.LongTensor(lengths).to(device)
        reply['raw_inst'] = torch.LongTensor(raws).to(device)
        return inst, inst_len, reply['cont'], reply

class AutoResetRNN(nn.Module):

    def __init__(self, input_dim, output_dim, num_layers=1, batch_first=False, rnn_type='gru'):
        super().__init__()
        self.__type = rnn_type
        if self.__type == 'gru':
            self.__net = nn.GRU(input_dim, output_dim, num_layers=num_layers, batch_first=batch_first)
        elif self.__type == 'lstm':
            self.__net = nn.LSTM(input_dim, output_dim, num_layers=num_layers, batch_first=batch_first)
        else:
            raise NotImplementedError(f'RNN type {self.__type} has not been implemented.')

    def __forward(self, x, h):
        if self.__type == 'lstm':
            h = torch.split(h, h.shape[-1] // 2, dim=-1)
            h = (h[0].contiguous(), h[1].contiguous())
        x_, h_ = self.__net(x, h)
        if self.__type == 'lstm':
            h_ = torch.cat(h_, -1)
        return x_, h_

    def forward(self, x, h, on_reset=None):
        if on_reset is None:
            x_, h_ = self.__forward(x, h)
        else:
            outputs = []
            for t in range(on_reset.shape[0]):
                x_, h = self.__forward(x[t:t + 1], (h * (1 - on_reset[t:t + 1])).contiguous())
                outputs.append(x_)
            x_ = torch.cat(outputs, 0)
            h_ = h
        return x_, h_


class ConvRnnCoachRL(ConvRnnCoach):

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

    def __init__(self, is_rnn=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.actions = [-1] + [self.inst_dict.get_inst_idx(x) for x in self.good_insts]
        self.action_idxs = {a: idx for idx, a in enumerate(self.actions)}
        self.n_actions = len(self.actions)

        self.action_cls = nn.Sequential(
            weight_norm(nn.Linear(self.glob_feat_dim, 512), dim=None),
            nn.ReLU(),
            weight_norm(nn.Linear(512, self.n_actions), dim=None),
            nn.Softmax(dim=1)
        )
        self.is_rnn = is_rnn
        if self.is_rnn:
            self.rnn = AutoResetRNN(self.glob_feat_dim, self.glob_feat_dim)
            self.register_buffer('_default_rnn_hidden_coach', torch.zeros([1, self.glob_feat_dim]))

    @classmethod
    def load(cls, model_file=None, is_rnn=False):
        if model_file is None:
            model_file = os.path.join(
                os.environ['MINIRTS_ROOT'], 'pretrained_models/coach_rnn500.pt'
            )
        with open(model_file + '.params', 'rb') as f:
            params = pickle.load(f)
        print(params)
        model = cls(is_rnn=is_rnn, **params)
        return model

    def parse_instruction(self, coach_action, word_based=True):
        
        bsz = coach_action['cont'].shape[0]
        device = coach_action['cont'].device
        reply = {
            'cont': coach_action['cont'].unsqueeze(1),
            'cont_pi': coach_action['cont_pi'],
            'inst': [self.actions[i] for i in coach_action['inst']],
            'inst_pi': torch.zeros((bsz, 500), device=device),
        }

        # convert format needed by executor
        
        samples = []
        lengths = []
        raws = []
        
        for idx in reply['inst']:
            inst = self.inst_dict.get_inst(int(idx))
            tokens, length = self.inst_dict.parse(inst, True)
            samples.append(tokens)
            lengths.append(length)
            raws.append(convert_to_raw_instruction(inst, self.max_raw_chars))

        # device = reply['cont'].device
        if word_based:
            # for word based
            inst = torch.LongTensor(samples).to(device)
        else:
            inst = torch.LongTensor(reply['inst']).unsqueeze(1).to(device)

        inst_len = torch.LongTensor(lengths).to(device)
        reply['raw_inst'] = torch.LongTensor(raws).to(device)
        reply['inst'] = torch.LongTensor(reply['inst']).unsqueeze(1).to(device)

        coach_batch = {
            "inst": inst,
            "inst_len": inst_len,
            "inst_cont": reply['cont']
        }
        return reply, coach_batch
    
    def deparse(self, inst):
        actions = [self.action_idxs[i] for i in inst.view(-1).tolist()]
        return torch.LongTensor(actions).view(inst.shape).to(inst)    
    
    def forward(self, inputs, policy_state=None, **kwargs):
        coach_input = inputs
        batch = self._format_rl_language_input(coach_input)
        glob_feat = self._forward(batch)

        if self.is_rnn:
            on_reset = kwargs.get('on_reset')                                                     # bsz, 1
            bsz = kwargs.get('batch_size')

            if bsz != batch['map'].size(0):
                need_reshape = True
                size_0, size_1 = batch['map'].size(0) // bsz, bsz
            else:
                need_reshape = False

            if need_reshape: # For ananlyze
                x = glob_feat.view(size_0, size_1, -1)
                hid = policy_state.transpose(0, 1)                               # num_layers, bsz, dim
                cur_on_reset = on_reset.view(size_0, size_1, -1)                    # T, bsz, 1
            else:            # For rollout
                hid = getattr(self, '_default_rnn_hidden_coach').repeat(1, bsz, 1)    # num_layers, bsz, dim
                if policy_state is not None:
                    # hid = hid * on_reset + policy_state.transpose(0, 1) * (1 - on_reset)
                    hid = policy_state.transpose(0, 1)
                x = glob_feat.unsqueeze(0)
                cur_on_reset = None
            x_, out_h = self.rnn(x, hid, cur_on_reset)
            if need_reshape:
                glob_feat = x_.view(size_0 * size_1, -1)
                new_policy_state = None                         # useless for analyze
            else:
                glob_feat = x_.squeeze(0)
                new_policy_state = out_h.transpose(0, 1)        # bsz, num_layers, dim
        else:
            new_policy_state = None

        value = self.value(glob_feat).squeeze()
        cont_prob = self.cont_cls.compute_prob(glob_feat)
        inst_prob = self.action_cls(glob_feat)

        probs = {
            "cont_pi": cont_prob,
            "inst_pi": inst_prob,
        }

        return value, probs, coach_input, new_policy_state


    def sample(self, batch, word_based=True, deter=False, policy_state=None):
        """used for actor in ELF and visually evaulating model

        return
            inst: [batch, max_sentence_len], even inst is one-hot
            inst_len: [batch]
        """
        bsz = batch['frame_passed'].shape[0]
        value, probs, coach_input, policy_state = self.forward(batch,
                                                            policy_state=policy_state,
                                                            on_reset=torch.zeros_like(batch['frame_passed']),
                                                            batch_size=bsz)
        if deter:
            inst_action = probs['inst_pi'].argmax(1)
            cont_action = probs['cont_pi'].argmax(1)
        else: 
            inst_action = probs['inst_pi'].multinomial(1).squeeze(1)
            cont_action = probs['cont_pi'].multinomial(1).squeeze(1)

        inst_action = cont_action * self.deparse(coach_input['prev_inst_idx']) + (1 - cont_action) * inst_action

        instruction_actions = {
            "cont": cont_action,
            "cont_pi": probs['cont_pi'],
            "inst": inst_action,
            "inst_pi": probs['inst_pi']
        }

        coach_reply, coach_batch = self.parse_instruction(instruction_actions)
        coach_reply['policy_state'] = policy_state

        return coach_batch['inst'], coach_batch['inst_len'], coach_reply['cont'], coach_reply


class ExecutorRNNSingle(ExecutorSingle):
    def _preprocess(self, batch):
        inst = batch['inst']
        inst_len = batch['inst_len']
        inst_cont = batch['inst_cont']
        executor_input = self.format_executor_input(batch, inst, inst_len, inst_cont)
        return executor_input


class Policy(Policy_):
    ACTION_KEYS = [
        'glob_cont_prob',
        'cmd_type_prob',
        'gather_idx_prob',
        'attack_idx_prob',
        'unit_type_prob',
        'building_type_prob',
        'building_loc_prob',
        'move_loc_prob',
        'unit_cont_prob',
    ]

    def __init__(self, pretrained: bool = True, use_rule_based: bool = False):
        assert pretrained
        # super().__init__()
        nn.Module.__init__(self)
        model_path = os.path.join('ckpt', 'executor_rnn_with_cont_bsz128', 'best_checkpoint.pt')
        self.base = ExecutorRNNSingle.load(model_path).to('cuda')
        if use_rule_based:
            self.coach = RuleCoach(os.path.join(os.environ['MINIRTS_ROOT'], 'data/dataset/dict.pt'))
        else:
            coach_path = os.path.join(
                os.environ['MINIRTS_ROOT'], 'pretrained_models/coach_rnn500.pt'
            )
            self.coach = ConvRnnCoach.load(coach_path).to('cuda')
        for param in self.coach.parameters():
            param.requires_grad_(False)

    def generate_instructions(self, batch, envs, infos=None, dropout:float=None, mix_rule_coach:RuleCoach=None, policy_state=None):
        batch_ = {key: batch[key].to('cuda') for key in batch}
        coach_input = self.coach.format_coach_input(batch_)
        if isinstance(self.coach, RuleCoach):
            inst, inst_len, inst_cont, coach_reply = self.coach.sample(coach_input, True, infos=infos, word_based=is_word_based(self.base.args.inst_encoder_type))
            # batch['hist_inst'] = coach_reply['inst'].repeat([1, 5])
        elif isinstance(self.coach, ConvRnnCoachRL):
            inst, inst_len, inst_cont, coach_reply = self.coach.sample(coach_input, is_word_based(self.base.args.inst_encoder_type), deter=True, policy_state=policy_state)
        else:
            inst, inst_len, inst_cont, coach_reply = self.coach.sample(coach_input, is_word_based(self.base.args.inst_encoder_type), deter=False)
        if dropout is not None:
            device = inst.device
            mask = torch.rand(inst_len.shape, device=device) < dropout
            inst_len.masked_fill_(mask, 0)
            coach_reply['cont_pi'][:, 0].masked_fill_(mask, 0)
            coach_reply['cont_pi'][:, 1].masked_fill_(mask, 0)
            mask.unsqueeze_(1)
            if is_word_based(self.base.args.inst_encoder_type):
                inst.masked_fill_(mask, 927)
            else:
                inst.masked_fill_(mask, -1)
            inst_cont.masked_fill_(mask, 0)
            coach_reply['cont'].masked_fill_(mask, 0)
            coach_reply['inst'].masked_fill_(mask, -1)
            coach_reply['raw_inst'].masked_fill_(mask, -1)
            coach_reply['inst_pi'].masked_fill_(mask, 0)
        if mix_rule_coach is not None:
            inst_, inst_len_, inst_cont_, coach_reply_ = mix_rule_coach.sample(batch_, True, infos=infos)
            i = inst.shape[0] // 2
            inst = torch.cat([inst[:i], inst_[i:]], dim=0)
            inst_len = torch.cat([inst_len[:i], inst_len_[i:]], dim=0)
            inst_cont = torch.cat([inst_cont[:i], inst_cont_[i:]], dim=0)
            for key in coach_reply:
                coach_reply[key] = torch.cat([coach_reply[key][:i], coach_reply_[key][i:]], dim=0)

        if 'policy_state' in coach_reply:
            policy_state = coach_reply['policy_state']
            del coach_reply['policy_state']
        else:
            policy_state = None

        for i, env in enumerate(envs.envs):
            tmp = {key: coach_reply[key][i, None] for key in coach_reply}
            env.set_coach_reply(tmp)
        batch['inst'] = inst
        batch['inst_len'] = inst_len
        batch['inst_cont'] = inst_cont

        return policy_state, coach_reply['inst']