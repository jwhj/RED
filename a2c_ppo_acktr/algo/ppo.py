from typing import Dict, Optional, OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

def to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, dict):
        return { key: to_device(batch[key], device) for key in batch}
    elif isinstance(batch, tuple):
        return tuple(to_device(x, device) for x in batch)
    else:
        assert False, f'unsupported type {type(batch)}'

class PPO():
    def __init__(self,
                 actor_critic,
                 refenrence,
                 beta,
                 kl_coeff:Dict[str, float],
                 reference_ema:Optional[float],
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 mini_batch_size,
                 value_loss_coef,
                 entropy_coef,
                 device,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 use_clipped_value_loss=True,
                 pretrained=False):

        self.actor_critic = actor_critic
        self.reference = refenrence
        self.device=device

        self.clip_param = clip_param
        self.kl_coeff = kl_coeff
        self.reference_ema = reference_ema
        self.d_targ = 0.01
        self.beta = beta
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch
        self.mini_batch_size = mini_batch_size

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.pretrained = pretrained
        if pretrained:
            params=[
                {'params':[p for n,p in actor_critic.named_parameters() if 'value_proj' in n and 'coach' not in n],'lr':lr},
                {'params':[p for n,p in actor_critic.named_parameters() if 'value_proj' not in n and 'coach' not in n],'lr':lr}
            ]
        else:
            params=actor_critic.parameters()
        self.optimizer = optim.Adam(params, lr=lr, eps=eps)

    def update(self, rollouts, writer, j):
        for index, param_group in enumerate(self.optimizer.param_groups):
            # in the first 10 updates, don't update value_proj
            if self.pretrained and j < 10 and index == 1:
                param_group['lr'] = 0
            writer.add_scalar(f'lr/{index}', param_group['lr'], j)

        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        mean_advantages = advantages[torch.where(rollouts.bad_masks[:-1] != 0)].mean()
        std_advantages = advantages[torch.where(rollouts.bad_masks[:-1] != 0)].std()
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)
        # advantages = (advantages - advantages.mean()) / (
        #     advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        num_backwards = 0
        num_updates = 0
        clip_ratio_list = []
        KL_list = { 'total': [] }
        entropy_list = { 'total': [] }

        totol_batch_size = (rollouts.bad_masks[:-1] != 0).sum()
        accumulate_steps = np.ceil(totol_batch_size / (self.mini_batch_size * self.num_mini_batch))

        for e in range(self.ppo_epoch):
            self.optimizer.zero_grad()

            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                    advantages, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, mini_batch_size=self.mini_batch_size)


            cur_accumulate = 0
            for sample in data_generator:
                obs_batch, recurrent_hidden_states_batch, actions_batch, \
                   value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                        adv_targ = to_device(sample, self.device) # TODO:

                with torch.no_grad():
                    _, _, reference_entropy, _, reference_dists = self.reference.evaluate_actions(
                        obs_batch, recurrent_hidden_states_batch, masks_batch,
                        actions_batch, return_entropy_dist=True)
                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy, _, _, KL_dict = self.actor_critic.evaluate_actions(
                    obs_batch, recurrent_hidden_states_batch, masks_batch,
                    actions_batch, reference_dists)
                KL = sum(
                    KL_dict[k] * self.kl_coeff.get(k, 1)
                    for k in KL_dict
                )
                
                for k in KL_dict:
                    if k not in KL_list:
                        KL_list[k] = []
                    KL_list[k].append(KL_dict[k].item())
                KL_list['total'].append(KL.item())
                # if KL < self.d_targ / 1.5:
                #     self.beta /= 2
                # elif KL > self.d_targ * 1.5:
                #     self.beta *= 2

                for k in reference_dists:
                    if k not in entropy_list:
                        entropy_list[k] = []
                    entropy_list[k].append(reference_entropy[k].item())
                entropy_list['total'].append(sum(reference_entropy.values()).item())

                ratio = torch.exp(action_log_probs -
                                  old_action_log_probs_batch)
                
                clip_ratio = ((ratio < 1.0 - self.clip_param) | (ratio > 1.0 + self.clip_param)).float().mean()
                clip_ratio_list.append(clip_ratio.item())

                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + \
                        (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses,
                                                 value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                # self.optimizer.zero_grad()
                (value_loss * self.value_loss_coef + action_loss -
                 dist_entropy * self.entropy_coef + self.beta * KL).backward()

                cur_accumulate += 1

                if cur_accumulate == accumulate_steps:
                    nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                             self.max_grad_norm)
                    self.optimizer.step()
                    num_updates += 1
                    cur_accumulate = 0
                    self.optimizer.zero_grad()
                    if self.reference_ema is not None:
                        with torch.no_grad():
                            params = OrderedDict(self.actor_critic.named_parameters())
                            r_params = OrderedDict(self.reference.named_parameters())
                            for name, param in params.items():
                                r_params[name].sub_((1 - self.reference_ema) * (r_params[name] - param))

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
                num_backwards += 1

            if cur_accumulate > 0:
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()
                num_updates += 1
                cur_accumulate = 0
                self.optimizer.zero_grad()
                if self.reference_ema is not None:
                    with torch.no_grad():
                        params = OrderedDict(self.actor_critic.named_parameters())
                        r_params = OrderedDict(self.reference.named_parameters())
                        for name, param in params.items():
                            r_params[name].sub_((1 - self.reference_ema) * (r_params[name] - param))
        
        # writer.add_scalar('KL_with_reference', np.mean(KL_list), j)
        for k in KL_list:
            writer.add_scalar(f'KL/{k}', np.mean(KL_list[k]), j)
        for k in entropy_list:
            writer.add_scalar(f'entropy/{k}', np.mean(entropy_list[k]), j)
        writer.add_scalar('clip_ratio', np.mean(clip_ratio_list), j)
        # num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_backwards
        action_loss_epoch /= num_backwards
        dist_entropy_epoch /= num_backwards

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch
