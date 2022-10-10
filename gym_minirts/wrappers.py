from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import gym
from gym_minirts.set_path import append_sys_path
from gym_minirts.utils import make_batch, Args, get_game_option, get_ai_options
from gym_minirts.main import MiniRTSEnv

append_sys_path()
import tube


class VecMiniRTSEnv(gym.Wrapper):
    envs: List[gym.Env]

    def __init__(
        self, num_games: int, game_option, ai1_option, ai2_option, level: str = 'medium', single_action = False, kind_enemy = 5
    ):
        self.num_games = num_games
        self.game_option = game_option
        self.ai1_option = ai1_option
        self.ai2_option = ai2_option
        self.envs = []
        self.context = tube.Context()
        for i in range(self.num_games):
            self.envs.append(
                gym.make(
                    'minirts-v0',
                    idx=i,
                    context=self.context,
                    game_option=self.game_option,
                    ai1_option=self.ai1_option,
                    ai2_option=self.ai2_option,
                    level=level,
                    single_action=single_action,
                    kind_enemy=kind_enemy
                )
            )
        self.started = False
        self.executor = ThreadPoolExecutor(max_workers=num_games)

    @property
    def action_space(self):
        return self.envs[0].action_space

    def single_step(self, i, action):
        env = self.envs[i]
        obs, reward, done, info = env.step(action)
        if done:
            obs = env.reset()
        return obs, reward, done, info

    def step(self, action):
        # obs_list = []
        # rewards = []
        # done_list = []
        # infos = []
        # for i, env in enumerate(self.envs):
        #     action_ = action[i, None]
        #     obs, reward, done, info = env.step(action_)
        #     if done:
        #         obs = env.reset()
        #     obs_list.append(obs)
        #     rewards.append(reward)
        #     done_list.append(done)
        #     infos.append(info)
        # obs = make_batch(obs_list)
        # rewards = torch.tensor(rewards).unsqueeze(1)
        # done = np.array(done_list)
        # return obs, rewards, done, infos
        result = list(
            self.executor.map(
                self.single_step,
                range(self.num_games),
                (action[i, None] for i in range(self.num_games)),
            )
        )
        obs_list, rewards, done_list, infos = tuple(
            [tmp[i] for tmp in result] for i in range(4)
        )
        obs = make_batch(obs_list)
        rewards = torch.tensor(rewards).unsqueeze(1)
        done = np.array(done_list)
        return obs, rewards, done, infos

    # TODO:

    def reset(self):
        assert not self.started, 'resetting a runntime environment is not supported'
        self.started = True
        self.context.start()
        return make_batch(list(self.executor.map(lambda env: env.reset(), self.envs)))

    def close(self):
        if self.envs is not None:
            for env in self.envs:
                env.close()


class VecMiniRTSEnvFixedHorizon(VecMiniRTSEnv):
    def __init__(
        self, num_games: int, game_option, ai1_option, ai2_option, level: str = 'medium', single_action = False, kind_enemy=5
    ):
        super().__init__(num_games, game_option, ai1_option, ai2_option, level, single_action, kind_enemy)
        self.running = [False] * num_games
        self.obs = [None] * num_games

    def single_step(self, i, action):
        env = self.envs[i]
        if self.running[i]:
            obs, reward, done, info = env.step(action)
            self.obs[i] = obs
            if done:
                self.running[i] = False
        else:
            obs = self.obs[i]
            reward = 0
            done = True
            info = {'bad_transition': True}
        return obs, reward, done, info

    def reset(self, **kwargs):
        assert not any(self.running)
        if not self.started:
            self.context.start()
            self.started = True
        for i, env in enumerate(self.envs):
            self.running[i] = True
        self.obs = list(self.executor.map(lambda env: env.reset(**kwargs), self.envs))
        return make_batch([obs for obs in self.obs])


def make_envs(num_games: int, args: Args):
    game_option = get_game_option(args)
    ai1_option, ai2_option = get_ai_options(args, 500)
    envs = VecMiniRTSEnvFixedHorizon(
        num_games, game_option, ai1_option, ai2_option, level=args.level, single_action=args.single_action, kind_enemy=args.kind_enemy
    )
    return envs
