import torch
import numpy as np
from gymnasium.spaces import Space, Box
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.player import Gen5EnvSinglePlayer, OpenAIGymEnv
import poke_env.data

import embeddings

class SimpleReplayBuffer:
    def __init__(self, device, obs_dim: int, size: int, batch_size: int = 32):
        self.obs_buf = torch.zeros([size, obs_dim]).to(device)
        self.next_obs_buf = torch.zeros([size, obs_dim]).to(device)
        self.acts_buf = torch.zeros(size).to(device)
        self.rewards_buf = torch.zeros(size).to(device)
        self.done_buf = torch.zeros(size).to(device)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.cur_size = 0, 0
        self.device = device
    
    def store(self, state: torch.Tensor, action: torch.Tensor, reward: torch.Tensor,
              next_state: torch.Tensor, done: bool):
        self.obs_buf[self.ptr] = state
        self.acts_buf[self.ptr] = action
        self.rewards_buf[self.ptr] = reward
        self.next_obs_buf[self.ptr] = next_state
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def sample_batch(self):
        idxs = torch.randperm(self.size)[:self.batch_size]
        return dict(
            state = self.obs_buf[idxs],
            action = self.acts_buf[idxs],
            reward = self.rewards_buf[idxs],
            next_state = self.next_obs_buf[idxs],
            done = self.done_buf[idxs]
        )
    
    def __len__(self):
        return self.size

class DQNAgent:
    def __init__(self, device, env: OpenAIGymEnv, memory_size: int, batch_size: int, target_update: int, epsilon_decay: float,
                 seed: int, max_epsilon: float = 1.0, min_epsilon: float = 0.1, gamma: float = 0.99):
        self.device = device
        self.env = env

        self.epsilon = max_epsilon
        self.epsilon_decay = epsilon_decay
        self.seed = seed
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.target_update = target_update
        self.gamma = gamma


        

class DQNPlayer(Gen5EnvSinglePlayer):
    def __init__(self, **kwargs):
        super(DQNPlayer, self).__init__(**kwargs)
        self.gendata = poke_env.data.GenData(5)

    def calc_reward(self, last_battle, current_battle) -> float:
        return self.reward_computing_helper(
            current_battle, fainted_value=2.0, hp_value=1.0, victory_value=30.0
        )

    def embed_battle(self, battle: AbstractBattle):
        return embeddings.ex_embedding(battle, self.gendata)

    def describe_embedding(self) -> Space:
        low = [-1, -1, -1, -1, 0, 0, 0, 0, 0, 0]
        high = [3, 3, 3, 3, 4, 4, 4, 4, 1, 1]
        return Box(
            np.array(low, dtype=np.float32),
            np.array(high, dtype=np.float32),
            dtype=np.float32
        )


if __name__ == "__main__":
    from poke_env.player import RandomPlayer
    from poke_env import LocalhostServerConfiguration
    from gymnasium.utils.env_checker import check_env
    opponent = RandomPlayer(
        battle_format = "gen5randombattle",
        server_configuration=LocalhostServerConfiguration
    )
    
    test_env = DQNPlayer(
        battle_format="gen5randombattle",
        server_configuration=LocalhostServerConfiguration,
        start_challenging=True,
        opponent=opponent
    )

    check_env(test_env)
    print(test_env.describe_embedding().shape[0])
    test_env.close()