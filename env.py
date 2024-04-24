import torch
import numpy as np
from poke_env.player import Gen5EnvSinglePlayer, OpenAIGymEnv
from poke_env.data import GenData

import matplotlib.pyplot as plt

import embeddings
import rewards
import nets

class DQNAgent:
    def __init__(self, device, shape: list[int], env: OpenAIGymEnv, memory_size: int, batch_size: int, target_update: int, epsilon_decay: float,
                 seed: int, max_epsilon: float = 1.0, min_epsilon: float = 0.1, gamma: float = 0.99):
        self.device = device
        self.env = env

        self.batch_size = batch_size
        self.epsilon = max_epsilon
        self.epsilon_decay = epsilon_decay
        self.seed = seed
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.target_update = target_update
        self.gamma = gamma

        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        self.memory = nets.SimpleReplayBuffer(device, obs_dim, memory_size, batch_size)
        self.dqn = nets.Network(obs_dim, action_dim, shape).to(device)
        self.dqn_target = nets.Network(obs_dim, action_dim, shape).to(device)
        self.dqn_target.eval()

        self.transition = list()
        self.optimizer = torch.optim.SGD(self.dqn.parameters())
        self.testing = False
    
    def select_action(self, state):
        if not self.testing:
            if self.epsilon > np.random.random():
                selected_action = self.env.action_space.sample()
            else:
                selected_action = self.dqn(torch.tensor(state, dtype=torch.float, device=self.device)).argmax()
                selected_action = selected_action.detach().cpu().numpy()
                        
                self.transition = [state, selected_action]
        else:
            selected_action = self.dqn(torch.tensor(state, dtype=torch.float, device=self.device)).argmax()
            selected_action = selected_action.detach().cpu().numpy()

        return selected_action

    def step(self, action):
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated

        if not self.is_testing:
            self.transition += [reward, next_state, done]
            self.memory.store(*[torch.tensor(val ,dtype=torch.float) for val in self.transition])
        
        return next_state, reward, done

    def train(self, num_shots: int, plotting_interval: int=200):
        self.testing = False

        state, _ = self.env.reset(seed=self.seed)
        update_cnt = 0
        epsilons = []
        losses = []
        scores = []
        score = 0
        for shot in range(1, num_shots+1):
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward

            if done:
                state, _ = self.env.reset(seed=self.seed)
                scores.append(score)
                score = 0
            if len(self.memory >= self.batch_size):
                samples = self.memory.sample_batch()
                loss = self._compute_dqn_loss(samples)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                losses.append(loss.item())
                update_cnt += 1

                self.epsilon = max(self.min_epsilon, self.epsilon - (self.max_epsilon - self.min_epsilon) * self.epsilon_decay)
                epsilons.append(self.epsilon)

                if update_cnt % self.target_update == 0:
                    self.dqn_target.load_state_dict(self.dqn.state_dict())

            if shot % plotting_interval == 0:
                self._plot(shot, scores, losses, epsilons)

        self.env.close()
    
    def test(self):
        self.testing = True

        state, _ = self.env.reset(seed=self.seed)
        done = False
        score = 0

        while not done:
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward

        print(f"score: {score}")
        self.env.close()


    def _compute_dqn_loss(self, samples):
        #state, action, reward, next_state, done
        state = samples['state'].to(self.device)
        action = samples['action'].reshape(-1, 1).to(self.device)
        reward = samples['reward'].reshape(-1, 1).to(self.device)
        next_state = samples['next_state'].to(self.device)
        done = samples['done'].reshape(-1, 1).to(self.device)

        curr_q_value = self.dqn(state).gather(1, action)
        next_q_value = self.dqn_target(next_state).max(dim=1, keepdim=True)[0].detach()
        target = (reward + self.gamma * next_q_value * (1 - done)).to(self.device)
        loss = torch.nn.functional.smooth_l1_loss(curr_q_value, target)

        return loss

    def _plot(self, shot, scores, losses, epsilons):
        plt.figure(figsize=(20,5))
        plt.subplot(131)
        plt.title(f'shot {shot}, score: {np.mean(scores[-10:])}')
        plt.plot(scores)
        plt.subplot(132)
        plt.title('loss')
        plt.plot(losses)
        plt.subplot(133)
        plt.title('epsilons')
        plt.plot(epsilons)
        plt.show()


class RLEnvPlayer(Gen5EnvSinglePlayer):
    def __init__(self, embed_type: str, reward_type: str, reward_params: tuple, **kwargs):
        self.gendata = GenData(5)
        self.embedding_fn, self.embedding_descriptor = embeddings.select_embedding(embed_type)
        self.reward_fn = rewards.select_reward(reward_type)
        self.reward_params = reward_params

        super(RLEnvPlayer, self).__init__(**kwargs)
        
    def calc_reward(self, last_battle, current_battle) -> float:
        #Last battle is last battle state
        if current_battle not in self._reward_buffer:
            self._reward_buffer[current_battle] = 0.0
        new_reward = self.reward_fn(last_battle, current_battle, self.reward_params)

        ret_val = new_reward - self._reward_buffer[current_battle]
        self._reward_buffer[current_battle] = new_reward

        return ret_val

    def embed_battle(self, battle):
        return self.embedding_fn(battle, self.gendata)

    def describe_embedding(self):
        return self.embedding_descriptor()


if __name__ == "__main__":
    from poke_env.player import RandomPlayer
    from poke_env import LocalhostServerConfiguration
    from gymnasium.utils.env_checker import check_env
    opponent = RandomPlayer(
        battle_format = "gen5randombattle",
        server_configuration=LocalhostServerConfiguration
    )
    
    testing_env = RLEnvPlayer(
        embed_type=embeddings.EMBED_DICT[0],
        reward_type=rewards.REW_DICT[0],
        reward_params=(2.0, 1.0, 6, 0.0, 30.0),
        battle_format="gen5randombattle",
        server_configuration=LocalhostServerConfiguration,
        start_challenging=True,
        opponent=opponent
    )

    check_env(testing_env)
    print(testing_env.observation_space.shape[0], testing_env.action_space.n)
    testing_env.close()