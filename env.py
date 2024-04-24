import torch
import numpy as np
from poke_env.player import Gen5EnvSinglePlayer, Player
from poke_env.data import GenData

import embeddings
import rewards
import nets


class RLAgent:
    """
    Prototype Class for RLAgents
    """
    def __init__(self):
        self.device = None
        self.memory = None

        self.testing = False
    
    def train(self):
        pass
    
    def test(self):
        pass

    def select_action(self, state) -> np.ndarray:
        pass
    
    def replay_train(self) -> tuple[float, float]:
        pass

    def store(self, transition):
        pass
    
    def save(self):
        pass

class EpsilonGreedyDQN(RLAgent):
    def __init__(self, device, shape: list[int], obs_dim: int, action_dim: int, memory_size: int, batch_size: int, target_update: int, epsilon_decay: float,
                 max_epsilon: float = 1.0, min_epsilon: float = 0.1, gamma: float = 0.99):
        self.device = device

        self.batch_size = batch_size
        self.epsilon = max_epsilon
        self.epsilon_decay = epsilon_decay
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.target_update = target_update
        self.gamma = gamma

        self.action_dim = action_dim

        self.memory = nets.SimpleReplayBuffer(device, obs_dim, memory_size, batch_size)
        self.dqn = nets.Network(obs_dim, action_dim, shape).to(device)
        self.dqn_target = nets.Network(obs_dim, action_dim, shape).to(device)
        self.dqn_target.eval()

        self.optimizer = torch.optim.SGD(self.dqn.parameters())
        self.testing = False
        self.update_cnt = 0
    
    def train(self):
        self.testing = False
        self.update_cnt = 0
    
    def test(self):
        self.testing = True
    
    def select_action(self, state):
        if not self.testing:
            if self.epsilon > np.random.random():
                return None #Interpreted as random action
            else:
                selected_action = self.dqn(torch.tensor(state, dtype=torch.float, device=self.device)).argmax()
                selected_action = selected_action.detach().cpu().numpy()
                        
            self.transition = [state, selected_action]
        else:
            selected_action = self.dqn(torch.tensor(state, dtype=torch.float, device=self.device)).argmax()
            selected_action = selected_action.detach().cpu().numpy()

        return selected_action

    def store(self, transition: list[np.ndarray]):
        self.memory.store(*[torch.tensor(val, dtype=torch.float) for val in transition])

    def replay_train(self):
        if len(self.memory) >= self.batch_size:
            samples = self.memory.sample_batch()
            loss = self._compute_dqn_loss(samples)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.update_cnt += 1
            self.epsilon = max(self.min_epsilon, self.epsilon - (self.max_epsilon - self.min_epsilon) * self.epsilon_decay)
            if self.update_cnt % self.target_update == 0:
                self.dqn_target.load_state_dict(self.dqn.state_dict())
            return loss.item(), self.epsilon

    def _compute_dqn_loss(self, samples):
        #state, action, reward, next_state, done
        state = samples['state'].to(self.device)
        action = samples['action'].reshape(-1, 1).to(device = self.device, dtype = torch.long)
        reward = samples['reward'].reshape(-1, 1).to(self.device)
        next_state = samples['next_state'].to(self.device)
        done = samples['done'].reshape(-1, 1).to(self.device)

        curr_q_value = self.dqn(state).gather(1, action)
        next_q_value = self.dqn_target(next_state).max(dim=1, keepdim=True)[0].detach()
        target = (reward + self.gamma * next_q_value * (1 - done)).to(self.device)
        loss = torch.nn.functional.smooth_l1_loss(curr_q_value, target)

        return loss

class RLEnv(Gen5EnvSinglePlayer):
    def __init__(self, embed_type: str, reward_type: str, reward_params: tuple, **kwargs):
        self.gendata = GenData(5)
        self.embedding_fn, self.embedding_descriptor = embeddings.select_embedding(embed_type)
        self.reward_fn = rewards.select_reward(reward_type)
        self.reward_params = reward_params

        super(RLEnv, self).__init__(**kwargs)
        
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

class SelfPlayRLEnv(RLEnv):
    def potato(self):
        pass

class AgentPlayer(Player):
    def __init__(self, agent: RLAgent, env: RLEnv, **kwargs):
        self.agent = agent
        self.agent.test()

        self.env = env
        super(AgentPlayer, self).__init__(**kwargs)
    
    def choose_move(self, battle):
        embed = self.env.embed_battle(battle)
        action = self.agent.select_action(embed)
        return self.env.action_to_move(action, battle)