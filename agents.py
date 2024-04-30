import os
import json
import torch
import numpy as np

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
    
    def save(self, path):
        pass

    def load(self, path):
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
        self.dqn.train()
    
    def test(self):
        self.testing = True
        self.dqn.eval()
    
    def select_action(self, state):
        if not self.testing and self.epsilon > np.random.random():
            selected_action = np.random.randint(low=0, high=self.action_dim)
        else:
            selected_action = self.dqn(torch.tensor(state, dtype=torch.float, device=self.device)).argmax()
            selected_action = selected_action.detach().cpu().numpy()
        
        return selected_action, None
            
    def store(self, transition: list[np.ndarray]):
        self.memory.store(*[torch.tensor(val, dtype=torch.float) for val in transition])

    def replay_train(self):
        if len(self.memory) >= self.batch_size:
            samples = self.memory.sample_batch()
            #state, action, reward, next_state, done
            state = samples['state'].to(self.device)
            action = samples['action'].reshape(-1, 1).to(device = self.device, dtype = torch.long)
            reward = samples['reward'].reshape(-1, 1).to(self.device)
            next_state = samples['next_state'].to(self.device)
            done = samples['done'].reshape(-1, 1).to(self.device)
            curr_q_value = self.dqn(state).gather(1, action)
            #Detach to keep target Q-Network frozen
            next_q_value = self.dqn_target(next_state).max(dim=1, keepdim=True)[0].detach()
            target = (reward + self.gamma * next_q_value * (1 - done)).to(self.device)
            loss = torch.nn.functional.smooth_l1_loss(curr_q_value, target)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.update_cnt += 1
            self.epsilon = max(self.min_epsilon, self.epsilon - (self.max_epsilon - self.min_epsilon) * self.epsilon_decay)
            if self.update_cnt % self.target_update == 0:
                self.dqn_target.load_state_dict(self.dqn.state_dict())
            return loss.item(), self.epsilon
        
    def save(self, directory):
        config_path = os.path.join(directory, 'AlgorithmConfig.json')
        params = dict(
            algorithm="EpsilonGreedyDQN",
            epsilon_decay=self.epsilon_decay,
            max_epsilon=self.max_epsilon,
            min_epsilon=self.min_epsilon,
            target_update=self.target_update,
            gamma=self.gamma,
            batch_size=self.batch_size
        )
        with open(config_path, 'w') as f:
            json.dump(params, f)

        models_path = os.path.join(directory, 'net.pt')
        torch.save(self.dqn, models_path)

        models_path = os.path.join(directory, 'target_net.pt')
        torch.save(self.dqn_target, models_path)
        
        optim_path = os.path.join(directory, 'optimizer.pt')
        torch.save(self.optimizer, optim_path)

    def load(self, directory):
        pass

def _projection(x: torch.Tensor):
    p = x.detach().clone()
    t = (1.0 - p.sum()) / len(p)
    p += t
    while True:
        n = 0
        excess = 0
        for i in range(len(p)):
            if p[i] < 0:
                excess -= p[i]
                p[i] = 0
            elif p[i] > 0:
                n += 1
        if excess == 0:
            break
        else:
            for i in range(len(p)):
                if p[i] > 0:
                    p[i] -= excess / n 
    return p

class PolicyGIGA(RLAgent):
    def __init__(self, device, shape: list[int], obs_dim: int, action_dim: int, gamma: float = 0.99):
        self.device = device

        self.gamma = gamma

        self.action_dim = action_dim
        
        self.memory = nets.EpisodeReplayBuffer(device, obs_dim)
        self.pi_net = nets.PolicyNet(obs_dim, action_dim, shape).to(device)
        self.pi_target = nets.PolicyNet(obs_dim, action_dim, shape).to(device)
        self.pi_target.eval()

        self.x_optimizer = torch.optim.SGD(self.pi_net.parameters())
        self.z_optimizer = torch.optim.SGD(self.pi_target.parameters())
        self.testing = False

    def train(self):
        self.testing = False
        self.pi_net.train()
    
    def test(self):
        self.testing = True
        self.pi_net.eval()

    def select_action(self, state) -> np.ndarray:
        probs = self.pi_net(torch.tensor(state, dtype=torch.float, device=self.device).unsqueeze(dim=0))
        action_distr = torch.distributions.Categorical(probs.squeeze())
        selected_action = action_distr.sample()
        log_prob = action_distr.log_prob(selected_action)
        selected_action = selected_action.detach().cpu().numpy()
        
        return selected_action, log_prob
    
    def replay_train(self) -> tuple[float, float]:
        samples = self.memory.sample()

        #Episode is finished
        if samples['done'][-1] == 1:
            states = samples['state'].to(self.device)
            actions = samples['action'].reshape(-1, 1).to(device = self.device, dtype = torch.long)
            reward = samples['reward'].reshape(-1, 1).to(self.device)
            x_logprobs = samples['log_probs'].reshape(-1, 1).to(self.device)
            
            T = reward.shape[0]
            deltas = torch.zeros(T)
            deltas[-1] = reward[-1]
            for t in reversed(range(T - 1)):
                deltas[t] = reward[t] + self.gamma * deltas[t+1]
            deltas = deltas.to(self.device)
            
            for idx in range(len(actions)):
                action, state, delta = actions[idx], states[idx].unsqueeze(dim=0), deltas[idx]
                z_logprobs = self._action_logprobs(state, action)
                adj = torch.nn.functional.one_hot(action, num_classes=self.action_dim).to(self.device)
                #Eqn(1) GIGA: Add, because subtract negative G_t
                state_action_distrs = self.pi_net(state)
                greedy_projection = _projection(state_action_distrs.squeeze() + (adj * x_logprobs[idx] * delta).squeeze())
                loss = torch.nn.functional.smooth_l1_loss(state_action_distrs.squeeze(), greedy_projection)
                self.x_optimizer.zero_grad()
                loss.backward()
                self.x_optimizer.step()

                #Smaller step for z policy --eqn(2)
                target_sa_distr = self.pi_target(state)
                z_greedy = _projection(target_sa_distr.squeeze() + (adj * z_logprobs * delta).squeeze())
                loss = torch.nn.functional.smooth_l1_loss(target_sa_distr.squeeze(), z_greedy)
                self.z_optimizer.zero_grad()
                loss.backward()
                self.z_optimizer.step()

                #Eqn(3) : 2nd x update
                new_x = self.pi_net(state)
                new_z = self.pi_target(state).detach()
                step_size = min(1, torch.nn.functional.l1_loss(new_z, target_sa_distr.detach()) / torch.nn.functional.l1_loss(new_z, new_x.detach()))

                loss = step_size * torch.nn.functional.smooth_l1_loss(new_z, new_x)
                self.x_optimizer.zero_grad()
                loss.backward()
                self.x_optimizer.step()

            self.memory.zero()
    
    def _action_logprobs(self, state, action):
        z_action_probs = self.pi_target(state)
        z_distr = torch.distributions.Categorical(z_action_probs)
        z_log = z_distr.log_prob(action)

        return z_log

    def store(self, transition):
        lst = [torch.tensor(val, dtype=torch.float).unsqueeze(dim=0) for val in transition[:-1]]
        lst.append(transition[-1].clone().detach().unsqueeze(dim=0).requires_grad_(True))
        self.memory.store(*lst)
    
    def save(self, directory):
        config_path = os.path.join(directory, 'AlgorithmConfig.json')
        params = dict(
            algorithm="PolicyGIGA",
            gamma=self.gamma,
        )
        with open(config_path, 'w') as f:
            json.dump(params, f)

        models_path = os.path.join(directory, 'x_net.pt')
        torch.save(self.pi_net, models_path)

        models_path = os.path.join(directory, 'z_net.pt')
        torch.save(self.pi_target, models_path)
        
        optim_path = os.path.join(directory, 'x_optimizer.pt')
        torch.save(self.x_optimizer, optim_path)

        optim_path = os.path.join(directory, 'z_optimizer.pt')
        torch.save(self.z_optimizer, optim_path)

    def load(self, path):
        pass

class REINFORCE(RLAgent):
    def __init__(self, device, shape: list[int], obs_dim: int, action_dim: int, gamma: float = 0.99):
        self.device = device
        self.gamma = gamma
        self.action_dim = action_dim
        
        self.memory = nets.EpisodeReplayBuffer(device, obs_dim)
        self.net = nets.PolicyNet(obs_dim, action_dim, shape).to(device)

        self.optimizer = torch.optim.SGD(self.net.parameters())
        self.testing = False

    def train(self):
        self.testing = False
        self.net.train()
    
    def test(self):
        self.testing = True
        self.net.eval()

    def select_action(self, state) -> np.ndarray:
        probs = self.net(torch.tensor(state, dtype=torch.float, device=self.device))
        action_distr = torch.distributions.Categorical(probs)
        selected_action = action_distr.sample()
        log_prob = action_distr.log_prob(selected_action)
        selected_action = selected_action.detach().cpu().numpy()
        
        return selected_action, log_prob
    
    def replay_train(self) -> tuple[float, float]:
        samples = self.memory.sample()

        #Episode is finished
        if samples['done'][-1] == 1:
            reward = samples['reward'].reshape(-1, 1).to(self.device)
            log_probs = samples['log_probs'].reshape(-1, 1).to(self.device)
            
            #Calculate rewards
            T = reward.shape[0]
            deltas = torch.zeros(T)
            deltas[-1] = reward[-1]
            for t in reversed(range(T - 1)):
                deltas[t] = reward[t] + self.gamma * deltas[t+1]
            deltas = deltas.to(self.device).detach()

            loss = -(deltas * log_probs).sum()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.memory.zero()

            return loss.item(), None

    def store(self, transition):
        lst = [torch.tensor(val, dtype=torch.float).unsqueeze(dim=0) for val in transition[:-1]]
        lst.append(transition[-1].clone().detach().unsqueeze(dim=0).requires_grad_(True))
        self.memory.store(*lst)
    
    def save(self, directory):
        config_path = os.path.join(directory, 'AlgorithmConfig.json')
        params = dict(
            algorithm="REINFORCE",
            gamma=self.gamma
        )
        with open(config_path, 'w') as f:
            json.dump(params, f)

        models_path = os.path.join(directory, 'net.pt')
        torch.save(self.net, models_path)

        optim_path = os.path.join(directory, 'optimizer.pt')
        torch.save(self.optimizer, optim_path)

    def load(self, path):
        pass

class AdvantageActorCritic(RLAgent):
    def __init__(self, device, shape: list[int], obs_dim: int, action_dim: int, gamma: float = 0.99):
        self.device = device

        self.gamma = gamma
        self.action_dim = action_dim

        self.memory = nets.EpisodeReplayBuffer(device, obs_dim)
        self.policy = nets.ActorCritic(
            nets.PolicyNet(obs_dim, action_dim, shape), 
            nets.Network(obs_dim, action_dim, shape)
        ).to(device)
        self.optimizer = torch.optim.SGD(self.policy.parameters())
        self.testing = False
    
    def train(self):
        self.testing = False
        self.policy.train()
    
    def test(self):
        self.testing = True
        self.policy.eval()

    def select_action(self, state) -> np.ndarray:
        probs, _ = self.policy(torch.tensor(state, dtype=torch.float, device=self.device))
        action_distr = torch.distributions.Categorical(probs)
        selected_action = action_distr.sample()
        log_prob = action_distr.log_prob(selected_action)
        selected_action = selected_action.detach().cpu().numpy()
        
        return selected_action, log_prob
    
    def replay_train(self) -> tuple[float, float]:
        samples = self.memory.sample()

        #Episode is finished
        if samples['done'][-1] == 1:
            states = samples['state'].to(self.device)
            actions = samples['action'].to(device = self.device, dtype = torch.long).unsqueeze(dim=-1)
            reward = samples['reward'].reshape(-1, 1).to(self.device)
            log_probs = samples['log_probs'].reshape(-1, 1).to(self.device)
            
            #Calculate rewards
            T = reward.shape[0]
            deltas = torch.zeros(T)
            deltas[-1] = reward[-1]
            for t in reversed(range(T - 1)):
                deltas[t] = reward[t] + self.gamma * deltas[t+1]
            deltas = deltas.to(self.device).detach()
            _, values = self.policy(states)
            values = values.gather(1, actions)
            advantage = deltas - values.detach().clone()

            value_loss = torch.nn.functional.smooth_l1_loss(deltas, values.squeeze())
            policy_loss = -(advantage * log_probs).sum()
            
            self.optimizer.zero_grad()
            policy_loss.backward()
            value_loss.backward()
            self.optimizer.step()
            
            self.memory.zero()

            return policy_loss.item(), value_loss.item()

    def store(self, transition):
        lst = [torch.tensor(val, dtype=torch.float).unsqueeze(dim=0) for val in transition[:-1]]
        lst.append(transition[-1].clone().detach().unsqueeze(dim=0).requires_grad_(True))
        self.memory.store(*lst)
    
    def save(self, directory):
        config_path = os.path.join(directory, 'AlgorithmConfig.json')
        params = dict(
            algorithm="ActorCritic",
            gamma=self.gamma
        )
        with open(config_path, 'w') as f:
            json.dump(params, f)

        models_path = os.path.join(directory, 'net.pt')
        torch.save(self.policy, models_path)

        optim_path = os.path.join(directory, 'optimizer.pt')
        torch.save(self.optimizer, optim_path)

    def load(self, path):
        pass


class WPL(RLAgent):
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
    
    def save(self, path):
        pass

    def load(self, path):
        pass