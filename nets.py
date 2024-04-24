import torch
import torch.nn as nn

class SimpleReplayBuffer:
    def __init__(self, device, obs_dim: int, size: int, batch_size: int = 32):
        self.obs_buf = torch.zeros([size, obs_dim]).to(device)
        self.next_obs_buf = torch.zeros([size, obs_dim]).to(device)
        self.acts_buf = torch.zeros(size).to(device)
        self.rewards_buf = torch.zeros(size).to(device)
        self.done_buf = torch.zeros(size).to(device)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size = 0, 0
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
    
class Network(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dims: list[int]):
        super(Network, self).__init__()
        shape = [in_dim] + hidden_dims + [out_dim]
        layers = []
        for i in range(len(shape)-1):
            layers.append(nn.Linear(shape[i], shape[i+1]))
            layers.append(nn.ReLU())
        del layers[-1] #Remove final ReLU

        self.layers = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor):
        return self.layers(x)
