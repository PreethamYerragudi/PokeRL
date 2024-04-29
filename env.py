import os, json
from poke_env.player import Gen5EnvSinglePlayer, Player
from poke_env.data import GenData

import embeddings
import rewards
import agents

class RLEnv(Gen5EnvSinglePlayer):
    def __init__(self, embed_type: str, reward_type: str, reward_params: tuple, **kwargs):
        self.gendata = GenData(5)

        self.embedding_type = embed_type
        self.reward_type = reward_type

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

    def save(self, directory):
        params = dict(
            battle_format = self._DEFAULT_BATTLE_FORMAT,
            embedding_type = self.embedding_type,
            reward_type = self.reward_type,
            reward_params = self.reward_params,
            opponent=str(type(self._opponent))
        )
        with open(os.path.join(directory, 'envparams.json'), 'w') as f:
            json.dump(params, f)

class SelfPlayRLEnv(RLEnv):
    def potato(self):
        pass

class AgentPlayer(Player):
    def __init__(self, agent: agents.RLAgent, env: RLEnv, **kwargs):
        self.agent = agent
        self.agent.test()

        self.env = env
        super(AgentPlayer, self).__init__(**kwargs)
    
    def choose_move(self, battle):
        embed = self.env.embed_battle(battle)
        action = self.agent.select_action(embed)[0]
        return self.env.action_to_move(action, battle)