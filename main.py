import torch
import numpy as np
import matplotlib.pyplot as plt
import env, embeddings, rewards

from poke_env.player import RandomPlayer, MaxBasePowerPlayer, SimpleHeuristicsPlayer, background_cross_evaluate
from poke_env import LocalhostServerConfiguration
from tabulate import tabulate

def plot(shot, scores, losses, epsilons):
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
    plt.savefig('train_results.png')

def train(agent, env: env.RLEnvPlayer, num_shots: int, seed: int):
    agent.testing = False

    state, _ = env.reset(seed=seed)
    epsilons = []
    losses = []
    scores = []
    score = 0

    for shot in range(1, num_shots+1):
        if shot % 1000 == 0:
            print(f"Shot: {shot}")
        
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.store([state, action, reward, next_state, done])

        state = next_state
        score += reward
        if done:
            state, _ = env.reset(seed)
            scores.append(score)
            score = 0
        
        agent.replay_train()
    
    plot(shot, scores, losses, epsilons)

    env.close()

if __name__ == "__main__":
    opponent = RandomPlayer(
        battle_format = "gen5randombattle",
        server_configuration=LocalhostServerConfiguration
    )
    opponent2 = MaxBasePowerPlayer(
        battle_format='gen5randombattle',
        server_configuration=LocalhostServerConfiguration
    )
    opponent3 = SimpleHeuristicsPlayer(
        battle_format='gen5randombattle',
        server_configuration=LocalhostServerConfiguration
    )
    
    testing_env = env.RLEnvPlayer(
        embed_type=embeddings.EMBED_DICT[0],
        reward_type=rewards.REW_DICT[0],
        reward_params=(2.0, 1.0, 6, 0.0, 30.0),
        battle_format="gen5randombattle",
        server_configuration=LocalhostServerConfiguration,
        start_challenging=True,
        opponent=opponent
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dqn = env.DQNAgent(device, [128, 128], testing_env, 1000, 32, 100, 1 / 2000, 42)

    dqn.train(10000)

    dqn_player = env.AgentPlayer(
        agent=dqn,
        battle_format='gen5randombattle',
        server_configuration=LocalhostServerConfiguration           
    )

    n_challenges = 50
    players = [
        opponent, opponent2, opponent3, dqn_player
    ]

    print("Beginning Cross Evaluation:")
    cross_eval_task = background_cross_evaluate(players, n_challenges)

    cross_evaluation = cross_eval_task.result()
    table = [["-"] + [p.username for p in players]]
    for p_1, results in cross_evaluation.items():
        table.append([p_1] + [cross_evaluation[p_1][p_2] for p_2 in results])
    print("Cross evaluation of DQN with baselines: ")
    print(tabulate(table))

    testing_env.close()