import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import env, embeddings, rewards, agents

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

def train(agent: agents.RLAgent, env: env.RLEnv, num_shots: int, seed: int):
    agent.train()

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
            state, _ = env.reset(seed=seed)
            scores.append(score)
            score = 0
        
        vals = agent.replay_train()
        if vals is not None:
            losses.append(vals[0])
            epsilons.append(vals[1])
    
    plot(shot, scores, losses, epsilons)

    env.close()

def train_policy(agent: agents.RLAgent, env: env.RLEnv, num_shots: int, seed: int):
    agent.train()

    epsilons = []
    losses = []
    scores = []
    score = 0

    for shot in range(1, num_shots+1):
        if shot % 100 == 0:
            print(f"Shot: {shot}")
        
        state, _ = env.reset(seed=seed)
        action = agent.select_action(state)
        while True:
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store([state, action, reward, next_state, done])

            state = next_state
            score += reward
            if done:
                scores.append(score)
                score = 0
                break

            action = agent.select_action(state)
            
        vals = agent.replay_train()
        if vals is not None:
            losses.append(vals[0])
            epsilons.append(vals[1])
    
    # plot(shot, scores, losses, epsilons)

    env.close()

def save_results(dir, agent: agents.RLAgent, env: env.RLEnv, result_table):
    existing_folders = [f for f in os.listdir(dir) if os.path.isdir(os.path.join(dir, f))]
    new_folder = os.path.join(dir, f'Experiment{len(existing_folders)}/')
    os.makedirs(new_folder)
    print(f"Saving to {new_folder}")
    with open(os.path.join(new_folder, 'results.txt'), 'w') as f:
        f.write(result_table)
    agent.save(new_folder)
    env.save(new_folder)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    
    training_env = env.RLEnv(
        embed_type=embeddings.EMBED_DICT[0],
        reward_type=rewards.REW_DICT[0],
        reward_params=(2.0, 1.0, 6, 0.0, 30.0),
        battle_format="gen5randombattle",
        server_configuration=LocalhostServerConfiguration,
        start_challenging=True,
        opponent=opponent2
    )
    obs_dim = training_env.observation_space.shape[0]
    action_dim = training_env.action_space.n
    # dqn = agents.EpsilonGreedyDQN(device, [128, 128], obs_dim, action_dim, 1000, 32, 100, 1 / 2000)

    # train(dqn, training_env, num_shots=10000, seed=42)

    # dqn_player = env.AgentPlayer(
        # agent=dqn,
        # env=training_env,
        # battle_format='gen5randombattle',
        # server_configuration=LocalhostServerConfiguration           
    # )

    giga = agents.EpsilonGreedyGIGA(device, [128, 128], obs_dim, action_dim, 1 / 2000)

    train_policy(giga, training_env, num_shots=1000, seed=42)

    giga_player = env.AgentPlayer(
        agent = giga,
        env = training_env,
        battle_format = 'gen5randombattle',
        server_configuration=LocalhostServerConfiguration
    )

    n_challenges = 50
    players = [
        opponent, opponent2, opponent3, giga_player
    ]

    print("Beginning Cross Evaluation:")
    cross_eval_task = background_cross_evaluate(players, n_challenges)

    cross_evaluation = cross_eval_task.result()
    table = [["-"] + [p.username for p in players]]
    for p_1, results in cross_evaluation.items():
        table.append([p_1] + [cross_evaluation[p_1][p_2] for p_2 in results])
    print("Cross evaluation of DQN with baselines: ")
    tab_string = tabulate(table)
    print(tab_string)

    save_results('results/', giga, training_env, tab_string)
