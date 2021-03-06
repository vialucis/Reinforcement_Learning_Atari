import argparse

import gym
import torch
import torch.nn as nn

import config
from utils import preprocess
from evaluate import evaluate_policy
from dqn import DQN, ReplayMemory, optimize

import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--env', choices=['CartPole-v0'])
parser.add_argument('--evaluate_freq', type=int, default=25, help='How often to run evaluation.', nargs='?')
parser.add_argument('--evaluation_episodes', type=int, default=5, help='Number of evaluation episodes.', nargs='?')

# Hyperparameter configurations for different environments. See config.py.
ENV_CONFIGS = {
    'CartPole-v0': config.CartPole
}

if __name__ == '__main__':
    args = parser.parse_args()

    # Initialize environment and config.
    args.env = "CartPole-v0"
    env = gym.make(args.env).unwrapped
    env_config = ENV_CONFIGS["CartPole-v0"]

    # Initialize deep Q-networks.
    dqn = DQN(env_config=env_config).to(device)
    target_dqn = DQN(env_config=env_config).to(device)
    target_dqn.load_state_dict(dqn.state_dict())
    target_dqn.eval()
    print(env_config["train_frequency"])
    # Create replay memory.
    memory = ReplayMemory(env_config['memory_size'])

    # Initialize optimizer used for training the DQN. We use Adam rather than RMSProp.
    optimizer = torch.optim.Adam(dqn.parameters(), lr=env_config['lr'])

    # Keep track of best evaluation mean return achieved so far.
    best_mean_return = -float("Inf")

    mean_return_history = []

    for episode in range(env_config['n_episodes']):
        done = False

        obs = preprocess(env.reset(), env=args.env).unsqueeze(0)
        steps = 0

        while not done:
            steps += 1

            action = dqn.act(obs)

            # Act in the true environment.
            obs_old = obs
            obs, reward, done, info = env.step(action.item())
            # Preprocess incoming observation.
            if not done:
                obs = preprocess(obs, env=args.env).unsqueeze(0)
            # TODO: Add the transition to the replay memory. Remember to convert
            #       everything to PyTorch tensors!
            reward = torch.tensor([reward], device=device)
            memory.push(obs_old, action, obs, reward)

            # TODO: Run DQN.optimize() every env_config["train_frequency"] steps.
            if episode % env_config["train_frequency"] == 0:
                optimize(dqn, target_dqn, memory, optimizer)

            # TODO: Update the target network every env_config["target_update_frequency"] steps.
            if episode % env_config["target_update_frequency"] == 0:
                target_dqn.load_state_dict(dqn.state_dict())

        # Evaluate the current agent.
        if episode % args.evaluate_freq == 0:
            mean_return = evaluate_policy(dqn, env, env_config, args, n_episodes=args.evaluation_episodes)
            mean_return_history.append(min(mean_return, 500))
            torch.save(mean_return_history, f'models/Tuf_{env_config["target_update_frequency"]}/{args.env}_tuf_{env_config["target_update_frequency"]}_return_history.pt')

            if episode % (args.evaluate_freq*40) == 0:
                plt.plot(range(0, len(mean_return_history) * args.evaluate_freq, args.evaluate_freq), mean_return_history)
                plt.axhline(y=200, color='r', linestyle='-')
                plt.xlabel("Episode")
                plt.ylabel("Mean return")
                plt.title("Mean return over episodes")
                plt.savefig(f'models/Tuf_{env_config["target_update_frequency"]}/{args.env}_tuf_{env_config["target_update_frequency"]}_episode_{episode}.png')

            print(f'Episode {episode}/{env_config["n_episodes"]}: {mean_return}')

            # Save current agent if it has the best performance so far.
            if mean_return >= best_mean_return:
                best_mean_return = mean_return

                print('Best performance so far! Saving model.')
                torch.save(dqn, f'models/Tuf_{env_config["target_update_frequency"]}/{args.env}_tuf_{env_config["target_update_frequency"]}.pt')

    plt.plot(range(0, len(mean_return_history) * args.evaluate_freq, args.evaluate_freq), mean_return_history)
    plt.axhline(y=200, color='r', linestyle='-')
    plt.xlabel("Episode")
    plt.ylabel("Mean return")
    plt.title("Mean return over episodes")
    plt.savefig(f'models/Tuf_{env_config["target_update_frequency"]}/{args.env}_tuf_{env_config["target_update_frequency"]}.png')
    plt.show()

    # Close environment after training is completed.
    env.close()
