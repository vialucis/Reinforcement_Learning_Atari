import gym
import torch
from utils import preprocess

if __name__ == '__main__':
    dqn = torch.load('models/CartPole-v0_best.pt', map_location=torch.device('cpu'))
    env = gym.make('CartPole-v0')
    env.render()
    obs = preprocess(env.reset(), env='CartPole-v0').unsqueeze(0)
    rewards = 0
    while True:
        action = dqn(obs).max(1)[1].item()
        obs, reward, done, info = env.step(action)
        rewards += reward
        if done:
            break
        obs = preprocess(obs, env='CartPole-v0').unsqueeze(0)
        env.render()

    print(rewards)


