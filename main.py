from game import Game
from blob import Blob
from tqdm import tqdm
import numpy as np
from DQN1 import DQN
import matplotlib.pyplot as plt

DISP = True
epsilon = 1
EPISODES = 25_000
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

env = Game()
input_shape = (env.SIZE, env.SIZE, 3)
agent = DQN(input_shape, 4, DISP)
ep_rewards = []
steps = []
won = []
for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
    current_state = env.get_image()
    episode_reward = 0
    step = 1
    done = False
    while not done:
        if np.random.random() > epsilon or DISP:
            # Get action from Q table
            action = np.argmax(agent.get_qs(current_state))
        else:
            # Get random action
            action = np.random.randint(0, 4)

        new_state, reward, done = env.step(action)
        episode_reward += reward

        if DISP:
            env.render(5)
            if reward == env.FOOD_REWARD:
                print(f"Agent won in episode: {episode}")
            elif reward == -env.ENEMY_PENALTY:
                print(f"Agent lost in episode: {episode}")

        agent.update_replay_memory((np.asarray(current_state), action, reward, np.asarray(new_state), done))
        agent.train(done, step)

        current_state = new_state
        step += 1
    ep_rewards.append(episode_reward)
    steps.append(step)
    won.append(done)

    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)

    if episode % 100 == 0:
        x = np.array(won)
        print(np.unique(x, return_counts=True))
