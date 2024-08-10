import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register
import matplotlib.pyplot as plt
from network import PolicyGradientAgent
from frame_stack import make_env

register(
    id="snakegame-v0",
    entry_point="env:SnakeGameEnv",
)



def plot_learning_curve(scores, x, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[(max(0, i-100)):(i+1)])
    
    plt.plot(x, running_avg)
    plt.title("Running average of previous 100 scores")
    plt.savefig(figure_file)



if __name__ == '__main__':
    env = make_env("snakegame-v0")

    n_games = 5000
    best_score = -np.inf

    agent = PolicyGradientAgent(gamma=0.99, lr=0.0005,
                                input_dims=env.observation_space.shape,
                                n_actions=env.action_space.n)
    

    fname = 'A2C_' + 'snakegame-v0' + str(agent.lr) + '_' + str(n_games) + 'games'

    figure_file = 'plots' + fname + '.png'

    scores = []
    for i in range(n_games):
        done = False
        observation, _ = env.reset()
        score = 0 
        n_steps = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, _, info = env.step(action)
            score += reward
            agent.store_rewards(reward)
            observation = observation_
            n_steps += 1
        agent.learn()
        scores.append(score)

        avg_score = np.mean(scores[-100:])
        print('episode ', i, 'score %.2f' % score, 'average score %.2f' % avg_score, 'best score %.2f' % best_score, 'score points ', info['score'])
        if avg_score > best_score:
            agent.save_model()
            best_score = avg_score

    
    x = [i+1 for i in range(len(scores))]
    plot_learning_curve(scores, x, figure_file)