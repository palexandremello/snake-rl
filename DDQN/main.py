import numpy as np
import gymnasium as gym

from agent import DDQNAgent
from frame_stack import make_env
from utils import plot_learning_curve


from gymnasium.envs.registration import register

register(
    id="snakegame-v0",
    entry_point="env:SnakeGameEnv",
)


if __name__ == "__main__":
    env = make_env("snakegame-v0")

    best_score = -np.inf
    load_checkpoint = False
    n_episodes = 50_000

    agent = DDQNAgent(
        gamma=0.99,
        epsilon=1.0,
        lr=0.00005,
        input_dims=(env.observation_space.shape),
        n_actions=env.action_space.n,
        mem_size=10000,
        eps_min=0.1,
        batch_size=32,
        replace=1000,
        eps_dec=1e-5,
        chkpt_dir="models/",
        algo="DDQNAgent",
        env_name="snakegame-v0",
    )

    if load_checkpoint:
        agent.load_models()

    fname = (
        agent.algo
        + "_"
        + agent.env_name
        + "_lr"
        + str(agent.lr)
        + "_"
        + "_"
        + str(n_episodes)
        + "episodes"
    )
    figure_file = "plots/" + fname + ".png"

    n_steps = 0
    scores, eps_history, steps_array = [], [], []

    for i in range(n_episodes):

        done = False
        score = 0
        observation, _ = env.reset()
        points = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, _, info = env.step(action)

            score += reward

            if not load_checkpoint:
                agent.store_transition(
                    observation, action, reward, observation_, int(done)
                )

                agent.learn()

            points += info["score"]
            observation = observation_
            n_steps += 1

        scores.append(score)
        steps_array.append(n_steps)

        avg_score = np.mean(scores[-100:])

        print(
            "episode",
            i,
            "reward: ",
            score,
            "average reward %.1f best reward %.1f epsilon %.2f"
            % (avg_score, best_score, agent.epsilon),
            "score points:" + str(points),
            "steps ",
            n_steps,
        )

        if avg_score > best_score:
            if not load_checkpoint:
                agent.save_models()

            best_score = avg_score

        eps_history.append(agent.epsilon)

    x = [i + 1 for i in range(len(scores))]
    plot_learning_curve(steps_array, scores, eps_history, figure_file)
