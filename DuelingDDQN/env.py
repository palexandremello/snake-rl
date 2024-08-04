import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env


import snake_game as sg
import numpy as np


register(
    id="snakegame-v0",
    entry_point="env:SnakeGameEnv",
)


class SnakeGameEnv(gym.Env):
    metadata = {
        "render_modes": ["human"],
        "render_fps": 15,
    }

    def __init__(self, grid_rows=32, grid_cols=32, render_mode=None):
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.render_mode = render_mode
        self.snake_game = sg.SnakeGame(
            grid_height=grid_rows, grid_width=grid_cols, fps=self.metadata["render_fps"]
        )

        self.action_space = spaces.Discrete(len(sg.SnakeAction))

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(
                self.grid_rows * self.snake_game.block_size,
                self.grid_cols * self.snake_game.block_size,
                3,
            ),
            dtype=np.float32,
        )

        self.steps_since_last_apple = 0

    def reset(self):
        super().reset()

        self.snake_game.reset()
        self.steps_since_last_apple = 0

        obs = self._get_obs()
        info = {}

        if self.render_mode == "human":
            self.render()

        return obs, info

    def step(self, action):
        info = {}
        action_enum = sg.SnakeAction(action)
        self.snake_game.step(action_enum)

        obs = self._get_obs()

        info["score"] = self.snake_game.score
        reward = 0
        terminated = self.snake_game.game_over

        if terminated:
            reward = -1

        if self.snake_game.got_apple:
            reward += 5
            self.steps_since_last_apple = 0
        else:
            self.steps_since_last_apple += 1
            if self.steps_since_last_apple > 100 and self.snake_game.score < 50:
                reward -= 0.01

        reward -= 0.04

        if self.render_mode == "human":
            self.render()

        self.steps_since_last_apple += 1


        total_reward = reward

        return obs, total_reward, terminated, False, info

    def render(self):
        self.snake_game.render()

    def _get_obs(self):
        screen = self.snake_game.get_screen()
        return screen

    def __calculate_distance_reward(self):
        distance = self.snake_game.snake_head_distance_to_apple
        if distance >= 1 and distance < 10:
            return 0.01

        return -0.04


if __name__ == "__main__":
    env = gym.make(
        "snakegame-v0", render_mode="human"
    )  

    obs, _ = env.reset()

    while True:
        rand_action = env.action_space.sample()
        obs, reward, terminated, _, _ = env.step(rand_action)

        print(obs)
        if terminated:
            obs, _ = env.reset()
