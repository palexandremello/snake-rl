import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
import numpy as np
import snake_game as sg

register(
    id="snakegame-v0",
    entry_point="__main__:SnakeGameEnv",  # Ajuste o entry_point para o nome correto do módulo onde está a classe SnakeGameEnv
)

class SnakeGameEnv(gym.Env):
    metadata = {
        "render_modes": ["human"],
        "render_fps": 15,  # Ajustei para ser consistente com o fps do jogo
    }

    def __init__(self, grid_rows=12, grid_cols=12, render_mode='human'):
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.render_mode = render_mode
        self.snake_game = sg.SnakeGame(
            grid_height=grid_rows, grid_width=grid_cols, fps=self.metadata["render_fps"]
        )

        self.action_space = spaces.Discrete(len(sg.SnakeAction))

        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(8,),
            dtype=np.float32,
        )

        self.steps_since_last_apple = 0

    def reset(self):
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

        reward = 0
        terminated = self.snake_game.game_over

        if terminated:
            reward = -1

        if self.snake_game.got_apple:
            reward += 10
            self.steps_since_last_apple = 0

        reward += -0.03

        if self.render_mode == "human":
            self.render()

        self.steps_since_last_apple += 1

        total_reward = reward

        info["score"] = self.snake_game.score
        info["n_steps"] = self.steps_since_last_apple

        return obs, total_reward, terminated, False, info

    def render(self):
        self.snake_game.render()

    def _get_obs(self):
        observation = self.snake_game.get_observation()
        return np.array(observation, dtype=np.float32)  # Converter a observação para numpy array

    def __calculate_distance_reward(self):
        distance = self.snake_game.snake_head_distance_to_apple
        if distance >= 1 and distance < 10:
            return 0.03  # Recompensa pequena se estiver próximo
        return -0.05  # Penalidade se estiver distante

if __name__ == "__main__":
    env = gym.make(
        "snakegame-v0", render_mode="human"
    )

    obs, _ = env.reset()

    print(env.observation_space)

    while True:
        rand_action = env.action_space.sample()
        obs, reward, terminated, _, info = env.step(rand_action)

        print(obs, reward, info)
        if terminated:
            obs, _ = env.reset()
