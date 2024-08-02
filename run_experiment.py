from hypertunning.optimizer import Optmizer
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

model_path = os.path.join("models", "best_model.pkl")
params_path = os.path.join("models", "best_params.pkl")

if not os.path.exists("models"):
    os.makedirs("models")

optimizer = Optmizer(
    model_name="DQN",
    env={"name": "snakegame-v0", "entrypoint": "envs.pygame.snake_env:SnakeGameEnv"},
    episodes=100,
    model_path=model_path,
    params_path=params_path
)

optimizer.optimize(n_trials=1, save_params=True)

with open(params_path, 'rb') as f:
    best_params = pickle.load(f)

print("Best hyperparameters found:", best_params)
