

from main.model_types import RLModelType
from main.train import Trainer
import pickle
import optuna

class Optmizer:

    def __init__(self, model_name, env, episodes, model_path, params_path):

        self.model_name = model_name
        self.env = env
        self.model_path = model_path
        self.params_path = params_path
        self.episodes = episodes
    

    def objective(self, trial, n_episodes=10):
        lr = trial.suggest_loguniform('lr', 5e-5, 0.99)
        gamma = trial.suggest_uniform('gamma', 0.01, 0.9999)
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
        target_replace_frequency = trial.suggest_categorical('target_replace_frequency', 
                                                             [500, 1000, 2000])
        min_eps = trial.suggest_loguniform('min_eps', 1e-5, 0.5)
        eps_decay = trial.suggest_loguniform('eps_decay',  0.0001, 0.2)
        trainer = Trainer(model_name="DQN", model_type=RLModelType.DQN, game_type="pygame")
        trainer.setup_env({"name": "snakegame-v0",
                       "entrypoint": "envs.pygame.snake_env:SnakeGameEnv"})
        agent_params = {
            "gamma": gamma,
            "lr": lr,
            "batch_size": batch_size,
            "target_replace_frequency": target_replace_frequency,
            "eps_decay": eps_decay,
            "min_eps": min_eps}
        
        trainer.setup_agent(agent_params)
        total_rewards, _ = trainer.trainer(self.episodes)
        return total_rewards
    

    def optimize(self, n_trials=100, save_params=True):
        print("Optimizing hyperparameters")
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=n_trials)
        
        best_params = study.best_params
        
        if save_params:
            with open(self.params_path, 'wb') as f:
                pickle.dump(best_params, f)
        print("Saved parameters to disk")
