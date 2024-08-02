import logging
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from envs.pygame.utils import make_env
from main.model_types import RLModelType
from DQN.agent import DQNAgent

class Trainer:
    def __init__(self, model_name, model_type, game_type):
        self.model_name = model_name
        self.model_type = model_type
        self.game_type = game_type

    def setup_env(self, options):
        if self.game_type == 'pygame':
            game_name, pygame_entrypoint = options["name"], options["entrypoint"]
            self.env_name = game_name
            self.env = make_env(game_name, pygame_entrypoint)

    def setup_agent(self, params: dict):
        if self.model_type == RLModelType.DQN:
            self.agent = DQNAgent(
                gamma=params['gamma'],
                epsilon=1.0,
                lr=params['lr'],
                input_dims=self.env.observation_space.shape,
                n_actions=self.env.action_space.n,
                mem_size=10000,
                eps_min=params['min_eps'],
                batch_size=params['batch_size'],
                replace=params['target_replace_frequency'],
                eps_dec=params['eps_decay'],
                chkpt_dir="models/",
                algo=self.model_name,
                env_name=self.env_name)
        elif self.model_type == RLModelType.DDQN:
            return "DDQN Agent"

    def trainer(self, episodes, load_checkpoint=False):

        if self.agent is None:
            raise RuntimeError("Agent has not been initialized.")
        

        logging.basicConfig(level=logging.INFO,  format='%(levelname)s - %(message)s')
        logger = logging.getLogger()

        n_steps = 0
        scores = []
        avg_scores = []

        if load_checkpoint:
            self.agent.load_models()

        for episode in range(episodes):
            done = False
            score = 0
            observation, _ = self.env.reset()
            points = 0


            while not done:
                action = self.agent.choose_action(observation)
                observation_, reward, done, _, info = self.env.step(action)

                score += reward 


                if not load_checkpoint:
                    self.agent.store_transition(observation, action,
                                                reward, observation_, int(done))
                    
                    self.agent.learn()

                points = info["score"]
                observation = observation_
                n_steps += 1

            
            scores.append(score)
            avg_score = sum(scores[-100:]) / min(100, len(scores))
            avg_scores.append(avg_score)
            logger.info(f'Episode {episode} - Score: {score:.2f}, Avg Score: {avg_score:.2f}, Apple points: {points:.2f}, Epsilon: {self.agent.epsilon:.3f}')

        return sum(scores) / len(scores), avg_scores



if __name__ == "__main__":
    trainer = Trainer(model_name="DQN", model_type=RLModelType.DQN, game_type="pygame")
    trainer.setup_env({"name": "snakegame-v0",
                       "entrypoint": "envs.pygame.snake_env:SnakeGameEnv"})
    agent_params = {
        "gamma": 0.99,
        "lr": 0.001,
        "batch_size": 64,
        "target_replace_frequency": 1000,
        "eps_decay": 0.995
    }
    trainer.setup_agent(agent_params)
    print("Agent setup complete")

    avg_score, avg_scores = trainer.trainer(episodes=500)
    print(f"Training complete. Average Score: {avg_score}")
