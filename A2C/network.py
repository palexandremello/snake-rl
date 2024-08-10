import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim

class PolicyNetwork(nn.Module):
    def __init__(self, lr, input_dims, n_actions):
        super(PolicyNetwork, self).__init__()
        
        self.fc1 = nn.Linear(*input_dims, 128)  # Primeira camada densa
        self.fc2 = nn.Linear(128, 128)            # Segunda camada densa
        self.fc3 = nn.Linear(128, n_actions)      # Saída com o número de ações

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))  # Aplicação da ReLU após a primeira camada
        x = F.relu(self.fc2(x))      # ReLU após a segunda camada
        x = self.fc3(x)              # Camada de saída (logits das ações)
        return x


import os

class PolicyGradientAgent:

    def __init__(self, lr, input_dims, gamma=0.99, n_actions=4, chkpt_dir='models', model_name='policy_network.pth'):
        self.gamma = gamma
        self.lr = lr
        self.reward_memory = []
        self.action_memory = []
        self.chkpt_dir = chkpt_dir
        self.model_name = model_name

        self.policy = PolicyNetwork(self.lr, input_dims, n_actions)

        if not os.path.exists(chkpt_dir):
            os.makedirs(chkpt_dir)

    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float32).to(self.policy.device)
        probabilities = F.softmax(self.policy.forward(state), dim=-1)
        action_probs = T.distributions.Categorical(probabilities)
        action = action_probs.sample()
        log_probs = action_probs.log_prob(action)
        self.action_memory.append(log_probs)

        return action.item()
    
    def store_rewards(self, reward):
        self.reward_memory.append(reward)

    def learn(self):
        self.policy.optimizer.zero_grad()

        G = np.zeros_like(self.reward_memory, dtype=np.float64)
        for t in range(len(self.reward_memory)):
            G_sum = 0
            discount = 1
            for k in range(t, len(self.reward_memory)):
                G_sum += self.reward_memory[k] * discount
                discount *= self.gamma
            G[t] = G_sum

        G = T.tensor(G, dtype=T.float32).to(self.policy.device)

        loss = 0
        for g, logprob in zip(G, self.action_memory):
            loss += -g * logprob
        
        loss.backward()
        self.policy.optimizer.step()

        self.action_memory = []
        self.reward_memory = []

    def save_model(self):
        print(f"Saving model to {self.chkpt_dir}/{self.model_name} ...")
        T.save(self.policy.state_dict(), os.path.join(self.chkpt_dir, self.model_name))
    
    def load_model(self):
        print(f"Loading model from {self.chkpt_dir}/{self.model_name} ...")
        self.policy.load_state_dict(T.load(os.path.join(self.chkpt_dir, self.model_name)))
        self.policy.eval()
