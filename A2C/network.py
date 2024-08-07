import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim


class PolicyNetwork(nn.Module):
    def __init__(self, lr, input_dims, n_actions):
        super(PolicyNetwork, self).__init__()
        
        self.conv1 = nn.Conv2d(input_dims[0], 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)

        fc_input_dims = self.calculate_conv_output_dims(input_dims)


        self.fc1 = nn.Linear(fc_input_dims, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def calculate_conv_output_dims(self, input_dims):
        state = T.zeros(1, *input_dims)
        dims = self.conv1(state)
        dims = self.conv2(dims)
        dims = self.conv3(dims)
        return int(np.prod(dims.size()))
    

    def forward(self, state):
        conv1 = F.relu(self.conv1(state))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))
        conv_state = conv3.view(conv3.size()[0], -1)

        x = F.relu(self.fc1(conv_state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class PolicyGradientAgent:

    def __init__(self, lr, input_dims, gamma=0.99, n_actions=4):
        self.gamma = gamma
        self.lr = lr
        self.reward_memory = []
        self.action_memory = []

        self.policy = PolicyNetwork(self.lr, input_dims, n_actions)

    

    def choose_action(self, observation):
        state = T.tensor([observation]).to(self.policy.device)
        probabilities = F.softmax(self.policy.forward(state))
        action_probs = T.distributions.Categorical(probabilities)
        action = action_probs.sample()
        log_probs = action_probs.log_prob(action)
        self.action_memory.append(log_probs)

        return action.item()
    

    def store_rewards(self, reward):
        self.reward_memory.append(reward)


    
    def learn(self):
        self.policy.optimizer.zero_grad()

        G = np.zeros_like(self.reward_memory)
        for t in range(len(self.reward_memory)):
            G_sum = 0
            discount = 1
            for k in range(t, len(self.reward_memory)):
                G_sum += self.reward_memory[k] * discount
                discount *= self.gamma
            
            G[t] = G_sum
        G = T.tensor(G, dtype=T.float).to(self.policy.device)

        loss = 0
        for g, logprob in zip(G, self.action_memory):
            loss += -g * logprob
        
        loss.backward()
        self.policy.optimizer.step()

        self.action_memory = []
        self.reward_memory = []