import numpy as np

class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions, prioritize=False, alpha=0.6):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.uint8)
        
        self.prioritize = prioritize
        self.alpha = alpha
        if prioritize:
            self.priority_memory = np.zeros(self.mem_size, dtype=np.float32)
        
    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = done
        
        if self.prioritize:
            self.priority_memory[index] = max(self.priority_memory.max(), 1.0)
        
        self.mem_cntr += 1
    
    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        
        if self.prioritize:
            priorities = self.priority_memory[:max_mem]
            probabilities = priorities ** self.alpha
            probabilities /= probabilities.sum()
            batch = np.random.choice(max_mem, batch_size, p=probabilities)
        else:
            batch = np.random.choice(max_mem, batch_size, replace=False)
        
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]
        
        return states, actions, rewards, states_, dones
    
    def update_priorities(self, indices, errors):
        if self.prioritize:
            for i, error in zip(indices, errors):
                self.priority_memory[i] = error

