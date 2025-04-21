import numpy as np
import pickle
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.pos = 0
        self.alpha = alpha
        self.total = 0
    
    def push(self, state, action, reward, next_state, done, priority=1.00):
        self.memory.append((state, action, reward, next_state, done))
        self.priorities.append(priority)
        self.total += 1

    def sample(self, batch_size, beta=0.4):
        
        priorities = np.array(self.priorities) ** self.alpha
        priorities /= priorities.sum()

        indices = np.random.choice(len(self.memory), batch_size, p=priorities)
        batch = [self.memory[idx] for idx in indices]
        
        weights = (len(self.memory) * priorities[indices]) ** (-beta)
        weights /= weights.max()

        return batch, indices, weights
    
    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = abs(td_error) + 1e-5
    
    def save(self, file_path):
        """Save the replay buffer to a file."""
        with open(file_path, 'wb') as f:
            pickle.dump({
                'memory': list(self.memory),
                'priorities': list(self.priorities),
                'pos': self.pos,
                'total': self.total
            }, f)

    def load(self, file_path):
        """Load the replay buffer from a file."""
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            self.memory = deque(data['memory'], maxlen=self.capacity)
            self.priorities = deque(data['priorities'], maxlen=self.capacity)
            self.pos = data['pos']
            self.total = data['total']
    
    def __len__(self):
        return self.total


