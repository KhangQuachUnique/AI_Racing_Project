import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
import pygame 
import os
from ReplayBuffer import ReplayBuffer  

# ---------------------------
# Định nghĩa mạng DQN
# ---------------------------
class DQN(nn.Module):
    def __init__(self, INPUT_DIM, OUTPUT_DIM):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, OUTPUT_DIM)
        )
    
    def forward(self, x):
        return self.net(x)

# ---------------------------
# Class Agent để quản lý logic của DQN
# ---------------------------
class Agent:
    def __init__(self, input_dim=6, output_dim=4, batch_size=512, gamma=0.99, lr=1e-3, memory_capacity=100000, 
                 eps_start=0.85, eps_end=0.05, eps_decay=3000, target_update=50, device=None):
        self.name = "DQN Agent"
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.memory_capacity = memory_capacity
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.target_update = target_update
        self.device = "cuda"
        
        self.policy_net = DQN(input_dim, output_dim).to(self.device)
        self.target_net = DQN(input_dim, output_dim).to(self.device)
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.memory = ReplayBuffer(memory_capacity)
        self.steps_done = 0
        self.stop_training = False  # Biến cờ để dừng huấn luyện

    def get_training_parameters(self, batch_size, gamma, lr, eps_start, eps_end, eps_decay, target_update):
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.target_update = target_update
        # print(f"Training parameters updated: gamma={gamma}, lr={lr}, eps_start={eps_start}, eps_end={eps_end}, eps_decay={eps_decay}, target_update={target_update}")

    def stop(self):
        self.stop_training = True  # Đặt cờ để dừng huấn luyện

    def select_action(self, state):
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if random.random() < eps_threshold:
            return random.randrange(self.output_dim)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.max(1)[1].item() 

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return None
        # print(len(self.memory))
        transitions, indices, weights = self.memory.sample(self.batch_size)
        batch = list(zip(*transitions))
        
        state_batch = torch.FloatTensor(batch[0]).to(self.device)
        action_batch = torch.LongTensor(batch[1]).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(batch[2]).to(self.device)
        next_state_batch = torch.FloatTensor(batch[3]).to(self.device)
        done_batch = torch.FloatTensor(batch[4]).to(self.device)

        weights = torch.FloatTensor(weights).unsqueeze(1).to(self.device)

        q_values = self.policy_net(state_batch).gather(1, action_batch)
        next_q_values = self.target_net(next_state_batch).max(1)[0].detach()
        expected_q_values = reward_batch + self.gamma * next_q_values * (1 - done_batch)
        expected_q_values = expected_q_values.unsqueeze(1)

        td_errors = (q_values - expected_q_values).abs().cpu().detach().numpy().squeeze()
        self.memory.update_priorities(indices, td_errors)

        loss = (weights * nn.SmoothL1Loss(reduction='none')(q_values, expected_q_values)).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def train(self, env, num_episodes=10000, update_status=None, render=True):
        self.steps_done = 0  # Reset steps_done for each training session
        self.stop_training = False  # Reset cờ dừng huấn luyện
        try:
            for episode in range(num_episodes):
                if self.stop_training:  # Kiểm tra cờ dừng huấn luyện
                    if update_status:
                        update_status("Training stopped by user.")
                    break

                state = env.reset()  # Reset environment and get initial observation
                total_reward = 0.0
                episode_loss = 0.0
                steps = 0
                
                while True:
                    # Xử lý sự kiện pygame để tránh bị đơ khi alt-tab
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            env.close()
                            if update_status:
                                update_status("Training interrupted by user.")
                            return

                    if self.stop_training:  # Kiểm tra cờ dừng huấn luyện trong vòng lặp
                        if update_status:
                            update_status("Training stopped by user.")
                        return

                    # Select action using epsilon-greedy strategy
                    action = self.select_action(state)
                    
                    # Perform action in the environment
                    next_state, reward, done = env.step(action)
                    total_reward += reward
                    steps += 1
                    
                    self.memory.push(state, action, reward, next_state, done, priority=1.0)
                    state = next_state
                    
                    loss = self.optimize_model()
                    if loss is not None:
                        episode_loss += loss.item()
                    
                    if self.steps_done % self.target_update == 0:
                        self.target_net.load_state_dict(self.policy_net.state_dict())
                    
                    epsilon = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)
                    env.update_info(episode + 1, total_reward, episode_loss / max(steps, 1), epsilon, steps)

                    if render:
                        env.render()
                    
                    if done:
                        break
                
                # # Log dữ liệu vào TensorBoard sau mỗi episode
                # self.logger.log_scalar("Total Reward", total_reward, episode)
                # self.logger.log_scalar("Average Loss", episode_loss / max(steps, 1), episode)
                # self.logger.log_scalar("Epsilon", epsilon, episode)

                # Update status in the UI if provided
                if update_status:
                    update_status(f"Episode: {episode + 1}/{num_episodes} - Reward: {total_reward:.2f}")
        except Exception as e:
            if update_status:
                update_status(f"Training interrupted: {e}")
        finally:
            # self.logger.close()  # Đóng TensorBoard logger
            env.close()

    def test(self, env, num_episodes=10):
        """Kiểm tra hiệu suất của Agent và hiển thị môi trường Pygame."""
        total_rewards = []
        for episode in range(num_episodes):
            state = env.reset()
            total_reward = 0

            while True:
                # Chọn hành động dựa trên chính sách (không khám phá)
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    action = self.policy_net(state_tensor).max(1)[1].item()

                # Thực hiện hành động trong môi trường
                next_state, reward, done = env.step(action)
                total_reward += reward
                state = next_state

                # Hiển thị môi trường
                env.render()
                pygame.event.pump()  # Đảm bảo xử lý sự kiện của Pygame

                if done:
                    break

            total_rewards.append(total_reward)

        avg_reward = sum(total_rewards) / num_episodes
        env.close()  # Đóng môi trường sau khi kiểm tra
        return avg_reward

    def get_params_info(self):
        """Trả về thông tin về các tham số huấn luyện."""
        return {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "batch_size": self.batch_size,
            "memory_capacity": self.memory_capacity,
            "gamma": self.gamma,
            "lr": self.lr,
            "eps_start": self.eps_start,
            "eps_end": self.eps_end,
            "eps_decay": self.eps_decay,
            "target_update": self.target_update
        }