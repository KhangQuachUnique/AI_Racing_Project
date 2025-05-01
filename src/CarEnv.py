import math
import pygame
import numpy as np
import threading  # Used for running Pygame in a separate thread
from Car import Car  # Assuming you have a Car class defined in Car.py

WIDTH = 1920
HEIGHT = 1080

# CarEnv class
class CarEnv:
    def __init__(self, map_choice="Random"):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()

        # Font để hiển thị thông tin
        self.font = pygame.font.Font(None, 36)  # Font mặc định, kích thước 36

        # Load the selected map or a random map
        if map_choice == "Random":
            map_index = np.random.randint(1, 5)
        else:
            map_index = int(map_choice.split(" ")[-1])  # Extract map number from "Map X"
        self.game_map = pygame.image.load(f'assets/map{map_index}.png').convert()
        self.game_map = pygame.transform.scale(self.game_map, (WIDTH, HEIGHT))

        self.car = Car()
        self.done = False

        # Biến để lưu thông tin hiển thị
        self.info = {
            "Episode": 0,
            "Reward": 0.0,
            "Loss": 0.0,
            "Epsilon": 0.0,
            "Steps": 0,
            "Angle": 0.0,
            "Speed": 0.0,
            "Distance": 0.0,
            "Time": 0.0,
        }

    def reset(self):
        self.car = Car()
        self.done = False
        state = self.car.get_data()
        
        # Thêm nhiễu Gaussian vào trạng thái
        noise = np.random.normal(0, 0.1, len(state))
        noisy_state = np.clip(state + noise, 0, 1)  # Giới hạn giá trị trong khoảng [0, 1]
        return noisy_state

    def step(self, action):
        old_angle = self.car.angle
        if action == 0:
            self.car.angle += 10
        elif action == 1:
            self.car.angle -= 10
        elif action == 2:
            if self.car.speed > 3:
                self.car.speed -= 1
        elif action == 3:
            if self.car.speed < 50:
                self.car.speed += 1
        self.car.update(self.game_map)
        reward = self.car.get_reward(old_angle)
        if not self.car.is_alive():
            self.done = True
        observation = self.car.get_data()
        return observation, reward, self.done

    def render(self):
        self.screen.blit(self.game_map, (0, 0))  # Vẽ bản đồ
        if self.car.is_alive():
            self.car.draw(self.screen)  # Vẽ xe

        # Hiển thị thông tin trên màn hình
        self.display_info()  # Đảm bảo hàm này được gọi

        pygame.display.flip()  # Cập nhật màn hình
        self.clock.tick(60)

    def display_info(self):
        """Hiển thị thông tin như reward, loss, epsilon, v.v. trên màn hình."""
        y_offset = 10  # Khoảng cách dòng đầu tiên
        for key, value in self.info.items():
            text_surface = self.font.render(f"{key}: {value}", True, (0, 255, 0))  # Màu trắng
            self.screen.blit(text_surface, (10, y_offset))  # Vẽ thông tin lên màn hình
            y_offset += 30  # Tăng khoảng cách giữa các dòng

    def update_info(self, episode, reward, loss, epsilon, steps):
        """Cập nhật thông tin để hiển thị."""
        self.info["Episode"] = episode
        self.info["Reward"] = f"{reward:.2f}"
        self.info["Loss"] = f"{loss:.4f}"
        self.info["Epsilon"] = f"{epsilon:.4f}"
        self.info["Steps"] = steps
        self.info["Angle"] = f"{self.car.angle:.2f}"
        self.info["Speed"] = f"{self.car.speed:.2f}"
        self.info["Distance"] = f"{self.car.distance:.2f}"
        self.info["Time"] = f"{self.car.time:.2f}"


    def close(self):
        pygame.quit()

# Run pygame in a separate thread
def run_pygame():
    env = CarEnv()
    observation = env.reset()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        import random
        action = random.choice([0, 1, 2, 3])
        observation, reward, done = env.step(action)
        env.render()
        if done:
            observation = env.reset()
        # Đảm bảo xử lý sự kiện Pygame trong vòng lặp chính
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.done = True

        # Giới hạn FPS để tránh chạy quá nhanh
        env.clock.tick(60)
    env.close()

if __name__ == "__main__":
    pygame_thread = threading.Thread(target=run_pygame)
    pygame_thread.start()

