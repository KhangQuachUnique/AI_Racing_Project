import pygame
import numpy as np
import math

WIDTH = 1920
HEIGHT = 1080
CAR_SIZE_X = 50
CAR_SIZE_Y = 50
BORDER_COLOR = (255, 255, 255, 255)

class Car:
    def __init__(self):
        self.sprite = pygame.image.load('assets/car.png').convert_alpha()
        self.sprite = pygame.transform.scale(self.sprite, (CAR_SIZE_X, CAR_SIZE_Y))
        self.rotated_sprite = self.sprite
        self.position = [830, 920]
        self.angle = 0
        self.speed = 0
        self.speed_set = False
        self.center = [self.position[0] + CAR_SIZE_X / 2, self.position[1] + CAR_SIZE_Y / 2]
        self.radars_size = 5
        self.radius = 80
        self.good_angle = self.radius * 2 / (self.radars_size - 1)
        self.radars = []
        self.alive = True
        self.distance = 0
        self.time = 0

    def draw(self, screen):
        screen.blit(self.rotated_sprite, self.position)
        self.draw_radar(screen)

    def draw_radar(self, screen):
        for i, radar in enumerate(self.radars):
            position = radar[0]
            distance = radar[1]
            angle = np.linspace(-self.radius, self.radius, self.radars_size)[i]  # Tính góc tương ứng cho radar

            # Vẽ đường radar
            pygame.draw.line(screen, (0, 255, 0), self.center, position, 1)
            pygame.draw.circle(screen, (0, 255, 0), position, 5)

            # Hiển thị số thứ tự và góc của radar
            font = pygame.font.Font(None, 24)
            text_surface = font.render(f"{i} ({int(angle)}°)", True, (0, 0, 255))
            screen.blit(text_surface, (position[0] + 5, position[1] - 15))

    def check_collision(self, game_map):
        self.alive = True
        for point in self.corners:
            if game_map.get_at((int(point[0]), int(point[1]))) == BORDER_COLOR:
                self.alive = False
                break

    def check_radar(self, degree, game_map):
        length = 0
        x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length)
        y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length)
        while not game_map.get_at((x, y)) == BORDER_COLOR and length < 300:
            length += 1
            x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length)
            y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length)
        dist = int(math.sqrt((x - self.center[0])**2 + (y - self.center[1])**2))
        self.radars.append([(x, y), dist])

    def update(self, game_map):
        if not self.speed_set:
            self.speed = 15
            self.speed_set = True
        self.rotated_sprite = self.rotate_center(self.sprite, self.angle)
        self.position[0] += math.cos(math.radians(360 - self.angle)) * self.speed
        self.position[0] = max(self.position[0], 20)
        self.position[0] = min(self.position[0], WIDTH - 120)
        self.position[1] += math.sin(math.radians(360 - self.angle)) * self.speed
        self.position[1] = max(self.position[1], 20)
        self.position[1] = min(self.position[1], HEIGHT - 120)
        self.distance += self.speed
        self.time += 1
        self.center = [int(self.position[0]) + CAR_SIZE_X / 2, int(self.position[1]) + CAR_SIZE_Y / 2]
        length = 0.5 * CAR_SIZE_X
        left_top = [self.center[0] + math.cos(math.radians(360 - (self.angle + 30))) * length,
                    self.center[1] + math.sin(math.radians(360 - (self.angle + 30))) * length]
        right_top = [self.center[0] + math.cos(math.radians(360 - (self.angle + 150))) * length,
                     self.center[1] + math.sin(math.radians(360 - (self.angle + 150))) * length]
        left_bottom = [self.center[0] + math.cos(math.radians(360 - (self.angle + 210))) * length,
                       self.center[1] + math.sin(math.radians(360 - (self.angle + 210))) * length]
        right_bottom = [self.center[0] + math.cos(math.radians(360 - (self.angle + 330))) * length,
                        self.center[1] + math.sin(math.radians(360 - (self.angle + 330))) * length]
        self.corners = [left_top, right_top, left_bottom, right_bottom]
        self.check_collision(game_map)
        self.radars.clear()
        angles = np.linspace(-self.radius, self.radius, num=self.radars_size)  # Tăng số lượng radar từ 7 lên 11
        for d in angles:
            self.check_radar(d, game_map)

    def get_data(self):
        return_values = np.zeros(self.radars_size + 1, dtype=int)
        for i, radar in enumerate(self.radars[:self.radars_size]):
            return_values[i] = int(radar[1] / 30)
            return_values[5] = int(self.speed / 20)
            # return_values[6] = int(self.angle / 360)
        return return_values

    def is_alive(self):
        return self.alive

    def get_reward(self, old_angle):
        total_reward = 0
        if not self.alive:
            return -50  # Phạt nặng nếu xe chết

        # Phần thưởng dựa trên tốc độ
        speed_reward = 4 * (2 / (1 + np.exp(-0.13 * (self.speed - 7))) - 1)  # Tăng tốc độ tối đa lên 20

        # Phần thưởng dựa trên khoảng cách di chuyển
        distance_reward = self.distance / max(self.time, 1) / 5  # Tránh chia cho 0

        # Phần thưởng dựa trên góc tốt
        max_length_couple_radar = 0
        max_couple_radar = [0, 0]
        for i in range(len(self.radars)-1):
            length_couple_radar = self.radars[i][1] + self.radars[i+1][1]
            if length_couple_radar > max_length_couple_radar:   
                max_length_couple_radar = length_couple_radar
                max_couple_radar[0] = i #0 under
                max_couple_radar[1] = i + 1 #1 upper
        angles = np.linspace(-self.radius, self.radius, num=self.radars_size)  
        good_angle = 0
        if self.radars[max_couple_radar[0]][1] > self.radars[max_couple_radar[1]][1]:
            ratio = self.radars[max_couple_radar[0]][1] / self.radars[max_couple_radar[1]][1]
            good_angle = (angles[max_couple_radar[0]] - self.good_angle) * ratio
        else:
            ratio = self.radars[max_couple_radar[1]][1] / self.radars[max_couple_radar[0]][1]
            good_angle = (angles[max_couple_radar[1]] + self.good_angle) * ratio

        good_angle_reward = abs(good_angle - old_angle) - abs(good_angle - self.angle)
        if good_angle_reward < 0:
            good_angle_reward = -6
        else:
            good_angle_reward = 4

        # Cải thiện logic tính distance_to_border
        distance_to_border = 0
        for i, radar in enumerate(self.radars):
            distance_to_border -= max(0, (25 - radar[1]) * 2)

            
        # Tổng hợp phần thưởng với trọng số
        total_reward += (distance_reward + good_angle_reward + speed_reward + distance_to_border)
        print(f"Distance: {distance_reward}, Angle: {good_angle_reward}, Speed: {speed_reward}, Border: {distance_to_border}, Total: {total_reward}")
        return total_reward

    def rotate_center(self, image, angle):
        rectangle = image.get_rect()
        rotated_image = pygame.transform.rotate(image, angle)
        rotated_rectangle = rectangle.copy()
        rotated_rectangle.center = rotated_image.get_rect().center
        rotated_image = rotated_image.subsurface(rotated_rectangle).copy()
        return rotated_image
