import pygame
import numpy as np
import math

# Kích thước môi trường
WIDTH = 1920
HEIGHT = 1080
# Kích thước xe
CAR_SIZE_X = 50
CAR_SIZE_Y = 50
# Màu sắc của đường biên
BORDER_COLOR = (255, 255, 255, 255)

class Car:
    def __init__(self):
        self.sprite = pygame.image.load('assets/car.png').convert_alpha() # Load ảnh xe
        self.sprite = pygame.transform.scale(self.sprite, (CAR_SIZE_X, CAR_SIZE_Y)) # Thay đổi kích thước ảnh xe
        self.rotated_sprite = self.sprite # Ảnh xe đã xoay
        self.position = [830, 920] # Vị trí khởi đầu của xe
        self.angle = 0 # Góc của xe
        self.speed = 0 # Tốc độ của xe
        self.speed_set = False
        self.center = [self.position[0] + CAR_SIZE_X / 2, self.position[1] + CAR_SIZE_Y / 2] # Tọa độ tâm của xe
        self.radars_size = 5 # Số lượng radar
        self.radius = 80 # Góc quét của radar
        self.good_angle = self.radius * 2 / (self.radars_size - 1)
        self.radars = [] # Danh sách chứa tọa độ và khoảng cách của radar
        self.alive = True
        self.distance = 0 # Khoảng cách đã đi được
        self.time = 0 # Thời gian đã trôi qua

    def draw(self, screen):
        screen.blit(self.rotated_sprite, self.position)
        self.draw_radar(screen)

    def draw_radar(self, screen):
        for i, radar in enumerate(self.radars):
            position = radar[0] 

            # Vẽ đường radar
            pygame.draw.line(screen, (0, 255, 0), self.center, position, 1)
            pygame.draw.circle(screen, (0, 255, 0), position, 5)

    def check_collision(self, game_map): #Kiểm tra va chạm -> Boolean
        self.alive = True
        for point in self.corners:
            if game_map.get_at((int(point[0]), int(point[1]))) == BORDER_COLOR:
                self.alive = False
                break

    def check_radar(self, degree, game_map): # Kiểm tra radar -> List.
        length = 0
        x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length)
        y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length)
        while not game_map.get_at((x, y)) == BORDER_COLOR and length < 300:
            length += 1
            x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length)
            y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length)
        dist = int(math.sqrt((x - self.center[0])**2 + (y - self.center[1])**2))
        self.radars.append([(x, y), dist]) # Thêm tọa độ và khoảng cách vào danh sách radar

    def update(self, game_map): # Cập nhật vị trí, góc, tốc độ, radar và va chạm
        if not self.speed_set:
            self.speed = 15
            self.speed_set = True
        self.rotated_sprite = self.rotate_center(self.sprite, self.angle)
        self.position[0] += math.cos(math.radians(360 - self.angle)) * self.speed
        self.position[1] += math.sin(math.radians(360 - self.angle)) * self.speed
        self.position[0] = max(0, min(self.position[0], WIDTH - CAR_SIZE_X))
        self.position[1] = max(0, min(self.position[1], HEIGHT - CAR_SIZE_Y))
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

    def get_data(self): # Trả về state của Agent để đưa vào mạng nơ-ron
        return_values = np.zeros(self.radars_size, dtype=float)
        for i, radar in enumerate(self.radars[:self.radars_size]):
            return_values[i] = float(radar[1]/30) 
        # return_values[5] = int(self.speed/5)
        print(return_values)
        return return_values

    def is_alive(self):
        return self.alive

    def get_reward(self, old_angle):
        total_reward = 0
        if not self.alive:
            return -1 

        # Phần thưởng cho tốc độ
        speed_reward = 5 * (2 / (1 + np.exp(-0.13 * (self.speed - 7))) - 1) # Max speed reward is 8 and min is -8
        
        # Phần thưởng cho tốc độ trung bình
        average_speed_by_step_reward = self.distance / max(self.time, 1) / 10  # Tránh chia cho 0

        # Phần thưởng cho góc tốt 
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

        good_turn = abs(good_angle - old_angle) - abs(good_angle - self.angle)
        if good_turn < 0:
            good_angle_reward = -7 
        else:
            good_angle_reward = 7

        # Phần thưởng cho vị trí xe ở giữa đường 
        dist_to_left_wall = (self.radars[0][1] + self.radars[1][1])/2 
        dist_to_right_wall = (self.radars[3][1] + self.radars[4][1])/2 
        center_reward = max(-7, 7 - abs(dist_to_left_wall - dist_to_right_wall)/10)

        # Tính tổng phần thưởng
        total_reward += ((average_speed_by_step_reward + good_angle_reward + speed_reward + center_reward)/21) # Chuẩn hóa tổng phần thưởng về khoảng [-1, 1]
        print(f"Distance: {average_speed_by_step_reward}, Angle: {good_angle_reward}, Speed: {speed_reward}, Center: {center_reward}, Total: {total_reward}")
        return total_reward

    def rotate_center(self, image, angle): # Xoay hình ảnh
        rectangle = image.get_rect()
        rotated_image = pygame.transform.rotate(image, angle)
        rotated_rectangle = rectangle.copy()
        rotated_rectangle.center = rotated_image.get_rect().center
        rotated_image = rotated_image.subsurface(rotated_rectangle).copy()
        return rotated_image
