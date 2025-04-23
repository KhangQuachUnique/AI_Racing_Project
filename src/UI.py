import sys
import torch
import pickle
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QFileDialog, QFormLayout,
    QLabel, QMessageBox, QLineEdit, QGridLayout, QComboBox, QTextEdit, QGroupBox, QInputDialog, QHBoxLayout, QStyleFactory, QStyle
)
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, Qt
from PyQt5.QtGui import QFont, QImage, QPixmap, QPainter, QColor
from PyQt5.QtWidgets import QGraphicsDropShadowEffect  # Hiệu ứng đổ bóng
from Agent import Agent  # Giả sử bạn đã định nghĩa lớp Agent trong file Agent.py
from CarEnv import CarEnv
from collections import deque
import pygame

from params import input_defaut

from PyQt5.QtWidgets import QStyledItemDelegate

class ComboBoxItemDelegate(QStyledItemDelegate):
    def sizeHint(self, option, index):
        size = super().sizeHint(option, index)
        size.setHeight(30)  # chỉnh chiều cao từng item
        return size



class PygameWidget(QWidget):
    def __init__(self, parent=None, width=800, height=600):
        super().__init__(parent)
        self.width = width
        self.height = height
        self.surface = pygame.Surface((self.width, self.height))  # Tạo một pygame.Surface
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)  # Gọi update() để kích hoạt paintEvent
        self.timer.start(16)  # ~60 FPS

    def update_pygame(self):
        # Vẽ nội dung lên pygame.Surface
        self.surface.fill((0, 0, 0))  # Fill screen with black
        pygame.draw.circle(self.surface, (255, 0, 0), (self.width // 2, self.height // 2), 50)  # Draw a red circle

    def paintEvent(self, event):
        # Cập nhật nội dung pygame trước khi vẽ
        self.update_pygame()

        # Chuyển đổi pygame.Surface thành QImage
        data = pygame.image.tostring(self.surface, "RGB")
        image = QImage(data, self.width, self.height, QImage.Format_RGB888)

        # Hiển thị QImage trên PyQt
        painter = QPainter(self)
        pixmap = QPixmap.fromImage(image)
        painter.drawPixmap(self.rect(), pixmap)
        painter.end()

    def closeEvent(self, event):
        pygame.quit()
        super().closeEvent(event)


class TrainingThread(QThread):
    update_status = pyqtSignal(str)  # Signal to update status in the UI

    def __init__(self, agent_params, num_episodes, map_choice, agent: Agent = None):
        super().__init__()
        self.agent_params = agent_params  # Dictionary of agent parameters
        self.num_episodes = num_episodes
        self.map_choice = map_choice
        self.agent = agent if agent is not None else None  # Agent instance sẽ được khởi tạo trong run()

    def run(self):
        try:
            self.agent.get_training_parameters(**self.agent_params)
            print(self.agent_params)
            env = CarEnv(map_choice=self.map_choice)
            
            self.agent.train(env, num_episodes=self.num_episodes, update_status=self.update_status.emit)
            if not self.agent.stop_training:
                self.update_status.emit("Training completed.")
            else:
                self.update_status.emit("Training stopped by user.")
        except Exception as e:
            self.update_status.emit(f"Training failed: {e}")

    def stop(self):
        if self.agent:
            self.agent.stop()  # Đặt cờ dừng huấn luyện trong Agent


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Agent Training UI")
        screen_geometry = QApplication.desktop().screenGeometry()  # Lấy kích thước màn hình hiện tại
        self.setGeometry(screen_geometry)  # Đặt kích thước cửa sổ khớp với màn hình
        self.showMaximized()  # Mở cửa sổ toàn màn hình với thanh tiêu đề
        self.setWindowState(self.windowState() | Qt.WindowMaximized)  # Đảm bảo cửa sổ được mở toàn màn hình với thanh tiêu đề
        self.setStyleSheet("""
            QMainWindow {
                background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1, stop:0 #1E1E1E, stop:1 #252526);
            }
            QLabel {
                color: #D4D4D4;
                font-size: 16px;
            }
            QPushButton {
                background-color: #007ACC;  /* Màu xanh giống VS Code */
                color: #FFFFFF;
                border: none;
                border-radius: 6px;
                padding: 10px;
                font-weight: bold;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #005A9E;  /* Màu xanh đậm hơn khi hover */
            }
            QLineEdit {
                background-color: #2D2D30;
                color: #D4D4D4;
                border: 1px solid #3C3C3C;
                border-radius: 6px;
                padding: 8px;
                font-size: 16px;
            }
            QComboBox {
                background-color: #2D2D30;  /* Màu nền tối giống VS Code */
                color: #D4D4D4;  /* Màu chữ sáng */
                border: 1px solid #3C3C3C;
                border-radius: 6px;
                padding: 8px;
                font-size: 16px;
            }
            QComboBox QAbstractItemView {
                background-color: #252526;  /* Màu nền danh sách */
                color: #D4D4D4;  /* Màu chữ danh sách */
                border: 1px solid #3C3C3C;
                selection-background-color: #007ACC;  /* Màu nền khi chọn */
                selection-color: #FFFFFF;  /* Màu chữ khi chọn */
                border-radius: 6px;  /* Bo góc danh sách */
                padding: 4px;  /* Khoảng cách giữa các mục */
            }
            QComboBox QAbstractItemView::item {
                min-height: 30px;  /* Chiều cao tối thiểu cho mỗi mục */
            }
            QComboBox QAbstractItemView::item:hover {
                background-color: #3A3D41;  /* Hover đẹp hơn */
            }
            QTextEdit {
                background-color: #1E1E1E;
                color: #D4D4D4;
                border: 1px solid #3C3C3C;
                border-radius: 6px;
                font-size: 16px;
            }
            QGroupBox {
                border: 1px solid #3C3C3C;
                border-radius: 8px;
                margin-top: 10px;
                color: #D4D4D4;
                font-size: 16px;
                padding: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 5px;
                font-weight: bold;
            }
        """)

        # Main layout
        main_layout = QHBoxLayout()

        left_layout = QVBoxLayout()
        # Status label
        self.status_label = QLabel("Status: Ready")
        self.status_label.setFont(QFont("Arial", 20))  # Tăng cỡ chữ của status label lên rõ ràng hơn
        self.status_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.add_shadow(self.status_label)  # Thêm hiệu ứng đổ bóng
        self.status_label.setStyleSheet("max-height: 40px; font-weight: bold; font-size: 20px")  # Set fixed height for status label
        left_layout.addWidget(self.status_label)

        # Pygame widget
        self.pygame_widget = PygameWidget(self, width=800, height=600)
        pygame_layout = QHBoxLayout()
        pygame_layout.addWidget(self.pygame_widget)
        main_layout.addLayout(pygame_layout)

        # Input fields group
        input_agent_group = QGroupBox("Agent Parameters")
        input_agent_group.setFont(QFont("Arial", 14))
        input_agent_layout = QFormLayout()
        
        self.num_episodes_input = QLineEdit()
        self.num_episodes_input.setPlaceholderText(f"Number of Episodes (default: {input_defaut['num_episodes']})")
        self.num_episodes_input.setFont(QFont("Arial", 12))
        input_agent_layout.addRow(QLabel("Episodes:", font=QFont("Arial", 12)), self.num_episodes_input)

        self.map_selector = QComboBox()
        self.map_selector.addItems(["Random", "Map 1", "Map 2", "Map 3", "Map 4", "Map 5", "Map 6"])
        self.map_selector.setFont(QFont("Arial", 12))
        self.map_selector.setItemDelegate(ComboBoxItemDelegate())
        self.map_selector.view().setUniformItemSizes(True)
        input_agent_layout.addRow(QLabel("Select Map:", font=QFont("Arial", 12)), self.map_selector)

        self.batch_size_input = QLineEdit()
        self.batch_size_input.setPlaceholderText(f"Batch Size (default: {input_defaut['batch_size']})")
        self.batch_size_input.setFont(QFont("Arial", 12))
        input_agent_layout.addRow(QLabel("Batch Size:", font=QFont("Arial", 12)), self.batch_size_input)

        self.gamma_input = QLineEdit()
        self.gamma_input.setPlaceholderText(f"Gamma (default: {input_defaut['gamma']})")
        self.gamma_input.setFont(QFont("Arial", 12))
        input_agent_layout.addRow(QLabel("Gamma:", font=QFont("Arial", 12)), self.gamma_input)

        self.lr_input = QLineEdit()
        self.lr_input.setPlaceholderText(f"Learning Rate (default: {input_defaut['lr']})")
        self.lr_input.setFont(QFont("Arial", 12))
        input_agent_layout.addRow(QLabel("Learning Rate:", font=QFont("Arial", 12)), self.lr_input)

        self.memory_capacity_input = QLineEdit()
        self.memory_capacity_input.setPlaceholderText(f"Memory Capacity (default: {input_defaut['memory_capacity']})")
        self.memory_capacity_input.setFont(QFont("Arial", 12))
        input_agent_layout.addRow(QLabel("Memory Capacity:", font=QFont("Arial", 12)), self.memory_capacity_input)

        self.eps_start_input = QLineEdit()
        self.eps_start_input.setPlaceholderText(f"Epsilon Start (default: {input_defaut['eps_start']})")
        self.eps_start_input.setFont(QFont("Arial", 12))
        input_agent_layout.addRow(QLabel("Epsilon Start:", font=QFont("Arial", 12)), self.eps_start_input)

        self.eps_end_input = QLineEdit()
        self.eps_end_input.setPlaceholderText(f"Epsilon End (default: {input_defaut['eps_end']})")
        self.eps_end_input.setFont(QFont("Arial", 12))
        input_agent_layout.addRow(QLabel("Epsilon End:", font=QFont("Arial", 12)), self.eps_end_input)

        self.eps_decay_input = QLineEdit()
        self.eps_decay_input.setPlaceholderText(f"Epsilon Decay (default: {input_defaut['eps_decay']})")
        self.eps_decay_input.setFont(QFont("Arial", 12))
        input_agent_layout.addRow(QLabel("Epsilon Decay:", font=QFont("Arial", 12)), self.eps_decay_input)

        self.target_update_input = QLineEdit()
        self.target_update_input.setPlaceholderText(f"Target Update (default: {input_defaut['target_update']})")
        self.target_update_input.setFont(QFont("Arial", 12))
        input_agent_layout.addRow(QLabel("Target Update:", font=QFont("Arial", 12)), self.target_update_input)


        # Name Model and Buffer
        name_model_and_buffer = QFormLayout()

        self.agent_label = QLabel("None")
        self.agent_label.setFont(QFont("Arial", 12))
        name_model_and_buffer.addRow(QLabel("Current Agent:", font=QFont("Arial", 12)), self.agent_label)

        self.buffer_label = QLabel("None")
        self.buffer_label.setFont(QFont("Arial", 12))
        name_model_and_buffer.addRow(QLabel("Current Buffer:", font=QFont("Arial", 12)), self.buffer_label)


        # Model information
        model_info_layout = QFormLayout()

        self.model_input_dim = QLabel("None")
        self.model_input_dim.setFont(QFont("Arial", 12))
        model_info_layout.addRow(QLabel("Input Dimension:", font=QFont("Arial", 12)), self.model_input_dim)

        self.model_output_dim = QLabel("None")
        self.model_output_dim.setFont(QFont("Arial", 12))
        model_info_layout.addRow(QLabel("Output Dimension:", font=QFont("Arial", 12)), self.model_output_dim)

        self.model_batch_size = QLabel("None")
        self.model_batch_size.setFont(QFont("Arial", 12))
        model_info_layout.addRow(QLabel("Batch Size:", font=QFont("Arial", 12)), self.model_batch_size)

        self.model_gamma = QLabel("None")
        self.model_gamma.setFont(QFont("Arial", 12))
        model_info_layout.addRow(QLabel("Gamma:", font=QFont("Arial", 12)), self.model_gamma)

        self.model_lr = QLabel("None")
        self.model_lr.setFont(QFont("Arial", 12))
        model_info_layout.addRow(QLabel("Learning Rate:", font=QFont("Arial", 12)), self.model_lr)

        self.model_memory_capacity = QLabel("None")
        self.model_memory_capacity.setFont(QFont("Arial", 12))
        model_info_layout.addRow(QLabel("Memory Capacity:", font=QFont("Arial", 12)), self.model_memory_capacity)

        self.model_eps_start = QLabel("None")
        self.model_eps_start.setFont(QFont("Arial", 12))
        model_info_layout.addRow(QLabel("Epsilon Start:", font=QFont("Arial", 12)), self.model_eps_start)

        self.model_eps_end = QLabel("None")
        self.model_eps_end.setFont(QFont("Arial", 12))
        model_info_layout.addRow(QLabel("Epsilon End:", font=QFont("Arial", 12)), self.model_eps_end)
        
        self.model_eps_decay = QLabel("None")
        self.model_eps_decay.setFont(QFont("Arial", 12))
        model_info_layout.addRow(QLabel("Epsilon Decay:", font=QFont("Arial", 12)), self.model_eps_decay)

        model_buffer_name = QGroupBox("Model Buffer Name")
        model_buffer_name.setFont(QFont("Arial", 14))
        model_buffer_name.setLayout(name_model_and_buffer)
        
        model_info = QGroupBox("Model Information")
        model_info.setFont(QFont("Arial", 14))
        model_info.setLayout(model_info_layout)

        input_agent_group.setLayout(input_agent_layout)
        input_grid_layout = QGridLayout()
        input_grid_layout.setColumnStretch(0, 5)
        input_grid_layout.setColumnStretch(1, 3)
        input_grid_layout.addWidget(input_agent_group, 0, 0, 2, 1)
        input_grid_layout.addWidget(model_buffer_name, 0, 1) 
        input_grid_layout.addWidget(model_info, 1, 1)
        self.add_shadow(input_agent_group) 
        left_layout.addLayout(input_grid_layout)  

        # Buttons group
        button_group = QGroupBox("Actions")
        button_group.setFont(QFont("Arial", 14))
        button_layout = QGridLayout()


        button_font = QFont("Arial", 14)

        load_model_btn = QPushButton("Load Model")
        load_model_btn.setFont(button_font)
        load_model_btn.clicked.connect(self.load_model)
        button_layout.addWidget(load_model_btn, 0, 0)

        save_model_btn = QPushButton("Save Model")
        save_model_btn.setFont(button_font)
        save_model_btn.clicked.connect(self.save_model)
        button_layout.addWidget(save_model_btn, 0, 1)

        load_memory_btn = QPushButton("Load Replay Memory")
        load_memory_btn.setFont(button_font)
        load_memory_btn.clicked.connect(self.load_replay_memory)
        button_layout.addWidget(load_memory_btn, 1, 0)

        save_memory_btn = QPushButton("Save Replay Memory")
        save_memory_btn.setFont(button_font)
        save_memory_btn.clicked.connect(self.save_replay_memory)
        button_layout.addWidget(save_memory_btn, 1, 1)

        train_agent_btn = QPushButton("Train Agent")
        train_agent_btn.setFont(button_font)
        train_agent_btn.clicked.connect(self.train_agent)
        button_layout.addWidget(train_agent_btn, 2, 0)

        stop_training_btn = QPushButton("Stop Training")
        stop_training_btn.setFont(button_font)
        stop_training_btn.clicked.connect(self.stop_training)
        button_layout.addWidget(stop_training_btn, 2, 1)

        create_agent_btn = QPushButton("Create New Agent")
        create_agent_btn.setFont(button_font)
        create_agent_btn.clicked.connect(self.create_new_agent)
        button_layout.addWidget(create_agent_btn, 3, 0)

        test_agent_btn = QPushButton("Test Agent")
        test_agent_btn.setFont(button_font)
        test_agent_btn.clicked.connect(self.test_agent)
        button_layout.addWidget(test_agent_btn, 3, 1)

        exit_btn = QPushButton("Exit")
        exit_btn.setFont(button_font)
        exit_btn.clicked.connect(self.close)
        button_layout.addWidget(exit_btn, 4, 0)

        button_group.setLayout(button_layout)
        self.add_shadow(button_group)  # Thêm hiệu ứng đổ bóng
        button_group.setFixedHeight(280)  # Set fixed height for button group
        left_layout.addWidget(button_group)

        # Training log
        self.training_log = QTextEdit()
        self.training_log.setReadOnly(True)
        self.training_log.setFont(QFont("Courier", 12))
        self.training_log.setStyleSheet("font-size: 20px; background-color: #000000; color:#FFFFFF; border: 1px solid #ccc;")
        right_layout = QVBoxLayout()
        training_log_label = QLabel("Training Log:")
        training_log_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #D4D4D4;")
        right_layout.addWidget(training_log_label)
        
        right_layout.addWidget(self.training_log)
        # right_layout.setStretch(1, 1)  # Stretch the training log to fill available space
        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)
        # self.training_log.setMinimumHeight(200)  # Set minimum height for training log


        # Set the main layout
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        self.agent = None  # Agent will be initialized when needed
        self.training_thread = None  # Training thread will be initialized when needed

        # Biến để lưu tên model hiện tại
        self.current_model_name = None
        self.current_replay_memory_name = None

    def add_shadow(self, widget):
        """Thêm hiệu ứng đổ bóng cho widget."""
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15)
        shadow.setColor(QColor(0, 0, 0, 160))
        shadow.setOffset(0, 4)
        widget.setGraphicsEffect(shadow)

    def closeEvent(self, event):
        # Đảm bảo pygame được tắt khi cửa sổ PyQt đóng
        pygame.quit()
        super().closeEvent(event)

    def get_input(self):
        # Lấy các tham số từ giao diện
        input_dim = input_defaut['input_dim']  # Đặt giá trị mặc định cho input_dim
        output_dim = input_defaut['output_dim']  # Đặt giá trị mặc định cho output_dim
        batch_size = int(self.batch_size_input.text()) if self.batch_size_input.text() else input_defaut['batch_size']
        gamma = float(self.gamma_input.text()) if self.gamma_input.text() else input_defaut['gamma']
        lr = float(self.lr_input.text()) if self.lr_input.text() else input_defaut['lr']
        memory_capacity = int(self.memory_capacity_input.text()) if self.memory_capacity_input.text() else input_defaut['memory_capacity']
        eps_start = float(self.eps_start_input.text()) if self.eps_start_input.text() else input_defaut['eps_start']
        eps_end = float(self.eps_end_input.text()) if self.eps_end_input.text() else input_defaut['eps_end']
        eps_decay = int(self.eps_decay_input.text()) if self.eps_decay_input.text() else input_defaut['eps_decay']
        target_update = int(self.target_update_input.text()) if self.target_update_input.text() else input_defaut['target_update']
        return {
            "batch_size": batch_size,
            "gamma": gamma,
            "lr": lr,
            "eps_start": eps_start,
            "eps_end": eps_end,
            "eps_decay": eps_decay,
            "target_update": target_update
        }  # Trả về các tham số và tên bản đồ được chọn

    def get_model_info(self):
        if self.agent:
            model_info = self.agent.get_params_info()
            self.model_input_dim.setText(str(model_info['input_dim']))
            self.model_output_dim.setText(str(model_info['output_dim']))
            self.model_batch_size.setText(str(model_info['batch_size']))
            self.model_gamma.setText(str(model_info['gamma']))
            self.model_lr.setText(str(model_info['lr']))
            self.model_memory_capacity.setText(str(model_info['memory_capacity']))
            self.model_eps_start.setText(str(model_info['eps_start']))
            self.model_eps_end.setText(str(model_info['eps_end']))
            self.model_eps_decay.setText(str(model_info['eps_decay']))

    # def get_sample_ratio(self):
    #     collision = float(self.collision_input.text()) if self.collision_input.text() else 0.25
    #     optimal = float(self.optimal_input.text()) if self.optimal_input.text() else 0.25
    #     normal = float(self.normal_input.text()) if self.normal_input.text() else 0.25
    #     bad = float(self.bad_input.text()) if self.bad_input.text() else 0.25
    #     return {
    #         "collision": collision,
    #         "optimal": optimal,
    #         "normal": normal,
    #         "bad": bad
    #     }

    def load_model(self):
        self.agent = Agent(
            # input_dim=7,  # Default input dimension
            # output_dim=4,  # Default output dimension
            batch_size=input_defaut['batch_size'],
            gamma=input_defaut['gamma'],
            lr=input_defaut['lr'],
            memory_capacity=input_defaut['memory_capacity'],
            eps_start=input_defaut['eps_start'],
            eps_end=input_defaut['eps_end'],
            eps_decay=input_defaut['eps_decay'],
            target_update=input_defaut['target_update'],
        )
        model_dir = "./model"
        os.makedirs(model_dir, exist_ok=True)  # Tạo thư mục nếu chưa tồn tại
        model_path, _ = QFileDialog.getOpenFileName(self, "Load Model", model_dir, "Model Files (*.pth)")
        if model_path:
            try:
                checkpoint = torch.load(model_path)
                self.agent.policy_net.load_state_dict(checkpoint['model_state_dict'])
                self.agent.target_net.load_state_dict(checkpoint['model_state_dict'])
                self.agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.current_model_name = os.path.basename(model_path)  # Lưu tên model hiện tại
                self.status_label.setText(f"Status: Model '{self.current_model_name}' loaded successfully.")
                self.agent_label.setText(f"Agent: {self.current_model_name}")
                self.get_model_info()  # Cập nhật thông tin model
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load model: {e}")

    def save_model(self):
        if self.current_model_name:
            # Nếu đã có model hiện tại, lưu lại vào model đó
            model_path = os.path.join("./model", self.current_model_name)
            try:
                torch.save({
                    'model_state_dict': self.agent.policy_net.state_dict(),
                    'optimizer_state_dict': self.agent.optimizer.state_dict()
                }, model_path)
                self.status_label.setText(f"Status: Model '{self.current_model_name}' saved successfully.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save model: {e}")
        else:
            # Nếu chưa có model hiện tại, yêu cầu nhập tên model mới
            model_name, ok = QInputDialog.getText(self, "Save Model", "Enter model name:")
            if ok and model_name:
                model_path = os.path.join("./model", f"{model_name}.pth")
                try:
                    torch.save({
                        'model_state_dict': self.agent.policy_net.state_dict(),
                        'optimizer_state_dict': self.agent.optimizer.state_dict()
                    }, model_path)
                    self.current_model_name = f"{model_name}.pth"  # Cập nhật tên model hiện tại
                    self.status_label.setText(f"Status: Model '{self.current_model_name}' saved successfully.")
                    self.get_model_info()  # Cập nhật thông tin model
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to save model: {e}")

    def load_replay_memory(self):
        replay_dir = "./replay_memory"
        os.makedirs(replay_dir, exist_ok=True)  # Tạo thư mục nếu chưa tồn tại
        memory_path, _ = QFileDialog.getOpenFileName(self, "Load Replay Memory", replay_dir, "Memory Files (*.pkl)")
        if memory_path:
            try:
                self.agent.memory.load(memory_path)
                self.current_replay_memory_name = os.path.basename(memory_path)  # Lưu tên replay memory hiện tại
                self.status_label.setText(f"Status: Replay memory '{self.current_replay_memory_name}' loaded successfully.")
                self.buffer_label.setText(f"Buffer: {self.current_replay_memory_name}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load replay memory: {e}")

    def save_replay_memory(self):
        if self.current_replay_memory_name:
            # Nếu đã có replay memory hiện tại, lưu lại vào file đó
            memory_path = os.path.join("./replay_memory", self.current_replay_memory_name)
            try:
                self.agent.memory.save(memory_path)
                self.status_label.setText(f"Status: Replay memory '{self.current_replay_memory_name}' saved successfully.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save replay memory: {e}")
        else:
            # Nếu chưa có replay memory hiện tại, yêu cầu nhập tên mới
            memory_name, ok = QInputDialog.getText(self, "Save Replay Memory", "Enter replay memory name:")
            if ok and memory_name:
                memory_path = os.path.join("./replay_memory", f"{memory_name}.pkl")
                try:
                    self.agent.memory.save(memory_path)
                    self.current_replay_memory_name = f"{memory_name}.pkl"  # Cập nhật tên replay memory hiện tại
                    self.status_label.setText(f"Status: Replay memory '{self.current_replay_memory_name}' saved successfully.")
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to save replay memory: {e}")

    def train_agent(self):
        try:
            if self.agent is None:
                QMessageBox.critical(self, "Error", "No agent available for training. Please create or load an agent.")
                return

            num_episodes = int(self.num_episodes_input.text()) if self.num_episodes_input.text() else input_defaut['num_episodes']
            map_choice = self.map_selector.currentText()

            self.training_log.clear()
            self.status_label.setText("Status: Training agent...")

            self.training_thread = TrainingThread(
                self.get_input(),  # Lấy các tham số từ giao diện
                num_episodes,
                map_choice,
                self.agent
            )
            
            self.get_model_info()  # Cập nhật thông tin model
            self.training_thread.update_status.connect(self.update_training_log)
            self.training_thread.start()
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid input for training parameters.")

    def test_agent(self):
        try:
            if self.agent is None:
                QMessageBox.critical(self, "Error", "No agent available for testing. Please create or load an agent.")
                return

            # Lấy bản đồ được chọn
            map_choice = self.map_selector.currentText()

            # Khởi tạo môi trường
            env = CarEnv(map_choice=map_choice)

            # Số tập để kiểm tra
            num_episodes, ok = QInputDialog.getInt(self, "Test Agent", "Enter number of test episodes:", 10, 1, 100)
            if not ok:
                return

            # Chạy kiểm tra và hiển thị kết quả
            avg_reward = self.agent.test(env, num_episodes=num_episodes)
            QMessageBox.information(self, "Test Completed", f"Testing completed for {num_episodes} episodes.\nAverage Reward: {avg_reward:.2f}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Testing failed: {e}")

    def update_training_log(self, message):
        self.training_log.append(message)
        self.status_label.setText(message)
        # print(message)  # In thông tin ra console

    def create_new_agent(self):
        try:
            # Kiểm tra nếu có model đang hoạt động
            if self.current_model_name:
                reply = QMessageBox.question(
                    self,
                    "Model in Use",
                    f"A model '{self.current_model_name}' is currently loaded. Do you want to save it before creating a new agent?",
                    QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel
                )
                if reply == QMessageBox.Yes:
                    self.save_model()  # Lưu model hiện tại
                elif reply == QMessageBox.Cancel:
                    return  # Hủy tạo Agent mới nếu người dùng chọn Cancel

            # Thu thập các tham số từ giao diện
            agent_params = self.get_input()

            # Khởi tạo Agent mới
            self.agent = Agent(**agent_params)
            self.current_model_name = None  # Reset tên model hiện tại
            self.current_replay_memory_name = None  # Reset tên replay memory hiện tại
            self.get_model_info()  # Cập nhật thông tin model
            self.status_label.setText("Status: New agent created successfully.")
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid input for agent parameters.")

    def stop_training(self):
        if self.training_thread:
            self.training_thread.stop()  # Gọi phương thức stop của TrainingThread
            self.status_label.setText("Status: Training stopped by user.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
