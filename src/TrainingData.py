import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd

def plot_models_logs(log_files, label1, label2):
    """
    Plots the training logs of two models from CSV files.

    Parameters:
    - log_files: List of tuples containing the file paths for the two models.
    - label1: Label for the first model.
    - label2: Label for the second model.
    """
    plt.figure(figsize=(10, 5))

    for log_file, label in zip(log_files, [label1, label2]):
        with open(log_file, 'r') as f:
            reader = csv.reader(f)
            data = list(reader)

        # Assuming the first row is headers and the first column is steps
        steps = [int(row[0]) for row in data[1:]]
        rewards = [float(row[1]) for row in data[1:]]

        plt.scatter(steps, rewards, label=label)

    plt.xlabel('Steps')
    plt.ylabel('Rewards')
    plt.title('Training Logs Comparison')
    plt.legend()
    plt.grid()
    plt.show()

def plot_scatter_from_csv(file_path, x_column, y_column, title="Scatter Plot", xlabel="X-Axis", ylabel="Y-Axis"):
    """
    Vẽ biểu đồ scatter từ file CSV.

    Args:
        file_path (str): Đường dẫn đến file CSV.
        x_column (str): Tên cột dùng làm trục X.
        y_column (str): Tên cột dùng làm trục Y.
        title (str): Tiêu đề biểu đồ.
        xlabel (str): Nhãn trục X.
        ylabel (str): Nhãn trục Y.
    """
    # Đọc dữ liệu từ file CSV
    data = pd.read_csv(file_path)

    # Kiểm tra xem các cột có tồn tại không
    if x_column not in data.columns or y_column not in data.columns:
        print(f"Columns '{x_column}' or '{y_column}' not found in the file.")
        return

    # Vẽ biểu đồ scatter
    plt.figure(figsize=(10, 6))
    plt.scatter(data[x_column], data[y_column], alpha=0.7, label="Agent 1")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_scatter_step_log(file_path, x_column_index, y_column_index, title="Scatter Plot", xlabel="X-Axis", ylabel="Y-Axis"):
    """
    Vẽ biểu đồ scatter từ file step_log.csv và in từng phần tử theo chỉ số.

    Args:
        file_path (str): Đường dẫn đến file step_log.csv.
        x_column_index (int): Chỉ số cột dùng làm trục X.
        y_column_index (int): Chỉ số cột dùng làm trục Y.
        title (str): Tiêu đề biểu đồ.
        xlabel (str): Nhãn trục X.
        ylabel (str): Nhãn trục Y.
    """
    try:
        with open(file_path, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)
            headers = next(reader)  # Đọc tiêu đề cột
            print(f"Headers: {headers}")

            x_data = []
            y_data = []

            for index, row in enumerate(reader, start=1):
                x_data.append(index)
                y_data.append(float(row[y_column_index]))

        # Vẽ biểu đồ scatter
        plt.figure(figsize=(10, 6))
        plt.scatter(x_data, y_data, alpha=0.7, label="Data Points")
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)
        plt.show()

    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"Error reading or plotting file: {e}")

# Ví dụ sử dụng
plot_scatter_from_csv(
    file_path="./training_log/model_2_episode.csv",
    x_column="Episode",
    y_column="Average Loss",
    title="Episode vs Average Loss",
    xlabel="Episode",
    ylabel="Average Loss"
)

plot_scatter_from_csv(
    file_path="./training_log/model_2_episode.csv",
    x_column="Episode",
    y_column="Total Reward",
    title="Episode vs Total Reward",
    xlabel="Episode",
    ylabel="Total Reward"
)

plot_scatter_step_log(
    file_path="./training_log/model_2_step.csv",
    x_column_index=0,  # Cột Episode
    y_column_index=1,  # Cột Reward
    title="Step vs Loss",
    xlabel="Step",
    ylabel="Loss"
)

# Ví dụ sử dụng
# plot_scatter_step_log(
#     file_path="c:/Users/kadfw/OneDrive/Dokumen/AI_Racing_Project/training_log/step_log.csv",
#     x_column_index=0,  # Cột Episode
#     y_column_index=2,  # Cột Reward
#     title="Episode vs Reward",
#     xlabel="Episode",
#     ylabel="Reward"
# )