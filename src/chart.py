import matplotlib as plt
import numpy as np
import matplotlib.pyplot as plt
import csv

def plot_from_csv(file_path, prob1, prob2):
    """
    Hàm vẽ biểu đồ loss theo từng step từ file CSV.
    :param file_path: Đường dẫn tới file CSV chứa dữ liệu training.
    """
    import csv

    episodes = []
    losses = []

    # Đọc dữ liệu từ file CSV
    with open(file_path, mode="r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            try:
                step = float(row[prob1].strip())  # Loại bỏ khoảng trắng hoặc ký tự không cần thiết
                loss = float(row[prob2].strip())  # Loại bỏ khoảng trắng hoặc ký tự không cần thiết
                episodes.append(step)
                losses.append(loss)
            except ValueError:
                print(f"Invalid data at Step: {row.get('Step', 'Unknown')}, Loss: {row.get('Loss', 'Unknown')}")
                continue  # Bỏ qua dòng không hợp lệ

    # Vẽ biểu đồ dạng điểm (scatter plot)
    plt.figure(figsize=(10, 6))
    plt.scatter(episodes, losses, label="Loss", color="blue", alpha=0.6)
    plt.title("Loss per Step (Scatter Plot)")
    plt.xlabel(prob1)
    plt.ylabel(prob2)
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()

plot_from_csv("./training_data/training_data_DQN Agent_1.csv", "Episode", "Average Reward")
# plot_loss_epsilon_from_csv("./training_data/training_data_DQN Agent.csv")

