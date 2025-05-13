import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit   # ← Thêm import này

def logistic(x, L, k, x0):
    """
    Hàm logistic cơ bản: L / (1 + exp(-k*(x - x0)))
    - L : độ cao cực đại
    - k : tốc độ tăng
    - x0: điểm giữa (midpoint)
    """
    return L / (1 + np.exp(-k * (x - x0)))

def plot_scatter_from_csv(file_path, x_column, y_column, degree=2,
                             title="Scatter Plot", xlabel="X-Axis", ylabel="Y-Axis"):
    """
    Vẽ scatter và hai đường xu hướng:
      - Polynomial fit (bậc degree)
      - Logistic fit (sigmoid)

    Args:
        file_path (str): đường dẫn file CSV
        x_column (str): tên cột cho trục X
        y_column (str): tên cột cho trục Y
        degree (int): bậc đa thức (1: thẳng, 2: parabolic, 3: cubic,…)
        title (str), xlabel (str), ylabel (str): nhãn hiển thị
    """
    # Đọc dữ liệu
    data = pd.read_csv(file_path)
    if x_column not in data.columns or y_column not in data.columns:
        print(f"Columns '{x_column}' or '{y_column}' not found.")
        return

    x = data[x_column].values
    y = data[y_column].values

    # 1. Fit đa thức
    x_smooth = np.linspace(x.min(), x.max(), 500)

    # 2. Fit logistic
    p0 = [max(y), 1, np.median(x)]
    try:
        popt, _ = curve_fit(logistic, x, y, p0=p0, maxfev=10000)
        y_logistic = logistic(x_smooth, *popt)
        do_logistic = True
    except RuntimeError:
        print("Logistic fit failed; skipping logistic curve.")
        do_logistic = False

    # Vẽ đồ thị
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, alpha=0.7, label="Data Points")

    # Logistic trendline
    if do_logistic:
        L, k, x0 = popt
        plt.plot(x_smooth, y_logistic, color='red', linewidth=2,
                 label=f"Logistic (L={L:.2f}, k={k:.2f}, x0={x0:.2f})")

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
            headers = next(reader)
            print(f"Headers: {headers}")

            x_data = []
            y_data = []
            for index, row in enumerate(reader, start=1):
                x_data.append(index)
                y_data.append(float(row[y_column_index]))

        x_arr = np.array(x_data)
        y_arr = np.array(y_data)

        # Fit logistic
        p0 = [max(y_arr), 1, np.median(x_arr)]
        try:
            popt, _ = curve_fit(logistic, x_arr, y_arr, p0=p0, maxfev=10000)
            x_smooth = np.linspace(x_arr.min(), x_arr.max(), 500)
            y_logistic = logistic(x_smooth, *popt)
            do_logistic = True
        except RuntimeError:
            print("Logistic fit failed; skipping logistic curve.")
            do_logistic = False

        # Vẽ biểu đồ
        plt.figure(figsize=(10, 6))
        plt.scatter(x_data, y_data, alpha=0.7, label="Data Points")

        # Vẽ logistic nếu có kết quả
        if do_logistic:
            L, k, x0 = popt
            plt.plot(x_smooth, y_logistic, color='red', linewidth=2,
                     label=f"Logistic fit (L={L:.2f}, k={k:.2f}, x0={x0:.2f})")

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

def plot_scatter_from_csv_multi(file_path, x_column, y_column, degree=2,
                             title="Scatter Plot", xlabel="X-Axis", ylabel="Y-Axis"):
    """
    Vẽ scatter và hai đường xu hướng:
      - Polynomial fit (bậc degree)
      - Logistic fit (sigmoid)

    Args:
        file_path (list): danh sách đường dẫn file CSV
        x_column (str): tên cột cho trục X
        y_column (str): tên cột cho trục Y
        degree (int): bậc đa thức (1: thẳng, 2: parabolic, 3: cubic,…)
        title (str), xlabel (str), ylabel (str): nhãn hiển thị
    """
    plt.figure(figsize=(10, 6))

    for file in file_path:
        # Đọc dữ liệu
        data = pd.read_csv(file)
        if x_column not in data.columns or y_column not in data.columns:
            print(f"Columns '{x_column}' or '{y_column}' not found in {file}.")
            continue

        x = data[x_column].values
        y = data[y_column].values

        # Vẽ scatter
        plt.scatter(x, y, alpha=0.7, label=f"Data from {file}")

        # Fit logistic
        x_smooth = np.linspace(x.min(), x.max(), 500)
        p0 = [max(y), 1, np.median(x)]
        try:
            popt, _ = curve_fit(logistic, x, y, p0=p0, maxfev=10000)
            y_logistic = logistic(x_smooth, *popt)
            L, k, x0 = popt
            plt.plot(x_smooth, y_logistic, linewidth=2, label=f"Logistic fit ({file})")
        except RuntimeError:
            print(f"Logistic fit failed for {file}; skipping logistic curve.")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()

def draw_charts(model_file):
    plot_scatter_from_csv(
        file_path=f"./training_log/model_{model_file}_episode.csv",
        x_column="Episode",
        y_column="Average Loss",
        title="Episode vs Average Loss",
        xlabel="Episode",
        ylabel="Average Loss"
    )

    plot_scatter_from_csv(
        file_path=f"./training_log/model_{model_file}_episode.csv",
        x_column="Episode",
        y_column="Total Reward",
        title="Episode vs Total Reward",
        xlabel="Episode",
        ylabel="Total Reward"
    )

    plot_scatter_from_csv(
        file_path=f"./training_log/model_{model_file}_episode.csv",
        x_column="Episode",
        y_column="Average Reward",
        title="Episode vs Average Reward",
        xlabel="Episode",
        ylabel="Average Reward"
    )

    plot_scatter_from_csv(
        file_path=f"./training_log/model_{model_file}_episode.csv",
        x_column="Average Loss",
        y_column="Average Reward",
        title="Average Loss vs Average Reward",
        xlabel="Average Loss",
        ylabel="Average Reward"
    )

    plot_scatter_step_log(
        file_path=f"./training_log/model_{model_file}_step.csv",
        x_column_index=0,  # Cột Episode
        y_column_index=1,  # Cột Reward
        title="Step vs Loss",
        xlabel="Step",
        ylabel="Loss"
    )

    plot_scatter_step_log(
        file_path=f"./training_log/model_{model_file}_step.csv",
        x_column_index=0,  # Cột Episode
        y_column_index=2,  # Cột Q-Value 
        title="Step vs Q-Value",
        xlabel="Step",
        ylabel="Q-Value"
    )

# plot_scatter_from_csv_multi(
#     file_path=["./training_log/model_1_episode.csv", "./training_log/model_2_episode.csv", "./training_log/model_3_episode.csv", "./training_log/model_4_episode.csv", "./training_log/model_5_episode.csv", "./training_log/model_6_episode.csv"],
#     x_column="Episode",
#     y_column="Average Loss",
#     title="Episode vs Average Loss",
#     xlabel="Episode",
#     ylabel="Average Loss"
# )   
# Ví dụ sử dụng
draw_charts("7")