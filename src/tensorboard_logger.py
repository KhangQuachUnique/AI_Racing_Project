import os
from torch.utils.tensorboard import SummaryWriter

class TensorBoardLogger:
    def __init__(self, log_dir="logs"):
        """
        Khởi tạo logger cho TensorBoard.
        :param log_dir: Thư mục lưu trữ log TensorBoard.
        """
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)

    def log_scalar(self, tag, value, step):
        """
        Ghi một giá trị scalar vào TensorBoard.
        :param tag: Tên của giá trị (ví dụ: "loss", "accuracy").
        :param value: Giá trị cần ghi.
        :param step: Bước hiện tại (epoch hoặc iteration).
        """
        self.writer.add_scalar(tag, value, step)

    def log_histogram(self, tag, values, step):
        """
        Ghi một histogram vào TensorBoard.
        :param tag: Tên của histogram.
        :param values: Dữ liệu cần ghi (ví dụ: trọng số của mạng).
        :param step: Bước hiện tại (epoch hoặc iteration).
        """
        self.writer.add_histogram(tag, values, step)

    def log_figure(self, tag, figure, step):
        """
        Ghi một biểu đồ matplotlib vào TensorBoard.
        :param tag: Tên của biểu đồ.
        :param figure: Biểu đồ matplotlib.
        :param step: Bước hiện tại (epoch hoặc iteration).
        """
        self.writer.add_figure(tag, figure, step)

    def close(self):
        """
        Đóng logger.
        """
        self.writer.close()

# Ví dụ sử dụng
if __name__ == "__main__":
    logger = TensorBoardLogger(log_dir="logs/train")

    # Ghi giá trị scalar
    for step in range(100):
        logger.log_scalar("loss", 0.01 * step, step)

    # Đóng logger
    logger.close()