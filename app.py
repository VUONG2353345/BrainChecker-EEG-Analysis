import sys
import os
import torch
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (QApplication, QMainWindow, QFileDialog, QVBoxLayout,
                             QHBoxLayout, QWidget, QGridLayout, QProgressBar,
                             QLabel, QPushButton, QStackedWidget)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy import signal  # để tính phổ tần số

# Thêm thư mục src vào path để import các module
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from model import TinyEEGNet
from preprocess import clean_eeg_signal

# ====================== Canvas vẽ biểu đồ EEG và phổ tần số ======================
class EEGCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(9, 7), facecolor='white')
        # 3 subplot: EEG (6 kênh), phổ tần số, xác suất
        self.ax_eeg = self.fig.add_subplot(311, facecolor='white')
        self.ax_spectrum = self.fig.add_subplot(312, facecolor='white')
        self.ax_prob = self.fig.add_subplot(313, facecolor='white')
        self.fig.tight_layout(pad=4.0)
        super().__init__(self.fig)

    def plot_data(self, raw, probabilities, file_name):
        """Vẽ 6 kênh EEG, phổ tần số trung bình, và xác suất động kinh."""
        self.ax_eeg.clear()
        self.ax_spectrum.clear()
        self.ax_prob.clear()

        # Lấy dữ liệu 4 giây đầu (1000 mẫu)
        data = raw.get_data()[:, :1000]
        sfreq = raw.info['sfreq']
        time = np.arange(1000) / sfreq

        n_chan = min(6, data.shape[0])  # hiển thị tối đa 6 kênh
        if n_chan == 0:
            return

        # --- Vẽ EEG (6 kênh) ---
        max_amp = np.max(np.abs(data[:n_chan])) if data.size > 0 else 1e-4
        offset = 4.0 * max_amp if max_amp > 0 else 0.001
        offsets = [i * offset for i in range(n_chan)]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

        for i in range(n_chan):
            y = data[i] + offsets[i]
            self.ax_eeg.plot(time, y, color=colors[i % len(colors)], lw=1.0)
            # Đặt tên kênh bên trái, ngoài khung
            self.ax_eeg.text(-2.2, offsets[i], raw.ch_names[i],
                             ha='right', va='center', fontsize=9, color=colors[i % len(colors)],
                             clip_on=False)

        self.ax_eeg.set_yticks([])
        self.ax_eeg.set_xlabel('Thời gian (giây)', fontsize=10)
        self.ax_eeg.set_title(f'Phân tích: {file_name}', loc='left', fontweight='bold', fontsize=12)
        self.ax_eeg.grid(True, linestyle='--', alpha=0.5)
        self.ax_eeg.set_xlim(-3.0, 4.0)
        self.ax_eeg.set_xticks(np.arange(0, 5, 1))
        self.ax_eeg.set_ylim([-offset, offsets[-1] + offset])

        # --- Vẽ phổ tần số trung bình (Power Spectral Density) ---
        freqs, psd = signal.welch(data, fs=sfreq, nperseg=256, axis=1)
        mean_psd = np.mean(psd, axis=0)  # trung bình theo các kênh
        self.ax_spectrum.semilogy(freqs, mean_psd, color='blue', lw=1.5)
        self.ax_spectrum.set_xlabel('Tần số (Hz)', fontsize=10)
        self.ax_spectrum.set_ylabel('Mật độ phổ công suất', fontsize=10)
        self.ax_spectrum.set_title('Phổ tần số trung bình (các kênh)', fontsize=11)
        self.ax_spectrum.grid(True, linestyle='--', alpha=0.5)
        self.ax_spectrum.set_xlim([0, 50])  # chỉ hiển thị đến 50 Hz (vùng quan trọng)

        # --- Vẽ xác suất động kinh ---
        n_windows = len(probabilities)
        if n_windows > 0:
            bars = self.ax_prob.bar(range(n_windows), probabilities,
                                    color=['red' if p > 0.5 else 'gray' for p in probabilities])
            self.ax_prob.set_xlabel('Cửa sổ 4 giây', fontsize=10)
            self.ax_prob.set_ylabel('Xác suất động kinh', fontsize=10)
            self.ax_prob.set_ylim([0, 1])
            self.ax_prob.axhline(y=0.5, color='k', linestyle='--', linewidth=0.8)
            self.ax_prob.set_title('Dự đoán theo từng đoạn', fontsize=11)

        self.draw()

# ====================== Worker xử lý dự đoán ======================
class PredictWorker(QThread):
    progress = pyqtSignal(int)
    done = pyqtSignal(dict, object, list, str)

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    def run(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TinyEEGNet().to(device)
        model.load_state_dict(torch.load("models/seizure_model.pth", map_location=device))
        model.eval()

        raw = clean_eeg_signal(self.file_path)
        data = raw.get_data()
        total_windows = data.shape[1] // 1000  # mỗi cửa sổ 4 giây
        predictions = []
        probabilities = []

        with torch.no_grad():
            for i in range(total_windows):
                segment = data[:, i*1000 : (i+1)*1000]
                inp = torch.from_numpy(segment).float().to(device)
                inp = inp.unsqueeze(0)          # (1, 23, 1000)
                out = model(inp)
                prob = torch.softmax(out, dim=1)[0][1].item()  # xác suất lớp Seizure
                pred = torch.argmax(out, dim=1).item()
                probabilities.append(prob)
                predictions.append(pred)
                self.progress.emit(int((i+1) / total_windows * 100))

        result = {
            'total': total_windows,
            'seizure_count': sum(predictions)
        }
        self.done.emit(result, raw, probabilities, os.path.basename(self.file_path))

# ====================== Cửa sổ chính ======================
class SeizureApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hệ thống Kiểm tra Động kinh - HCMUT")
        self.resize(1100, 800)
        self.setStyleSheet("background-color: white;")

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        # ---------- Trang 1: Màn hình chính ----------
        self.home_page = QWidget()
        home_layout = QVBoxLayout(self.home_page)
        home_layout.setContentsMargins(20, 20, 20, 20)

        # Logo não (brain.png)
        brain_logo = QLabel()
        pixmap = QPixmap("brain.png")
        if not pixmap.isNull():
            pixmap = pixmap.scaled(120, 120, Qt.AspectRatioMode.KeepAspectRatio,
                                   Qt.TransformationMode.SmoothTransformation)
            brain_logo.setPixmap(pixmap)
        else:
            brain_logo.setText("🧠")
            brain_logo.setStyleSheet("font-size: 80px; color: #2c3e50;")
        brain_logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        home_layout.addWidget(brain_logo)

        # Tiêu đề
        title = QLabel("KIỂM TRA ĐỘNG KINH")
        title.setStyleSheet("font-size: 42px; font-weight: bold; color: #2c3e50; margin: 20px;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        home_layout.addWidget(title)

        # Nút chọn file
        self.btn_open = QPushButton("📁 NẠP FILE DỮ LIỆU EDF")
        self.btn_open.setFixedSize(340, 90)
        self.btn_open.setStyleSheet("""
            QPushButton {
                border: 3px solid #3498db;
                border-radius: 20px;
                font-size: 20px;
                font-weight: bold;
                background-color: white;
                color: #2c3e50;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #ecf0f1;
            }
        """)
        self.btn_open.clicked.connect(self.load_file)
        home_layout.addWidget(self.btn_open, alignment=Qt.AlignmentFlag.AlignCenter)

        home_layout.addStretch()

        # Footer: "Developed by HCMUTer" bên trái, logo BKU bên phải
        footer = QHBoxLayout()
        footer.setContentsMargins(0, 10, 0, 10)

        dev_label = QLabel("Developed by HCMUTer")
        dev_label.setStyleSheet("font-size: 16px; color: #7f8c8d;")
        footer.addWidget(dev_label)

        footer.addStretch()

        bku_logo = QLabel()
        bku_pix = QPixmap("bku_logo.png")
        if not bku_pix.isNull():
            bku_pix = bku_pix.scaled(50, 50, Qt.AspectRatioMode.KeepAspectRatio,
                                     Qt.TransformationMode.SmoothTransformation)
            bku_logo.setPixmap(bku_pix)
        else:
            bku_logo.setText("BKU")
            bku_logo.setStyleSheet("font-size: 20px; color: #2c3e50;")
        bku_logo.setAlignment(Qt.AlignmentFlag.AlignRight)
        footer.addWidget(bku_logo)

        home_layout.addLayout(footer)

        self.stack.addWidget(self.home_page)

        # ---------- Trang 2: Kết quả phân tích ----------
        self.result_page = QWidget()
        result_layout = QGridLayout(self.result_page)

        # Canvas vẽ biểu đồ
        self.canvas = EEGCanvas(self)
        result_layout.addWidget(self.canvas, 0, 0, 1, 3)

        # Thanh tiến trình (ẩn ban đầu)
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #bdc3c7;
                border-radius: 8px;
                text-align: center;
                height: 25px;
                font-size: 14px;
                background-color: white;
            }
            QProgressBar::chunk {
                background-color: #27ae60;
                border-radius: 8px;
            }
        """)
        self.progress_bar.setVisible(False)
        result_layout.addWidget(self.progress_bar, 1, 0, 1, 3)

        # Label kết quả (hiển thị trạng thái)
        self.result_label = QLabel("")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setStyleSheet("font-size: 32px; font-weight: bold; margin: 20px;")
        result_layout.addWidget(self.result_label, 2, 0, 1, 3)

        # Các nút điều hướng (Quay lại, Thoát)
        nav_layout = QHBoxLayout()
        btn_back = QPushButton("⬅ Quay lại")
        btn_exit = QPushButton("✖ Thoát")
        btn_style = """
            QPushButton {
                border: 2px solid #7f8c8d;
                border-radius: 10px;
                padding: 12px 30px;
                font-size: 16px;
                font-weight: bold;
                background-color: white;
                color: black;
            }
            QPushButton:hover {
                background-color: #ecf0f1;
            }
        """
        btn_back.setStyleSheet(btn_style)
        btn_exit.setStyleSheet(btn_style + "QPushButton:hover { background-color: #e74c3c; color: white; }")
        btn_back.clicked.connect(lambda: self.stack.setCurrentIndex(0))
        btn_exit.clicked.connect(self.close)
        nav_layout.addStretch()
        nav_layout.addWidget(btn_back)
        nav_layout.addWidget(btn_exit)
        result_layout.addLayout(nav_layout, 3, 0, 1, 3)

        result_layout.setRowStretch(0, 10)
        result_layout.setRowStretch(2, 1)
        self.stack.addWidget(self.result_page)

        # Timer để ẩn progress bar sau khi hiển thị kết quả
        self.timer = QTimer()
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.hide_progress)

    def load_file(self):
        """Mở hộp thoại chọn file EDF và bắt đầu phân tích."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Chọn file EDF", "data/", "*.edf")
        if file_path:
            self.stack.setCurrentIndex(1)
            self.progress_bar.setValue(0)
            self.progress_bar.setVisible(True)
            self.result_label.setText("🔍 Đang phân tích...")
            self.result_label.setStyleSheet("font-size: 32px; font-weight: bold; color: #3498db;")

            self.worker = PredictWorker(file_path)
            self.worker.progress.connect(self.progress_bar.setValue)
            self.worker.done.connect(self.show_result)
            self.worker.start()

    def show_result(self, result, raw, probabilities, file_name):
        """Hiển thị kết quả lên canvas và label."""
        self.canvas.plot_data(raw, probabilities, file_name)

        if result['seizure_count'] > 0:
            status = "⚠️ PHÁT HIỆN ĐỘNG KINH"
            color = '#e74c3c'
        else:
            status = "✅ BÌNH THƯỜNG"
            color = '#27ae60'

        self.result_label.setText(f"{status}\nSố đoạn bất thường: {result['seizure_count']} / {result['total']}")
        self.result_label.setStyleSheet(f"font-size: 36px; font-weight: bold; color: {color};")

        self.timer.start(2000)  # 2 giây sau ẩn progress bar

    def hide_progress(self):
        self.progress_bar.setVisible(False)

# ====================== Chạy ứng dụng ======================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SeizureApp()
    window.show()
    sys.exit(app.exec())