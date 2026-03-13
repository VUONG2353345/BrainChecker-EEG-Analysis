import torch
import numpy as np
from torch.utils.data import Dataset, WeightedRandomSampler
import random

class SeizureDataset(Dataset):
    """
    Dataset nhẹ, không lưu toàn bộ segments trong RAM.
    Hỗ trợ data augmentation (thêm nhiễu, dịch chuyển thời gian).
    """
    def __init__(self, raw_data_list, seizure_times_list, window_size=4, normal_ratio=5, augment=False):
        """
        raw_data_list: list các đối tượng MNE Raw đã lọc
        seizure_times_list: list các dict (hoặc list) chứa thời gian động kinh tương ứng
        augment: có áp dụng data augmentation không (chỉ cho training)
        """
        self.raw_list = raw_data_list
        self.seizure_times_list = seizure_times_list
        self.sfreq = int(raw_data_list[0].info['sfreq']) if raw_data_list else 250
        self.window_samples = window_size * self.sfreq
        self.augment = augment

        self.indices = []  # (raw_idx, start_sample, label)

        for raw_idx, raw in enumerate(raw_data_list):
            seizure_times = seizure_times_list[raw_idx]
            signals = raw.get_data()
            total_samples = signals.shape[1]

            for start in range(0, total_samples - self.window_samples, self.window_samples):
                end = start + self.window_samples
                current_time = start / self.sfreq
                is_seizure = any(s <= current_time <= e for s, e in seizure_times)
                label = 1 if is_seizure else 0
                self.indices.append((raw_idx, start, label))

        # --- Cân bằng dữ liệu (chọn ngẫu nhiên) ---
        labels = [idx[2] for idx in self.indices]
        idx_1 = [i for i, lbl in enumerate(labels) if lbl == 1]
        idx_0 = [i for i, lbl in enumerate(labels) if lbl == 0]

        if idx_1:
            n_normal = min(len(idx_0), len(idx_1) * normal_ratio)
            random.seed(42)
            idx_0_selected = random.sample(idx_0, n_normal)
            final_indices = sorted(idx_1 + idx_0_selected)
            self.indices = [self.indices[i] for i in final_indices]
            print(f"📊 Dataset tạo xong: {len(idx_1)} đoạn Seizure, {n_normal} đoạn Normal.")
        else:
            print("⚠️ Không có đoạn Seizure nào trong dữ liệu.")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        raw_idx, start, label = self.indices[idx]
        raw = self.raw_list[raw_idx]
        data = raw.get_data()[:, start:start+self.window_samples].copy()  # copy để tránh ảnh hưởng

        if self.augment:
            data = self._augment(data)

        # Chuyển về tensor float32
        return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

    def _augment(self, data):
        """Data augmentation: thêm nhiễu Gaussian, dịch chuyển thời gian nhẹ."""
        # Thêm nhiễu Gaussian nhỏ
        noise = np.random.normal(0, 0.01, data.shape)
        data = data + noise

        # Dịch chuyển thời gian ngẫu nhiên (tối đa 10 mẫu)
        shift = np.random.randint(-5, 5)
        if shift != 0:
            data = np.roll(data, shift, axis=1)
            # Đặt phần bị dịch về 0 (hoặc có thể giữ nguyên)
            if shift > 0:
                data[:, :shift] = 0
            else:
                data[:, shift:] = 0
        return data

def get_weighted_sampler(dataset):
    """Tạo sampler cân bằng theo class (dùng khi dữ liệu mất cân bằng)."""
    labels = [dataset.indices[i][2] for i in range(len(dataset))]
    class_counts = np.bincount(labels)
    weights = 1.0 / class_counts
    sample_weights = weights[labels]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    return sampler