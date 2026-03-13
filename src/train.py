import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import os
import numpy as np
from preprocess import clean_eeg_signal, get_seizure_details
from dataset import SeizureDataset, get_weighted_sampler
from model import TinyEEGNet
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Sử dụng thiết bị: {device}")

def plot_loss(train_losses, val_losses, val_accs, save_path='training_curves.png'):
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1,2,2)
    plt.plot(val_accs, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    print(f"📈 Đã lưu biểu đồ tại {save_path}")

def train():
    # Đường dẫn dữ liệu
    data_dir = os.path.join("data", "chb-mit", "chb01")
    if not os.path.exists(data_dir):
        print("❌ Không tìm thấy thư mục dữ liệu. Hãy chạy download_data.py trước.")
        return

    # Lấy thông tin động kinh
    seizure_info = get_seizure_details("data/chb-mit")

    # Liệt kê tất cả file .edf trong chb01
    all_edf = [f for f in os.listdir(data_dir) if f.endswith('.edf')]
    print(f"📁 Tìm thấy {len(all_edf)} file EDF.")

    # Tiền xử lý từng file và lưu raw vào list
    raw_list = []
    seizure_times_list = []
    for fname in all_edf:
        print(f"⏳ Đang xử lý {fname}...")
        file_path = os.path.join(data_dir, fname)
        raw = clean_eeg_signal(file_path, apply_normalize=True)
        raw_list.append(raw)
        times = seizure_info.get(fname, [])
        seizure_times_list.append(times)

    # Tạo dataset (có augmentation cho training)
    full_dataset = SeizureDataset(raw_list, seizure_times_list, normal_ratio=4, augment=False)  # augment sẽ bật sau khi split

    # Chia train/validation (80/20) theo tỷ lệ stratified
    labels = [full_dataset.indices[i][2] for i in range(len(full_dataset))]
    from sklearn.model_selection import train_test_split
    train_idx, val_idx = train_test_split(np.arange(len(full_dataset)), test_size=0.2,
                                          stratify=labels, random_state=42)

    # Tạo subset
    train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
    val_dataset = torch.utils.data.Subset(full_dataset, val_idx)

    # Bật augmentation cho train
    full_dataset.augment = True  # chú ý: augment ảnh hưởng đến cả subset vì cùng tham chiếu

    # Tạo sampler cân bằng cho train (tùy chọn)
    # train_sampler = get_weighted_sampler(train_dataset)  # nếu dùng, bỏ shuffle=True

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)

    # Khởi tạo model
    model = TinyEEGNet(dropout_rate=0.5).to(device)

    # Loss có trọng số để xử lý mất cân bằng (nếu cần)
    # Tính tỷ lệ lớp trên tập train
    train_labels = [full_dataset.indices[i][2] for i in train_idx]
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * 2  # chuẩn hóa
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    epochs = 50
    best_val_acc = 0.0
    best_model_path = "models/seizure_model_best.pth"
    train_losses, val_losses, val_accs = [], [], []

    print("🚀 Bắt đầu huấn luyện...")
    for epoch in range(1, epochs+1):
        model.train()
        train_loss = 0.0
        for signals, labels in train_loader:
            signals, labels = signals.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(signals)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for signals, labels in val_loader:
                signals, labels = signals.to(device), labels.to(device)
                outputs = model(signals)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_acc = 100 * correct / total

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Epoch {epoch:2d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Lưu model tốt nhất
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"   💾 Đã lưu model tốt nhất với Val Acc: {best_val_acc:.2f}%")

    # Lưu model cuối cùng
    torch.save(model.state_dict(), "models/seizure_model.pth")
    print("✅ Đã lưu model cuối cùng tại models/seizure_model.pth")

    # Vẽ biểu đồ
    plot_loss(train_losses, val_losses, val_accs)

if __name__ == "__main__":
    train()