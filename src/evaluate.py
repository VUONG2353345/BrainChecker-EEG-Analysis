import torch
import os
import numpy as np
from preprocess import clean_eeg_signal, get_seizure_details
from dataset import SeizureDataset
from model import TinyEEGNet
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(test_file, model_path="models/seizure_model_best.pth"):
    data_dir = os.path.join("data", "chb-mit", "chb01")
    if not os.path.exists(test_file):
        test_path = os.path.join(data_dir, test_file)
    else:
        test_path = test_file

    if not os.path.exists(test_path):
        print(f"❌ Không tìm thấy file {test_path}")
        return

    # Nạp model
    model = TinyEEGNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Lấy thông tin động kinh
    seizure_info = get_seizure_details("data/chb-mit")
    times = seizure_info.get(os.path.basename(test_path), [])

    # Tiền xử lý
    raw = clean_eeg_signal(test_path, apply_normalize=True)
    test_ds = SeizureDataset([raw], [times], normal_ratio=1, augment=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for signals, labels in test_loader:
            signals = signals.to(device)
            outputs = model(signals)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    print(f"\n📊 KẾT QUẢ ĐÁNH GIÁ FILE: {test_file}")
    print(confusion_matrix(all_labels, all_preds))
    print(classification_report(all_labels, all_preds, target_names=['Normal', 'Seizure']))
    if len(np.unique(all_labels)) > 1:
        auc = roc_auc_score(all_labels, all_probs)
        print(f"AUC: {auc:.4f}")

if __name__ == "__main__":
    evaluate('chb01_26.edf')