import mne
import numpy as np
import os
import warnings

def clean_eeg_signal(file_path, target_channels=None, apply_normalize=True):
    """
    Đọc file EDF, chuẩn hóa về 23 kênh, resample 250Hz, lọc 0.5-40 Hz.
    Nếu apply_normalize=True, chuẩn hóa từng kênh về mean=0, std=1.
    """
    if target_channels is None:
        target_channels = [
            'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
            'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
            'FZ-CZ', 'CZ-PZ', 'P7-T7', 'T7-P7', 'T8-P8', 'P8-O2', 'T8-P8'  # có thể lặp, nhưng sẽ xử lý sau
        ]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)

    # Resample về 250 Hz nếu cần
    if raw.info['sfreq'] != 250:
        raw.resample(250, npad="auto")

    # Chuẩn hóa tên kênh (xoá dấu chấm, viết hoa)
    raw.rename_channels(lambda x: x.strip('.').upper())
    ch_names = raw.ch_names
    data = raw.get_data()

    # Tạo mảng 23 kênh, điền zero nếu thiếu
    final_data = np.zeros((23, data.shape[1]))
    found_count = 0
    for i, target in enumerate(target_channels):
        # Tìm tên kênh trong file
        matched = False
        # Ưu tiên tìm chính xác
        if target in ch_names:
            idx = ch_names.index(target)
            final_data[i] = data[idx]
            found_count += 1
            matched = True
        else:
            # Thử tìm với tên gốc (bỏ hậu tố -1, -2)
            base = target.split('-')[0] + '-' + target.split('-')[1] if '-' in target else target
            if base in ch_names:
                idx = ch_names.index(base)
                final_data[i] = data[idx]
                found_count += 1
                matched = True
        if not matched:
            # Nếu không tìm thấy, có thể dùng kênh gần nhất? Ở đây ta để zero.
            pass

    # Tạo RawArray mới với tên kênh đã chuẩn hoá
    info = mne.create_info(ch_names=target_channels, sfreq=250, ch_types='eeg')
    new_raw = mne.io.RawArray(final_data, info, verbose=False)
    new_raw.filter(0.5, 40, fir_design='firwin', verbose=False)

    # Chuẩn hóa từng kênh (z-score)
    if apply_normalize:
        data_normalized = (new_raw.get_data().T - new_raw.get_data().mean(axis=1)) / (new_raw.get_data().std(axis=1) + 1e-8)
        new_raw = mne.io.RawArray(data_normalized.T, info, verbose=False)

    print(f"Đã tìm thấy {found_count} kênh trong {os.path.basename(file_path)} (23 kênh mục tiêu).")
    return new_raw

def get_seizure_details(data_root='data/chb-mit'):
    """
    Duyệt tất cả thư mục con chb* trong data_root, đọc file *-summary.txt
    Trả về dict: {tên_file: [(start, end), ...]}
    """
    seizure_info = {}
    if not os.path.exists(data_root):
        print(f"Thư mục {data_root} không tồn tại.")
        return seizure_info

    for folder in os.listdir(data_root):
        if folder.startswith('chb') and os.path.isdir(os.path.join(data_root, folder)):
            summary_file = os.path.join(data_root, folder, f'{folder}-summary.txt')
            if not os.path.exists(summary_file):
                continue
            with open(summary_file, 'r') as f:
                lines = f.readlines()
            current_file = None
            start = None
            for line in lines:
                line = line.strip()
                if 'File Name' in line:
                    current_file = line.split(':')[1].strip()
                elif 'Seizure Start Time' in line:
                    start_str = line.split(':')[1].strip().split()[0]
                    try:
                        start = float(start_str)
                    except:
                        continue
                elif 'Seizure End Time' in line:
                    end_str = line.split(':')[1].strip().split()[0]
                    try:
                        end = float(end_str)
                    except:
                        continue
                    if current_file and start is not None:
                        if current_file not in seizure_info:
                            seizure_info[current_file] = []
                        seizure_info[current_file].append((start, end))
                        start = None
    return seizure_info