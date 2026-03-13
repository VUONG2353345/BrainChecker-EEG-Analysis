import os
import requests
from tqdm import tqdm
from pathlib import Path

def download_chb_mit_comprehensive():
    # Cấu hình các tham số cơ bản
    base_url = "https://physionet.org/files/chbmit/1.0.0/"
    data_dir = Path("data/chb-mit")
    data_dir.mkdir(parents=True, exist_ok=True)

    # 1. Truy xuất danh sách RECORDS chính thức (chứa đường dẫn của tất cả các file .edf)
    print("Đang truy xuất danh sách tệp tin từ PhysioNet...")
    try:
        response = requests.get(base_url + "RECORDS", timeout=15)
        response.raise_for_status()
    except Exception as e:
        print(f"Lỗi kết nối với máy chủ PhysioNet: {e}")
        return

    # Danh sách các tệp tín hiệu .edf
    edf_files = response.text.strip().split('\n')
    
    # 2. Tự động xác định danh sách bệnh nhân và các tệp summary tương ứng
    # Bệnh nhân được định danh theo thư mục (chb01, chb02, ..., chb24)
    patients = sorted(list(set([f.split('/')[0] for f in edf_files])))
    summary_files = [f"{p}/{p}-summary.txt" for p in patients]
    
    # Tổng hợp danh sách tất cả các tệp cần kiểm tra (EDF + Summary + Info)
    raw_list = edf_files + summary_files + ["RECORDS-WITH-SEIZURES", "SUBJECT-INFO"]
    check_list = sorted(raw_list)
    
    print(f"Hệ thống đã xác định {len(check_list)} tệp tin cần xác minh.")

    # 3. Duyệt qua từng tệp để kiểm tra và tải xuống nếu thiếu
    for file_path in check_list:
        local_path = data_dir / file_path
        url = base_url + file_path
        
        # Đảm bảo thư mục con cục bộ đã tồn tại
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Kiểm tra trạng thái tệp cục bộ
        if local_path.exists() and local_path.stat().st_size > 0:
            # Nếu tệp đã tồn tại và không trống, bỏ qua
            continue

        # Tiến hành tải xuống nếu tệp chưa có hoặc bị lỗi (0 byte)
        try:
            # Sử dụng stream=True để tải các tệp lớn (EDF) hiệu quả hơn
            resp = requests.get(url, stream=True, timeout=20)
            if resp.status_code == 200:
                total_size = int(resp.headers.get('content-length', 0))
                
                with open(local_path, 'wb') as f, tqdm(
                    desc=f"Tải xuống {file_path}",
                    total=total_size,
                    unit='iB',
                    unit_scale=True,
                    unit_divisor=1024,
                    leave=False
                ) as bar:
                    for chunk in resp.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            bar.update(len(chunk))
            else:
                print(f"\nBỏ qua {file_path}: Máy chủ phản hồi mã lỗi {resp.status_code}")
        except Exception as e:
            print(f"\nLỗi khi tải {file_path}: {e}")

    print("\n" + "="*50)
    print("XÁC MINH HOÀN TẤT: Toàn bộ dữ liệu từ chb01 đến cuối đã sẵn sàng.")
    print(f"Thư mục lưu trữ: {data_dir.absolute()}")
    print("="*50)

if __name__ == "__main__":
    download_chb_mit_comprehensive()