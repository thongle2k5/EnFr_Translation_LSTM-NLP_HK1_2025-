import os
import requests
from tqdm import tqdm  # Thư viện hiển thị thanh tiến trình (pip install tqdm)

# 1. Cấu hình đường dẫn lưu file
DATA_DIR = "data/raw"
os.makedirs(DATA_DIR, exist_ok=True)

# 2. Các URL chính thức của bộ dữ liệu Multi30K (Task 1)
# Nguồn: https://github.com/multi30k/dataset
BASE_URL = "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw"

files_to_download = {
    "train.en": f"{BASE_URL}/train.en",
    "train.fr": f"{BASE_URL}/train.fr",
    "val.en":   f"{BASE_URL}/val.en",
    "val.fr":   f"{BASE_URL}/val.fr",
    "test.en":  f"{BASE_URL}/test_2016_flickr.en", # Test set chuẩn 2016
    "test.fr":  f"{BASE_URL}/test_2016_flickr.fr"
}

# Lưu ý: File test trong đề bài ghi là "test.en/fr", trên repo gốc nó thường tên là test_2016...
# Script này sẽ tải về và đổi tên thành test.en / test.fr cho đúng chuẩn đề bài.

def download_file(url, save_path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        total_size = int(response.headers.get('content-length', 0))
        with open(save_path, 'wb') as file, tqdm(
            desc=save_path,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                file.write(data)
                bar.update(len(data))
        print(f"✅ Đã tải: {save_path}")
    else:
        print(f"❌ Lỗi tải: {url}")

print("⏳ Đang bắt đầu tải dữ liệu Multi30K (En-Fr)...")

for filename, url in files_to_download.items():
    save_path = os.path.join(DATA_DIR, filename)
    
    # Nếu tải file test có tên dài dòng, ta lưu ngắn gọn lại theo đề bài
    if "test_2016" in url:
        if url.endswith(".en"): save_path = os.path.join(DATA_DIR, "test.en")
        if url.endswith(".fr"): save_path = os.path.join(DATA_DIR, "test.fr")

    if not os.path.exists(save_path):
        download_file(url, save_path)
    else:
        print(f"ℹ️ File đã tồn tại: {save_path}")

print("\n🎉 Hoàn tất! Kiểm tra thư mục 'data/raw/'")