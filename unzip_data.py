import gzip
import shutil
import os

# Đây là đường dẫn đến thư mục chứa mấy cái "vali" của bạn
RAW_DIR = "data/raw"

# Bên trái là tên file nén (vali), Bên phải là tên file ta muốn tạo ra (quần áo)
files_map = {
    "train.en.gz": "train.en",
    "train.fr.gz": "train.fr",
    "val.en.gz":   "val.en",
    "val.fr.gz":   "val.fr",
    # Đây là bước quan trọng: Đổi tên file test cho gọn và chuẩn đề bài
    "test_2016_flickr.en.gz": "test.en",
    "test_2016_flickr.fr.gz": "test.fr"
}

print("🔨 Đang bắt đầu giải nén...")

# Vòng lặp đi qua từng cặp tên file ở trên
for gz_name, new_name in files_map.items():
    
    # Tạo đường dẫn đầy đủ (VD: data/raw/train.en.gz)
    path_to_zip = os.path.join(RAW_DIR, gz_name)
    path_to_new = os.path.join(RAW_DIR, new_name)
    
    # Kiểm tra xem file nén có tồn tại không
    if os.path.exists(path_to_zip):
        # Mở file nén (rb = read binary)
        with gzip.open(path_to_zip, 'rb') as f_in:
            # Mở file mới để ghi vào (wb = write binary)
            with open(path_to_new, 'wb') as f_out:
                # Copy nội dung từ file nén sang file mới
                shutil.copyfileobj(f_in, f_out)
        print(f"✅ Đã giải nén xong: {new_name}")
    else:
        print(f"⚠️ Không tìm thấy file: {gz_name}")

print("\n🎉 Hoàn tất! Giờ bạn có thể dùng Notepad mở các file mới để xem chữ bên trong.")