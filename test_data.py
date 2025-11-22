# File: test_data.py
import torch
# Import hàm từ file dataset.py của chúng ta
from src.dataset import create_dataset_and_loaders

if __name__ == "__main__":
    print("🚀 Đang khởi động kiểm tra dữ liệu...")
    
    try:
        # Thử load dữ liệu với batch_size nhỏ = 4
        train_loader, val_loader, test_loader, en_vocab, fr_vocab = create_dataset_and_loaders(batch_size=4)
        
        print("\n✅ Xử lý dữ liệu THÀNH CÔNG!")
        print(f" - Số từ vựng tiếng Anh: {len(en_vocab)}")
        print(f" - Số từ vựng tiếng Pháp: {len(fr_vocab)}")
        
        # Lấy thử 1 batch ra xem
        src_batch, trg_batch = next(iter(train_loader))
        
        print("\n📦 Kiểm tra kích thước 1 Batch:")
        print(f" - Shape Input (Anh): {src_batch.shape} (Format: [Seq_Len, Batch_Size])")
        print(f" - Shape Target (Pháp): {trg_batch.shape}")
        
        print("\n🎉 Dữ liệu đã sẵn sàng để đưa vào mô hình!")
        
    except Exception as e:
        print(f"\n❌ Có lỗi xảy ra: {e}")
        print("💡 Gợi ý: Kiểm tra lại xem đã tải đủ file vào data/raw/ chưa?")