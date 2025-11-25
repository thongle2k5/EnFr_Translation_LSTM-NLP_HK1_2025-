import torch
from src.dataset import create_dataset_and_loaders
from src.model import Encoder, Decoder, Seq2Seq

# Cấu hình giả lập
INPUT_DIM = 1000  # Giả sử từ điển Anh có 1000 từ
OUTPUT_DIM = 1000 # Giả sử từ điển Pháp có 1000 từ
ENC_EMB_DIM = 32
DEC_EMB_DIM = 32
HID_DIM = 64
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

# Thiết bị (GPU hoặc CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    print(f"⚙️ Đang kiểm tra Model trên thiết bị: {device}")
    
    # 1. Khởi tạo các khối
    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
    model = Seq2Seq(enc, dec, device).to(device)
    
    print("✅ Khởi tạo Model thành công!")
    
    # 2. Tạo dữ liệu giả để test
    # Batch size = 4, Câu dài 10 từ
    src = torch.randint(0, INPUT_DIM, (10, 4)).to(device) # [src_len, batch_size]
    trg = torch.randint(0, OUTPUT_DIM, (12, 4)).to(device) # [trg_len, batch_size]
    
    print(f" - Shape Input (Anh): {src.shape}")
    print(f" - Shape Target (Pháp): {trg.shape}")
    
    # 3. Chạy thử (Forward pass)
    output = model(src, trg)
    
    print(f" - Shape Output (Dự đoán): {output.shape}")
    
    # Kiểm tra shape output
    # Output chuẩn phải là: [trg_len, batch_size, output_dim]
    expected_shape = (12, 4, 1000)
    
    if output.shape == expected_shape:
        print("\n🎉 CHÚC MỪNG! Model hoạt động chuẩn shape. Sẵn sàng để train!")
    else:
        print(f"\n❌ Sai shape rồi! Mong đợi {expected_shape}, nhưng nhận được {output.shape}")