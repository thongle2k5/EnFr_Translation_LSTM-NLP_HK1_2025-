import torch
import torch.nn as nn
import torch.optim as optim
import time
import math
from src.dataset import create_dataset_and_loaders, PAD_IDX
from src.model import Encoder, Decoder, Seq2Seq
from src.utils import init_weights, count_parameters, epoch_time

# --- 1. Cấu hình Hyperparameters ---
BATCH_SIZE = 128
N_EPOCHS = 10           # Số lần học lặp lại toàn bộ dữ liệu
CLIP = 1                # Cắt gradient để tránh bùng nổ (đặc trưng của LSTM)
LEARNING_RATE = 0.001

# Cấu hình Model (như trong đề bài gợi ý)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

def train(model, iterator, optimizer, criterion, clip):
    model.train() # Bật chế độ train (để Dropout hoạt động)
    epoch_loss = 0
    
    for i, (src, trg) in enumerate(iterator):
        src, trg = src.to(device), trg.to(device)
        
        optimizer.zero_grad() # Xóa sạch đạo hàm cũ
        
        # Forward pass
        output = model(src, trg)
        # output: [trg len, batch size, output dim]
        # trg: [trg len, batch size]
        
        # Reshape để tính loss (bỏ qua token đầu tiên <sos>)
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        
        # Tính sai số
        loss = criterion(output, trg)
        
        # Backward pass (Lan truyền ngược)
        loss.backward()
        
        # Cắt gradient để tránh lỗi exploding gradient
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        # Cập nhật trọng số
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval() # Bật chế độ kiểm tra (tắt Dropout)
    epoch_loss = 0
    
    with torch.no_grad(): # Không tính đạo hàm cho nhẹ máy
        for i, (src, trg) in enumerate(iterator):
            src, trg = src.to(device), trg.to(device)

            output = model(src, trg, teacher_forcing_ratio=0) # Tắt Teacher Forcing khi test
            
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
            
            loss = criterion(output, trg)
            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

if __name__ == "__main__":
    # Chọn thiết bị (ưu tiên GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Đang chạy trên thiết bị: {device}")

    # 1. Load dữ liệu
    print("⏳ Đang load dữ liệu...")
    train_loader, val_loader, test_loader, en_vocab, fr_vocab = create_dataset_and_loaders(BATCH_SIZE)
    
    INPUT_DIM = len(en_vocab)
    OUTPUT_DIM = len(fr_vocab)
    print(f"✅ Vocab size: Anh={INPUT_DIM}, Pháp={OUTPUT_DIM}")

    # 2. Khởi tạo Model
    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
    model = Seq2Seq(enc, dec, device).to(device)
    
    # Khởi tạo trọng số & đếm tham số
    model.apply(init_weights)
    print(f"📊 Mô hình có {count_parameters(model):,} tham số cần học.")

    # 3. Optimizer & Loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX) # Bỏ qua token <pad> khi tính lỗi

    # 4. Vòng lặp Training
    best_valid_loss = float('inf')
    
    print("\n🔥 BẮT ĐẦU HUẤN LUYỆN 🔥")
    for epoch in range(N_EPOCHS):
        start_time = time.time()
        
        train_loss = train(model, train_loader, optimizer, criterion, CLIP)
        valid_loss = evaluate(model, val_loader, criterion)
        
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        # Nếu loss giảm kỷ lục thì lưu model lại
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'checkpoints/best_model.pth')
            saved_msg = "💾 (Đã lưu model tốt nhất)"
        else:
            saved_msg = ""
        
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Val. Loss: {valid_loss:.3f} {saved_msg}')
        print(f'\tPPL: {math.exp(valid_loss):.3f}') # Perplexity (chỉ số độ bối rối của model)

    print("\n🎉 HUẤN LUYỆN HOÀN TẤT!")