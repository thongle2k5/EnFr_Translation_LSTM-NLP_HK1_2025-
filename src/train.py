import torch
import torch.nn as nn
import time
import math

def train_epoch(model, iterator, optimizer, criterion, clip, device):
    """
    Hàm huấn luyện cho 1 Epoch (1 lần duyệt qua toàn bộ dữ liệu train).
    """
    model.train() # Chuyển mô hình sang chế độ huấn luyện (bật Dropout)
    epoch_loss = 0
    
    for i, (src, trg, src_len) in enumerate(iterator):
        # Chuyển dữ liệu lên GPU/CPU
        src = src.to(device)
        trg = trg.to(device)
        
        # Xóa sạch gradient cũ trước khi tính cái mới
        optimizer.zero_grad()
        
        # Chạy mô hình (Forward)
        # trg bao gồm cả <sos> và <eos>
        output = model(src, trg, src_len)
        
        # --- TÍNH LOSS ---
        # Output shape: [trg len, batch size, output dim]
        # Trg shape:    [trg len, batch size]
        
        # Bỏ qua token đầu tiên <sos> vì nó không cần dự đoán
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        
        # Tính sai số (Loss)
        loss = criterion(output, trg)
        
        # Lan truyền ngược (Backward) để tính gradient
        loss.backward()
        
        # Cắt gradient (Gradient Clipping) để tránh bùng nổ gradient (lỗi NaN)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        # Cập nhật trọng số
        optimizer.step()
        
        epoch_loss += loss.item()
        
    # Trả về loss trung bình của epoch này
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion, device):
    """
    Hàm đánh giá mô hình trên tập Validation.
    Không update trọng số, chỉ kiểm tra xem học tốt không.
    """
    model.eval() # Chuyển sang chế độ đánh giá (tắt Dropout)
    epoch_loss = 0
    
    with torch.no_grad(): # Tắt tính toán gradient cho nhẹ máy
        for i, (src, trg, src_len) in enumerate(iterator):
            src = src.to(device)
            trg = trg.to(device)

            # Lưu ý: Khi evaluate, teacher_forcing_ratio = 0 (tắt nhắc bài)
            output = model(src, trg, src_len, teacher_forcing_ratio=0)

            # Bỏ token <sos>
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)
            epoch_loss += loss.item()
            
    return epoch_loss / len(iterator)

def epoch_time(start_time, end_time):
    """Hàm tính thời gian chạy 1 epoch"""
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs