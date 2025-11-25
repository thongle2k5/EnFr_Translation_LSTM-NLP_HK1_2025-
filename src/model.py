import torch
import torch.nn as nn
import random

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        # 1. Embedding: Biến số thứ tự từ (VD: 45) thành vector (VD: [0.1, -0.5...])
        self.embedding = nn.Embedding(input_dim, emb_dim)
        
        # 2. LSTM: Tim của Encoder
        self.lstm = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        
        # 3. Dropout: Giúp chống học vẹt (Overfitting)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        # src shape: [src_len, batch_size]
        
        embedded = self.dropout(self.embedding(src))
        # embedded shape: [src_len, batch_size, emb_dim]
        
        outputs, (hidden, cell) = self.lstm(embedded)
        
        # outputs: chứa hidden state của tất cả các bước (chúng ta ko dùng cái này trong model cơ bản)
        # hidden, cell: Trạng thái cuối cùng của câu (Đây chính là CONTEXT VECTOR)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        # 1. Embedding
        self.embedding = nn.Embedding(output_dim, emb_dim)
        
        # 2. LSTM
        self.lstm = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        
        # 3. Linear: Biến đổi output của LSTM ra kích thước từ điển Pháp để dự đoán từ
        self.fc_out = nn.Linear(hid_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell):
        # input shape: [batch_size] (Vì Decoder giải mã từng từ một)
        # Chúng ta cần unsqueeze để thêm chiều seq_len = 1
        input = input.unsqueeze(0)
        # input shape: [1, batch_size]
        
        embedded = self.dropout(self.embedding(input))
        # embedded shape: [1, batch_size, emb_dim]
        
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        # output shape: [1, batch_size, hid_dim]
        
        prediction = self.fc_out(output.squeeze(0))
        # prediction shape: [batch_size, output_dim]
        
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src: [src_len, batch_size] (Câu Tiếng Anh)
        # trg: [trg_len, batch_size] (Câu Tiếng Pháp - Đáp án)
        
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        # Tensor chứa kết quả dự đoán
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        # 1. Đưa câu Anh qua Encoder để lấy Context Vector (hidden, cell)
        hidden, cell = self.encoder(src)
        
        # 2. Bắt đầu giải mã
        # Input đầu tiên cho Decoder luôn là thẻ <sos> (Start of Sentence)
        input = trg[0, :]
        
        # Vòng lặp giải mã từng từ một
        for t in range(1, trg_len):
            # Chạy Decoder 1 bước
            output, hidden, cell = self.decoder(input, hidden, cell)
            
            # Lưu kết quả dự đoán vào tensor outputs
            outputs[t] = output
            
            # --- Teacher Forcing ---
            # Quyết định xem input cho bước tiếp theo là gì?
            # Cách 1 (Học vẹt): Lấy đáp án đúng từ trg (target) đưa vào.
            # Cách 2 (Tự lực): Lấy từ mà model vừa dự đoán ra (top1) đưa vào.
            teacher_force = random.random() < teacher_forcing_ratio
            
            top1 = output.argmax(1) 
            input = trg[t] if teacher_force else top1
            
        return outputs