import torch
import spacy
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

# --- 1. Cấu hình Token & Spacy ---
# Load mô hình ngôn ngữ Spacy (đã tải ở bước trước)
spacy_en = spacy.load("en_core_web_sm")
spacy_fr = spacy.load("fr_core_news_sm")

def tokenize_en(text):
    """Tách từ tiếng Anh: "Hello world." -> ["Hello", "world", "."]"""
    return [tok.text for tok in spacy_en.tokenizer(text)]

def tokenize_fr(text):
    """Tách từ tiếng Pháp"""
    return [tok.text for tok in spacy_fr.tokenizer(text)]

# Các chỉ số đặc biệt
UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3
SPECIAL_SYMBOLS = ['<unk>', '<pad>', '<sos>', '<eos>']

# --- 2. Class Tự Xây Dựng Từ Điển (Thay thế Torchtext) ---
class Vocab:
    def __init__(self, counter, min_freq=2):
        # Khởi tạo map: Token -> ID (bắt đầu bằng các token đặc biệt)
        self.stoi = {tok: i for i, tok in enumerate(SPECIAL_SYMBOLS)}
        self.itos = {i: tok for i, tok in enumerate(SPECIAL_SYMBOLS)}
        idx = len(SPECIAL_SYMBOLS)
        
        # Duyệt qua các từ đếm được, nếu xuất hiện đủ nhiều thì thêm vào từ điển
        for word, count in counter.items():
            if count >= min_freq:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1
                
    def __len__(self):
        return len(self.stoi)
        
    def __getitem__(self, token):
        # Lấy ID của token, nếu không có trả về UNK_IDX
        return self.stoi.get(token, UNK_IDX)

def build_vocab_manual(filepath, tokenizer):
    """Đọc file text và đếm tần suất từ"""
    counter = Counter()
    print(f"📖 Đang quét từ vựng file: {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = tokenizer(line.strip())
            counter.update(tokens)
    return Vocab(counter, min_freq=2)

# --- 3. Dataset & Transform ---
class EnFrDataset(Dataset):
    def __init__(self, src_path, trg_path, src_vocab, trg_vocab, src_tokenizer, trg_tokenizer):
        self.data = []
        print(f"loading data {src_path}...")
        with open(src_path, 'r', encoding='utf-8') as f_src, open(trg_path, 'r', encoding='utf-8') as f_trg:
            for line_src, line_trg in zip(f_src, f_trg):
                # Tokenize và chuyển sang ID ngay lúc load để code gọn
                src_tokens = [SOS_IDX] + [src_vocab[t] for t in src_tokenizer(line_src.strip())] + [EOS_IDX]
                trg_tokens = [SOS_IDX] + [trg_vocab[t] for t in trg_tokenizer(line_trg.strip())] + [EOS_IDX]
                self.data.append((torch.tensor(src_tokens), torch.tensor(trg_tokens)))
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    src_batch, trg_batch = [], []
    for src_item, trg_item in batch:
        src_batch.append(src_item)
        trg_batch.append(trg_item)
    
    # Padding
    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    trg_batch = pad_sequence(trg_batch, padding_value=PAD_IDX)
    return src_batch, trg_batch

# --- 4. Hàm Main gọi từ bên ngoài ---
def create_dataset_and_loaders(batch_size=128):
    # Bước 1: Xây dựng từ điển thủ công
    en_vocab = build_vocab_manual('data/raw/train.en', tokenize_en)
    fr_vocab = build_vocab_manual('data/raw/train.fr', tokenize_fr)
    
    print(f"✅ Đã xây xong Vocab! Anh: {len(en_vocab)}, Pháp: {len(fr_vocab)}")

    # Bước 2: Tạo Dataset
    train_ds = EnFrDataset('data/raw/train.en', 'data/raw/train.fr', en_vocab, fr_vocab, tokenize_en, tokenize_fr)
    val_ds = EnFrDataset('data/raw/val.en', 'data/raw/val.fr', en_vocab, fr_vocab, tokenize_en, tokenize_fr)
    test_ds = EnFrDataset('data/raw/test.en', 'data/raw/test.fr', en_vocab, fr_vocab, tokenize_en, tokenize_fr)

    # Bước 3: DataLoader
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader, en_vocab, fr_vocab