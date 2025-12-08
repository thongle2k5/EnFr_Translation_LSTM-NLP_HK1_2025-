# src/data_loader.py
import torch
import io
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

# Khai báo các hằng số
PAD_TOKEN = '<pad>'
SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
UNK_TOKEN = '<unk>'
#tokenization
def get_tokenizers():
    try:
        en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
        fr_tokenizer = get_tokenizer('spacy', language='fr_core_news_sm')
        return en_tokenizer, fr_tokenizer
    except OSError:
        print("Lỗi get_tokenizers")
        return None, None
#xây dựng từ điển (Vocabulary)
def build_vocabulary(filepath, tokenizer):
    def yield_tokens(file_path):
        with io.open(file_path, encoding='utf-8') as f:
            for line in f:
                yield tokenizer(line.strip())
                
    special_tokens = [UNK_TOKEN, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN]
    vocab = build_vocab_from_iterator(
        yield_tokens(filepath),
        specials=special_tokens,
        max_tokens=10000
    )
    vocab.set_default_index(vocab[UNK_TOKEN])
    return vocab

def text_transform(tokenizer, vocab, text):
    token_list = tokenizer(text.strip())
    sos_idx = vocab[SOS_TOKEN]
    eos_idx = vocab[EOS_TOKEN]
    index_list = [vocab[token] for token in token_list]
    return torch.tensor([sos_idx] + index_list + [eos_idx])

# Hàm tạo Collate_fn với biến vocab được truyền vào (Closure)
def create_collate_fn(en_tokenizer, fr_tokenizer, vocab_en, vocab_fr):
    pad_idx = vocab_en[PAD_TOKEN]
    
    def collate_batch(batch):
        src_batch, trg_batch = [], []
        src_lens = []
        
        for src_sample, trg_sample in batch:
            src_item = text_transform(en_tokenizer, vocab_en, src_sample)
            trg_item = text_transform(fr_tokenizer, vocab_fr, trg_sample)
            src_batch.append(src_item)
            trg_batch.append(trg_item)
            src_lens.append(len(src_item))
            
        # Sorting for packing
        zipped = list(zip(src_batch, trg_batch, src_lens))
        zipped.sort(key=lambda x: x[2], reverse=True)
        src_batch, trg_batch, src_lens = zip(*zipped)
        
        src_batch = pad_sequence(src_batch, padding_value=pad_idx)
        trg_batch = pad_sequence(trg_batch, padding_value=pad_idx)
        
        return src_batch, trg_batch, torch.tensor(src_lens)
        
    return collate_batch
#data loader
def get_loaders(batch_size, path_en, path_fr):
    # 1. Tokenizers
    en_tokenizer, fr_tokenizer = get_tokenizers()
    
    # 2. Vocab
    print("⏳ Building Vocab...")
    vocab_en = build_vocabulary(path_en, en_tokenizer)
    vocab_fr = build_vocabulary(path_fr, fr_tokenizer)
    
    # 3. Data Loading
    def read_raw_data(p_en, p_fr):
        with open(p_en, encoding='utf-8') as f_en, open(p_fr, encoding='utf-8') as f_fr:
            return list(zip(f_en, f_fr))
            
    train_data = read_raw_data(path_en, path_fr)
    
    # 4. Collate function
    collate_fn = create_collate_fn(en_tokenizer, fr_tokenizer, vocab_en, vocab_fr)
    
    # 5. DataLoader
    train_loader = DataLoader(
        train_data, 
        batch_size=batch_size, 
        collate_fn=collate_fn, 
        shuffle=True
    )
    
    return train_loader, vocab_en, vocab_fr