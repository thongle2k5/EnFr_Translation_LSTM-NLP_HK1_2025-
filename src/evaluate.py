import torch
from torchtext.data.metrics import bleu_score
from torch.utils.data import DataLoader
import sys
import os
from src.data_loader import get_tokenizers, build_vocabulary, create_collate_fn
from src.model import Seq2Seq, Encoder, Decoder

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

def calculate_bleu_score():
    train_path_en = os.path.join(project_root, 'data', 'raw', 'train.en')
    train_path_fr = os.path.join(project_root, 'data', 'raw', 'train.fr')
    test_path_en = os.path.join(project_root, 'data', 'raw', 'val.en')
    test_path_fr = os.path.join(project_root, 'data', 'raw', 'val.fr')
    checkpoint_path = os.path.join(project_root, 'checkpoints', 'best_model.pth')

    en_tokenizer, fr_tokenizer = get_tokenizers()
    vocab_en = build_vocabulary(train_path_en, en_tokenizer)
    vocab_fr = build_vocabulary(train_path_fr, fr_tokenizer)
    print(f" Kích thước từ điển: Anh={len(vocab_en)}, Pháp={len(vocab_fr)}")

    with open(test_path_en, encoding='utf-8') as f_en, open(test_path_fr, encoding='utf-8') as f_fr:
        test_data = list(zip(f_en, f_fr))

    collate_fn = create_collate_fn(en_tokenizer, fr_tokenizer, vocab_en, vocab_fr)
    test_loader = DataLoader(test_data, batch_size=32, collate_fn=collate_fn, shuffle=False)
    INPUT_DIM = len(vocab_en)
    OUTPUT_DIM = len(vocab_fr)

    enc = Encoder(INPUT_DIM, 256, 512, 2, 0.5)
    dec = Decoder(OUTPUT_DIM, 256, 512, 2, 0.5)
    model = Seq2Seq(enc, dec, device).to(device)

    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print(f"Không tìm thấy file model")
        return
    
    model.eval()
    trgs = []
    pred_trgs = []
    with torch.no_grad():
        for i, (src, trg, src_len) in enumerate(test_loader):
            src = src.to(device)
            trg = trg.to(device)
            output = model(src, trg, src_len, teacher_forcing_ratio=0)
            preds = output.argmax(2) 
            for j in range(preds.shape[1]):
                pred_sent = preds[1:, j].tolist() 
                eos_idx = vocab_fr['<eos>']
                if eos_idx in pred_sent:
                    pred_sent = pred_sent[:pred_sent.index(eos_idx)]
                pred_words = [vocab_fr.lookup_token(idx) for idx in pred_sent]
                pred_trgs.append(pred_words)
                trg_sent = trg[1:, j].tolist()
                if eos_idx in trg_sent:
                    trg_sent = trg_sent[:trg_sent.index(eos_idx)]
                trg_words = [vocab_fr.lookup_token(idx) for idx in trg_sent]
                trgs.append([trg_words])
    score = bleu_score(pred_trgs, trgs)
    print(f"   Score = {score*100:.2f} / 100")

if __name__ == "__main__":
    calculate_bleu_score()