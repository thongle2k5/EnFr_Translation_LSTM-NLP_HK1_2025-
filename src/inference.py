import torch
import spacy
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import get_loaders
from src.model import Encoder, Decoder, Seq2Seq


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 64
PATH_EN = 'data/raw/train.en'
PATH_FR = 'data/raw/train.fr'
CHECKPOINT_PATH = 'checkpoints/best_model.pth'

def load_system():
    _, vocab_en, vocab_fr = get_loaders(BATCH_SIZE, PATH_EN, PATH_FR)
    INPUT_DIM = len(vocab_en)
    OUTPUT_DIM = len(vocab_fr)
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    HID_DIM = 512
    N_LAYERS = 2
    DROPOUT = 0.5
    
    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DROPOUT)
    model = Seq2Seq(enc, dec, DEVICE).to(DEVICE)
    if os.path.exists(CHECKPOINT_PATH):
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    else:
        print("Không tìm thấy file checkpoint !")
        return None, None, None
    
    return model, vocab_en, vocab_fr

def translate(sentence, model, vocab_en, vocab_fr, device, max_len=50):
    model.eval()
    if isinstance(sentence, str):
        nlp_en = spacy.load("en_core_web_sm")
        tokens = [token.text for token in nlp_en(sentence)]
    else:
        tokens = [token for token in sentence]

    tokens = ['<sos>'] + tokens + ['<eos>']
    
    src_indexes = [vocab_en['<sos>']] + [vocab_en[token] for token in tokens] + [vocab_en['<eos>']]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)
    src_len = torch.LongTensor([len(src_indexes)])
    
    with torch.no_grad():
        hidden, cell = model.encoder(src_tensor, src_len)

    trg_indexes = [vocab_fr['<sos>']]
    for i in range(max_len):
        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
        
        with torch.no_grad():
            output, hidden, cell = model.decoder(trg_tensor, hidden, cell)
        
        pred_token = output.argmax(1).item()
        trg_indexes.append(pred_token)
        if pred_token == vocab_fr['<eos>']:
            break
    
    trg_tokens = [vocab_fr.lookup_token(i) for i in trg_indexes]
    
    return trg_tokens[1:]

if __name__ == "__main__":
    model, vocab_en, vocab_fr = load_system()
    
    if model:
        print("\nHỆ THỐNG DỊCH MÁY ANH - PHÁP")
        print("(Gõ 'q' để thoát)\n")
        
        while True:
            sentence = input("Nhập câu tiếng Anh: ")
            if sentence.lower() == 'q':
                break
                
            translation = translate(sentence, model, vocab_en, vocab_fr, DEVICE)
        
            if translation[-1] == '<eos>':
                translation = translation[:-1]
                
            translated_text = " ".join(translation)
            
            print(f"Dịch sang tiếng Pháp: {translated_text}")
            print("-" * 40)