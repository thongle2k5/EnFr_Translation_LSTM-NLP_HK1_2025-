# EnFr_Translation_LSTM-NLP_HK1_2025-

# I cấu trúc thư mục

EnFr_Translation_LSTM/
├── data/
│ ├── raw/ # Dữ liệu thô (train.en, train.fr,...)
│ └── processed/ # Dữ liệu đã qua xử lý (nếu có)
├── notebooks/
│ └── set_data.ipynb # Notebook chuẩn bị dữ liệu & DataLoader
├── src/
│ ├── model.py # (Dự kiến) Chứa class Encoder, Decoder, Seq2Seq
│ └── train.py # (Dự kiến) Vòng lặp huấn luyện
├── checkpoints/ # Nơi lưu model tốt nhất (best_model.pth)
├── README.md # Tài liệu hướng dẫn
└── requirements.txt # Danh sách thư viện cần thiết

# II cài đặt môi trường

## 1 tạo môi trường ảo

## 2 cài đặt thư viện

--- Cài đặt các thư viện chính
pip install torch torchtext spacy

------ Fix lỗi xung đột version (hạ phiên bản python xuống dưới 2.0)
pip install "numpy<2.0"

-----Tải gói ngôn ngữ cho Spacy
python -m spacy download en_core_web_sm
python -m spacy download fr_core_news_sm
