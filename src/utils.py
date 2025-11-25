import torch
import torch.nn as nn
import time
import math

def init_weights(m):
    """
    Khởi tạo trọng số ngẫu nhiên nhưng theo phân phối chuẩn (Normal Distribution).
    Giúp model hội tụ nhanh hơn.
    """
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

def count_parameters(model):
    """Đếm xem model có bao nhiêu tham số cần học"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def epoch_time(start_time, end_time):
    """Tính thời gian chạy 1 epoch"""
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs