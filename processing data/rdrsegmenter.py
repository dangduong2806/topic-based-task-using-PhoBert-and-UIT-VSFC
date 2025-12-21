import pandas as pd
import py_vncorenlp
import os
from tqdm import tqdm # Thư viện hiện thanh tiến trình

# 1. Setup RDRSegmenter
if not os.path.exists('vncorenlp_models'):
    os.makedirs('vncorenlp_models')
    py_vncorenlp.download_model(save_dir='./vncorenlp_models')

rdrsegmenter = py_vncorenlp.VnCoreNLP(save_dir='./vncorenlp_models', annotators=["wseg"])

def segment_text_robust(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    try:
        sentences = rdrsegmenter.word_segment(text)
        return " ".join([" ".join(sent) for sent in sentences])
    except:
        return text

# 2. Danh sách các file cần xử lý
# Đảm bảo đường dẫn file của bạn chính xác
files = [
    "data/processed/train_augmented_2.csv", 
    "data/processed/dev.csv", 
    "data/processed/test.csv"
]

for file_path in files:
    if os.path.exists(file_path):
        print(f"Dang xu ly: {file_path}...")
        df = pd.read_csv(file_path)
        
        # Áp dụng tách từ cho cột 'text' (hoặc 'sents' tùy file của bạn)
        # Kiểm tra tên cột chứa văn bản
        col_name = 'sents' if 'sents' in df.columns else 'text'
        
        # Chạy tách từ với thanh hiển thị tiến trình
        tqdm.pandas()
        df[col_name] = df[col_name].progress_apply(segment_text_robust)
        
        # Lưu ra file mới có đuôi _segmented
        new_path = file_path.replace(".csv", "_segmented.csv")
        df.to_csv(new_path, index=False)
        print(f"✅ Đã xong! File lưu tại: {new_path}")
    else:
        print(f"❌ Không tìm thấy file: {file_path}")