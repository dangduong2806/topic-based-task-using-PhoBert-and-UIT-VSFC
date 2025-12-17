# scripts/prepare_data.py
import pandas as pd
import os

def process_uit_vsfc(data_dir, output_dir):
    # Các tập con trong dataset
    subsets = ['train', 'dev', 'test']
    
    # Tạo thư mục output nếu chưa có
    os.makedirs(output_dir, exist_ok=True)

    for subset in subsets:
        subset_path = os.path.join(data_dir, subset)
        
        # Đường dẫn file text gốc
        sents_path = os.path.join(subset_path, 'sents.txt')
        topics_path = os.path.join(subset_path, 'topics.txt')
        
        if not os.path.exists(sents_path):
            print(f"Skipping {subset}: File not found at {sents_path}")
            continue

        print(f"Processing {subset}...")

        # 1. Đọc dữ liệu text
        with open(sents_path, 'r', encoding='utf-8') as f:
            sents = [line.strip() for line in f.readlines()]
            
        # 2. Đọc nhãn sentiment
        with open(topics_path, 'r', encoding='utf-8') as f:
            topics = [line.strip() for line in f.readlines()]

        # Kiểm tra độ dài
        assert len(sents) == len(topics), f"Lỗi lệch dữ liệu ở tập {subset}"

        # 3. Tạo DataFrame
        df = pd.DataFrame({
            'text': sents,
            'label': topics
        })

        # Convert label sang số nguyên (int) để tránh lỗi khi training
        df['label'] = df['label'].astype(int)

        # 4. Lưu ra CSV
        output_file = os.path.join(output_dir, f"{subset}.csv")
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"--> Saved to {output_file} ({len(df)} samples)")

if __name__ == "__main__":
    # Cấu hình đường dẫn (có thể dùng Hydra ở đây, nhưng hardcode cho đơn giản script này)
    RAW_DATA_DIR = "./data/raw"
    PROCESSED_DATA_DIR = "./data/processed"
    
    process_uit_vsfc(RAW_DATA_DIR, PROCESSED_DATA_DIR)