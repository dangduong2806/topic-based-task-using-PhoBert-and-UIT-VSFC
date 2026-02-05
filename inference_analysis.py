import torch
import pandas as pd
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
# Import Class Model của bạn (Giả sử bạn để trong file model.py)
# Nếu bạn định nghĩa class trong notebook thì copy class đó vào đây
from src.models.phobert_module import PhoBERTClassifier
from tqdm import tqdm

# Cấu hình
MODEL_PATH = "/kaggle/input/checkpoint-main/model2.ckpt"
TEST_DATA_PATH = "data/processed/test_segmented.csv"
OUTPUT_FILE = "error_analysis_report.xlsx"

LABEL_MAP = {0: 'Lecturer', 1: 'Curriculum', 2: 'Facility', 3: 'Others'}

def run_error_analysis():
    # 1. Load model and tokenizer
    print(f"Loading model from {MODEL_PATH}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = PhoBERTClassifier.load_from_checkpoint(MODEL_PATH)
        print(f"Load model thành công bằng lightning")
    except Exception as e:
        model = PhoBERTClassifier(
            model_name= "vinai/phobert-large",
            num_labels=4,
            learning_rate=2e-5
        )
        checkpoint = torch.load(MODEL_PATH, map_location='cpu')
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        print(f"Load model thủ công thành công")
    
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-large")

    # 2. Load test data
    print(f"Loading test data from {TEST_DATA_PATH}...")
    df_test = pd.read_csv(TEST_DATA_PATH)
    texts = df_test['text'].tolist()
    labels = df_test['label'].tolist()

    print("Running inference...")
    
    errors = [] # Danh sách chứa các ca đoán sai
    with torch.no_grad():
        for i in tqdm(range(len(texts))):
            text = texts[i]
            true_label_idx = int(labels[i])

            # Tokenize
            encoding = tokenizer(
                text,
                return_tensors='pt',
                max_length=128,
                padding='max_length',
                truncation=True
            )

            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)

            # Predict
            output = model(input_ids, attention_mask)
            # Output là logits, lấy argmax để dự đoán
            logits = output.logits
            pred_label_idx = torch.argmax(logits, dim=1).item()

            # So sánh nếu sai thì lưu lại
            if pred_label_idx != true_label_idx:
                true_label_str = LABEL_MAP[true_label_idx]
                pred_label_str = LABEL_MAP[pred_label_idx]

                # Tạo tag lỗi
                error_type = f"{true_label_str}_predicted_as_{pred_label_str}"

                errors.append({
                    "Original_text": text,
                    "True_label": true_label_str,
                    "Predicted_label": pred_label_str,
                    "Error_Type": error_type
                })
    if errors:
        df_errors = pd.DataFrame(errors)

        # Thống kê số lượng lỗi từng loại
        print("\n--- THỐNG KÊ LỖI (TOP MISTAKES) ---")
        print(df_errors['Error_Type']).value_counts()

        # Lưu ra excel
        df_errors.to_excel(OUTPUT_FILE, index=False)
        print(f"\nĐã lưu chi tiết {len(df_errors)} mẫu sai vào file: {OUTPUT_FILE}")
        print("Hãy mở file này lên để phân tích ngữ nghĩa!")
    else:
        print("Chúc mừng! Model đoán đúng 100%")

if __name__ == "__main__":
    run_error_analysis()