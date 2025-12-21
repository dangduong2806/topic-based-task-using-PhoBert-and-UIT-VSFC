import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import yaml
from src.datamodules.uit_vsfc_module import UITVSFCDataModule
from src.models.phobert_module import PhoBERTClassifier


# Giả sử 'model' là model PhoBERT của bạn đã load checkpoint tốt nhất
# Giả sử 'test_loader' là dữ liệu test của bạn

def evaluate_and_report(model, data_loader, device):
    model.eval()
    model.to(device)
    
    all_preds = []
    all_labels = []
    
    print("Đang chạy đánh giá...")
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask, labels=None)
            _, preds = torch.max(outputs.logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 1. In Báo cáo chi tiết (Classification Report)
    # Đây chính là cái bảng giống hệt ảnh mẫu bạn thích
    target_names = ['Lecturer', 'Curriculum', 'Facility', 'Others']
    report = classification_report(all_labels, all_preds, target_names=target_names, digits=4)
    print("\n--- KẾT QUẢ CHI TIẾT (Standard Report) ---")
    print(report)
    
    # 2. Vẽ Ma trận nhầm lẫn (Confusion Matrix)
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(10, 8))
    # Vẽ heatmap màu xanh
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Dự đoán (Predicted)')
    plt.ylabel('Thực tế (Actual)')
    plt.title('Confusion Matrix')

    # THAY DÒNG plt.show() BẰNG DÒNG DƯỚI ĐÂY:
    plt.savefig('confusion_matrix.png') 
    print("✅ Đã lưu ảnh ma trận nhầm lẫn vào file: confusion_matrix.png")
    
    # Nếu muốn chắc ăn thì in luôn text ra
    print("\n--- MA TRẬN NHẦM LẪN (DẠNG SỐ) ---")
    print(cm)

# --- CÁCH GỌI HÀM ---
if __name__ == "__main__":
    config_path = "configs/config.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    dm = UITVSFCDataModule(
        data_dir=cfg['data']['data_dir'],
        model_name=cfg['model']['model_name'],
        batch_size=cfg['data']['batch_size'],
        max_len=cfg['data']['max_len'],
        num_workers=cfg['data']['num_workers']
    )
    dm.setup(stage="test")
    test_loader = dm.test_dataloader()
    if len(test_loader) == 0:
        print("Lỗi đường dẫn")
    print("\n🔥 DEBUG KIỂM TRA DỮ LIỆU THỰC TẾ 🔥")
    # Lấy 1 batch đầu tiên ra soi
    batch = next(iter(test_loader))
    input_ids = batch['input_ids']
    labels = batch['labels']

    # 1. Kiểm tra Label
    print(f"Labels (3 mẫu đầu): {labels[:3].cpu().numpy()}")

    # 2. Quan trọng nhất: GIẢI MÃ LẠI TEXT XEM CÓ DẤU GẠCH NỐI KHÔNG?
    # Dùng chính tokenizer của dm để decode
    decoded_text = dm.tokenizer.decode(input_ids[0], skip_special_tokens=True)
    print(f"👉 Text mẫu 1 model nhìn thấy: '{decoded_text}'")

    if "_" in decoded_text:
        print("✅ Text CÓ chứa dấu gạch nối (Data chuẩn Segmented).")
    else:
        print("❌ CẢNH BÁO: Text KHÔNG có dấu gạch nối! (Model đang đọc data thô)")
        
    print("------------------------------------------\n")
    # Load model
    checkpoint_path = "checkpoints/model2.ckpt"
    print(f"Đang load model từ: {checkpoint_path}")
    model = PhoBERTClassifier.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.freeze()
    print("✅ Đã load model thành công! Sẵn sàng để dự đoán.")

    evaluate_and_report(model, test_loader, device='cuda')