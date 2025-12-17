import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
import pandas as pd
import torch
import os
from typing import Optional

class UITVSFCDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len):
        super().__init__()
        # File csv đã clean
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_len = max_len

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Không tìm thấy file dữ liệu tại: {data_path}")
        self.data = pd.read_csv(data_path)


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        text = str(self.data.iloc[index]['text'])
        label = self.data.iloc[index]['label']

        encoding = self.tokenizer.encode_plus(
            text,
            add_specila_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
    
class UITVSFCDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, model_name, batch_size, max_len, num_workers):
        super().__init__()
        self.save_hyperparameters()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None
    
    def setup(self, stage: Optional[str] = None):
        """
        Load dữ liệu và khởi tạo Dataset.
        Hàm này được gọi tự động bởi Trainer trên mỗi GPU (nếu train multi-gpu).
        """
        # Định nghĩa đường dẫn file
        train_path = os.path.join(self.hparams.data_dir, "train.csv")
        val_path = os.path.join(self.hparams.data_dir, "dev.csv") # UIT-VSFC thường dùng 'dev' cho validation
        test_path = os.path.join(self.hparams.data_dir, "test.csv")

        # Setup cho giai đoạn Train/Validate
        if stage == "fit" or stage is None:
            self.train_dataset = UITVSFCDataset(
                train_path, 
                self.tokenizer, 
                self.hparams.max_len
            )
            self.val_dataset = UITVSFCDataset(
                val_path, 
                self.tokenizer, 
                self.hparams.max_len
            )

        # Setup cho giai đoạn Test
        if stage == "test" or stage is None:
            self.test_dataset = UITVSFCDataset(
                test_path, 
                self.tokenizer, 
                self.hparams.max_len
            )
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True, # Train cần shuffle
            num_workers=self.hparams.num_workers,
            pin_memory=True # Tăng tốc khi dùng GPU
        )
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True
        )
    
    
    
