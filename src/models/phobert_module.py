from typing import Optional

import pytorch_lightning as pl
import torch
from torch.optim import AdamW
from torchmetrics import Accuracy, F1Score
from transformers import AutoConfig, AutoModelForSequenceClassification

class PhoBERTClassifier(pl.LightningModule):
    """
    LightningModule cho PhoBERT Sentiment Analysis.
    Bao gồm kiến trúc model, logic training/val/test và optimizer.
    """
    def __init__(self, model_name, num_labels, learning_rate):
        super().__init__()
        # Lưu hyperparameters (model_name, learning_rate,...) vào checkpoint
        self.save_hyperparameters()

        # 1. Load Model Config & Architecture
        self.config = AutoConfig.from_pretrained(
            model_name, 
            num_labels=num_labels
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            config=self.config
        )

        # 2. Định nghĩa Metrics
        # Dùng Macro F1 vì dataset sentiment thường mất cân bằng (Imbalanced)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_labels)
        self.val_f1 = F1Score(task="multiclass", num_classes=num_labels, average="macro")
        
        self.test_acc = Accuracy(task="multiclass", num_classes=num_labels)
        self.test_f1 = F1Score(task="multiclass", num_classes=num_labels, average="macro")

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward pass của model.
        Nếu có labels, model sẽ trả về cả loss.
        """
        output = self.model(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            labels=labels
        )
        return output

    def training_step(self, batch, batch_idx):
        """Logic cho một bước huấn luyện (Training Step)"""
        # Forward pass
        outputs = self(
            input_ids=batch['input_ids'], 
            attention_mask=batch['attention_mask'], 
            labels=batch['labels']
        )
        
        loss = outputs.loss
        
        # Log loss để theo dõi trên WandB/Tensorboard
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Logic cho kiểm thử (Validation Step)"""
        outputs = self(
            input_ids=batch['input_ids'], 
            attention_mask=batch['attention_mask'], 
            labels=batch['labels']
        )
        
        loss = outputs.loss
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        labels = batch['labels']

        # Cập nhật metrics
        self.val_acc(preds, labels)
        self.val_f1(preds, labels)

        # Log
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_f1', self.val_f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        """Logic cho kiểm tra cuối cùng (Test Step)"""
        outputs = self(
            input_ids=batch['input_ids'], 
            attention_mask=batch['attention_mask'], 
            labels=batch['labels']
        )
        
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        labels = batch['labels']

        # Cập nhật metrics
        self.test_acc(preds, labels)
        self.test_f1(preds, labels)

        # Log
        self.log('test_acc', self.test_acc, on_epoch=True, logger=True)
        self.log('test_f1', self.test_f1, on_epoch=True, logger=True)

    def configure_optimizers(self):
        """Cấu hình Optimizer"""
        # AdamW là chuẩn mực cho BERT-based models
        optimizer = AdamW(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer