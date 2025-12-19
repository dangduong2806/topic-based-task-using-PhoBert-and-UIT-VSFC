from typing import Optional

import pytorch_lightning as pl
import torch
from torch.optim import AdamW
from torchmetrics import Accuracy, F1Score
from torchmetrics import MetricCollection
from transformers import AutoConfig, AutoModelForSequenceClassification
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassF1Score
import wandb
import pandas as pd
import pytorch_lightning


class PhoBERTClassifier(pl.LightningModule):
    """
    LightningModule cho PhoBERT Sentiment Analysis.
    Bao gồm kiến trúc model, logic training/val/test và optimizer.
    """
    def __init__(self, model_name, num_labels, learning_rate, class_names: list = None, weight_decay: float = None, dropout: float = None):
        super().__init__()
        # Lưu hyperparameters (model_name, learning_rate,...) vào checkpoint
        self.save_hyperparameters()

        # 1. Load Model Config & Architecture
        self.config = AutoConfig.from_pretrained(
            model_name, 
            num_labels=num_labels
        )
        self.config.hidden_dropout_prob = dropout
        self.config.attention_probs_dropout_prob = dropout

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            config=self.config
        )
        self.class_names = class_names if class_names else [f"Class_{i}" for i in range(num_labels)]

        # --- METRICS SETUP ---
        # 1. Metric tổng hợp (Weighted Average) - Để theo dõi quá trình train
        metrics = MetricCollection({
            'acc': MulticlassPrecision(num_classes=num_labels, average='weighted'), # Tạm dùng Precision weighted làm acc đại diện
            'recall': MulticlassRecall(num_classes=num_labels, average='weighted'),
            'f1': MulticlassF1Score(num_classes=num_labels, average='weighted')
        })
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')

        # 2. Metric chi tiết từng class (Average = None) - Dùng để vẽ bảng cuối epoch
        self.per_class_precision = MulticlassPrecision(num_classes=num_labels, average=None)
        self.per_class_recall = MulticlassRecall(num_classes=num_labels, average=None)
        self.per_class_f1 = MulticlassF1Score(num_classes=num_labels, average=None)
        
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
        preds = torch.argmax(outputs.logits, dim=1)
        
        # Log metrics cơ bản
        output_metrics = self.train_metrics(preds, batch['labels'])
        self.log_dict(output_metrics, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        """Logic cho kiểm thử (Validation Step)"""
        outputs = self(
            input_ids=batch['input_ids'], 
            attention_mask=batch['attention_mask'], 
            labels=batch['labels']
        )
        loss = outputs.loss
        preds = torch.argmax(outputs.logits, dim=1)
        labels = batch['labels']

        # 1. Update metric tổng
        self.val_metrics.update(preds, labels)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # 2. Update metric chi tiết (tích lũy để tính cuối epoch)
        self.per_class_precision.update(preds, labels)
        self.per_class_recall.update(preds, labels)
        self.per_class_f1.update(preds, labels)

        return loss
        
    def test_step(self, batch, batch_idx):
        """Logic cho kiểm tra cuối cùng (Test Step)"""
        outputs = self(
            input_ids=batch['input_ids'], 
            attention_mask=batch['attention_mask'], 
            labels=batch['labels']
        )
        
        preds = torch.argmax(outputs.logits, dim=1)
        self.test_metrics(preds, batch['labels'])
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        """Cấu hình Optimizer"""
        # AdamW là chuẩn mực cho BERT-based models
        optimizer = AdamW(
            self.parameters(), 
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
            )
        return optimizer
    
    def on_validation_epoch_end(self):
        # 1. Tính toán metric tổng và log
        output_metrics = self.val_metrics.compute()
        self.log_dict(output_metrics)
        self.val_metrics.reset()

        # 2. Tính toán metric chi tiết từng class
        precisions = self.per_class_precision.compute().cpu().tolist()
        recalls = self.per_class_recall.compute().cpu().tolist()
        f1s = self.per_class_f1.compute().cpu().tolist()

        # Reset sau khi tính
        self.per_class_precision.reset()
        self.per_class_recall.reset()
        self.per_class_f1.reset()

        # 3. TẠO BẢNG LOG WANDB (Giống hình bạn gửi)
        if isinstance(self.logger, pytorch_lightning.loggers.WandbLogger):
            # Tạo data cho bảng
            columns = ["Class", "Precision", "Recall", "F1-score"]
            data = []
            
            # Dòng cho từng class
            for i, class_name in enumerate(self.class_names):
                data.append([
                    class_name, 
                    round(precisions[i] * 100, 2), 
                    round(recalls[i] * 100, 2), 
                    round(f1s[i] * 100, 2)
                ])
            
            # --- FIX LỖI KEY ERROR Ở ĐÂY ---
            # Dùng .get() để lấy giá trị dù key có prefix 'val_' hay không
            # Ưu tiên lấy 'val_precision', nếu không có thì lấy 'precision'
            
            p_val = output_metrics.get('val_precision', output_metrics.get('precision'))
            r_val = output_metrics.get('val_recall', output_metrics.get('recall'))
            f1_val = output_metrics.get('val_f1', output_metrics.get('f1'))

            # Chuyển sang float và nhân 100 (xử lý trường hợp tensor 0-dim)
            avg_prec = p_val.item() * 100 if p_val is not None else 0.0
            avg_rec = r_val.item() * 100 if r_val is not None else 0.0
            avg_f1 = f1_val.item() * 100 if f1_val is not None else 0.0
            
            # Bây giờ toàn bộ cột đều là số (Number), WandB sẽ không báo lỗi nữa
            data.append(["Average", round(avg_prec, 2), round(avg_rec, 2), round(avg_f1, 2)])
            
            # Đẩy lên WandB
            self.logger.experiment.log({
                f"classification_report_epoch_{self.current_epoch}": wandb.Table(data=data, columns=columns)
            })