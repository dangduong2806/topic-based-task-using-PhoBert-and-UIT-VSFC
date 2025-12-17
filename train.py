import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar, TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger, WandbLogger

# Import module của chúng ta
from src.datamodules.uit_vsfc_module import UITVSFCDataModule
from src.models.phobert_module import PhoBERTClassifier

@hydra.main(config_path="configs", config_name="config", version_base="1.2")
def train(cfg: DictConfig):
    # 1. Thiết lập Seed để đảm bảo tái lập kết quả (Reproducibility)
    pl.seed_everything(cfg.seed)
    
    # 2. Khởi tạo Data Module
    print(f"Loading data from: {cfg.data.data_dir}")
    dm = UITVSFCDataModule(
        data_dir=cfg.data.data_dir,
        model_name=cfg.model.model_name,
        batch_size=cfg.data.batch_size,
        max_len=cfg.data.max_len,
        num_workers=cfg.data.num_workers
    )

    # 3. Khởi tạo Model
    print(f"Initializing model: {cfg.model.model_name}")
    model = PhoBERTClassifier(
        model_name=cfg.model.model_name,
        num_labels=cfg.model.num_labels,
        learning_rate=cfg.model.learning_rate
    )


    # 4. Định nghĩa Callbacks (MLOps standard)
    
    # Lưu model có Macro F1 trên tập Val cao nhất
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.paths.ckpt_dir,
        filename="phobert-{epoch:02d}-{val_f1:.4f}",
        save_top_k=1,
        monitor="val_f1",
        mode="max"
    )

    # Dừng train nếu Val Loss không giảm sau 'patience' epochs (tránh Overfitting)
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=3,
        mode="min"
    )

    # Thanh tiến trình đẹp mắt
    # progress_bar = RichProgressBar()
    progress_bar = TQDMProgressBar(refresh_rate=20)

    # --- SỬA ĐỔI CHÍNH Ở ĐÂY: WANDB LOGGER ---
    print(f"Initializing WandB Logger for project: {cfg.logger.wandb.project}")
    wandb_logger = WandbLogger(
        project=cfg.logger.wandb.project,
        name=cfg.run_name,
        log_model=cfg.logger.wandb.log_model, # "all" hoặc True để lưu checkpoints lên cloud
        save_dir=cfg.paths.log_dir # Lưu các file meta cục bộ vào thư mục logs
    )

    # (Tùy chọn) Theo dõi Gradients và Histogram của weights
    # Giúp bạn biết model có bị kẹt gradient hay không
    wandb_logger.watch(model, log="all", log_freq=100)
    # ------------------------------------------
    
    # 6. Khởi tạo Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        callbacks=[checkpoint_callback, early_stop_callback, progress_bar],
        logger=wandb_logger,
        # Dùng precision 16 (mixed precision) giúp train nhanh hơn và nhẹ hơn trên GPU
        precision="16-mixed" if cfg.trainer.accelerator == "gpu" else 32,
        log_every_n_steps=20
    )

    # 7. Bắt đầu Training
    print(">>> Starting Training...")
    trainer.fit(model, dm)

    # 8. Tự động Test với model tốt nhất vừa train
    print(">>> Starting Testing with Best Checkpoint...")
    trainer.test(model, datamodule=dm, ckpt_path="best")
    
    print(f">>> Best model saved at: {checkpoint_callback.best_model_path}")

if __name__ == "__main__":
    train()