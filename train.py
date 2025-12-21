import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar, TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger, WandbLogger

# Import module của chúng ta
from src.datamodules.uit_vsfc_module import UITVSFCDataModule
from src.models.phobert_module import PhoBERTClassifier

class KaggleProgressBar(TQDMProgressBar):
    def __init__(self):
        super().__init__()
    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.disable = True
        return bar
    
def train_model(cfg: DictConfig):
    if "seed" in cfg:
        pl.seed_everything(cfg.seed)
    
    print(f"Loading data from: {cfg.data.data_dir}")
    dm = UITVSFCDataModule(
        data_dir=cfg.data.data_dir,
        model_name=cfg.model.model_name,
        batch_size=cfg.data.batch_size,
        max_len=cfg.data.max_len,
        num_workers=cfg.data.num_workers
    )

    print(f"Initializing model: {cfg.model.model_name}")
    # --- CẬP NHẬT: Truyền class_names từ config ---
    model = PhoBERTClassifier(
        model_name=cfg.model.model_name,
        num_labels=cfg.model.num_labels,
        learning_rate=cfg.model.learning_rate,
        weight_decay=cfg.model.weight_decay, # Mới
        dropout=cfg.model.dropout,           # Mới
        class_names=list(cfg.model.class_names) # Chuyển OmegaConf list thành python list
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.paths.ckpt_dir,
        filename="phobert-{epoch:02d}-{val_f1:.4f}",
        save_top_k=1,
        monitor="val_f1",
        mode="max"
    )

    early_stop_callback = EarlyStopping(
        monitor="val_f1",
        min_delta=0.0001,
        patience=4,
        mode="max"
    )

    progress_bar = KaggleProgressBar()

    if "wandb" in cfg.logger:
        print(f"Initializing WandB Logger: {cfg.logger.wandb.project}")
        logger = WandbLogger(
            project=cfg.logger.wandb.project,
            name=cfg.run_name,
            log_model=cfg.logger.wandb.log_model, 
            save_dir=cfg.paths.log_dir
        )
    else:
        logger = CSVLogger(save_dir=cfg.paths.log_dir)

    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        callbacks=[checkpoint_callback, early_stop_callback, progress_bar],
        logger=logger,
        precision="16-mixed" if cfg.trainer.accelerator == "gpu" else 32,
        log_every_n_steps=10
    )

    print(">>> Starting Training...")
    trainer.fit(model, dm)

    print(">>> Starting Testing with Best Checkpoint...")
    trainer.test(model, datamodule=dm, ckpt_path="best")
    
    if "wandb" in cfg.logger:
        import wandb
        wandb.finish()

@hydra.main(config_path="configs", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    train_model(cfg)

if __name__ == "__main__":
    main()