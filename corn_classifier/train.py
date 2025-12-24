import subprocess
from pathlib import Path

import mlflow
import pytorch_lightning as pl
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import MLFlowLogger
from torch.utils.data import DataLoader

from .data.dataset import CornDataset
from .data.download import dvc_pull_or_download
from .models.lit_model import CornLitModel


def get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def normalize_cfg(cfg: DictConfig) -> DictConfig:
    if "train" in cfg:
        return cfg.train
    return cfg


def load_config(repo_root: Path) -> DictConfig:
    configs_dir = repo_root / "configs"
    with initialize_config_dir(config_dir=str(configs_dir)):
        return compose(config_name="train/default.yaml")


def ensure_abs_paths(cfg: DictConfig, repo_root: Path) -> None:
    if not Path(cfg.dataset.data_root).is_absolute():
        cfg.dataset.data_root = str(repo_root / cfg.dataset.data_root)
    if not Path(cfg.output.ckpt_dir).is_absolute():
        cfg.output.ckpt_dir = str(repo_root / cfg.output.ckpt_dir)


def main():
    repo_root = Path.cwd()
    raw_cfg = load_config(repo_root)
    cfg = normalize_cfg(raw_cfg)

    print("Используем конфиг:\n", OmegaConf.to_yaml(cfg))

    ensure_abs_paths(cfg, repo_root)

    # ===== DATA =====
    dvc_pull_or_download()

    train_ds = CornDataset(split="train", cfg=cfg)
    val_ds = CornDataset(split="val", cfg=cfg)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.dataset.batch_size,
        shuffle=True,
        num_workers=cfg.dataset.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.dataset.batch_size,
        num_workers=cfg.dataset.num_workers,
    )

    # ===== MLFLOW =====
    mlflow.set_tracking_uri(cfg.logging.mlflow_uri)

    mlogger = MLFlowLogger(
        tracking_uri=cfg.logging.mlflow_uri,
        experiment_name=cfg.logging.experiment_name,
    )

    mlogger.log_hyperparams(
        {
            "git_commit": get_git_commit(),
            "batch_size": cfg.dataset.batch_size,
            "lr": cfg.train.lr,
            "epochs": cfg.train.max_epochs,
        }
    )

    # ===== MODEL =====
    model = CornLitModel(cfg)

    ckpt_dir = Path(cfg.output.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        accelerator = "gpu"
        devices = torch.cuda.device_count()
    else:
        accelerator = "cpu"
        devices = 1

    trainer = pl.Trainer(
        max_epochs=cfg.train.max_epochs,
        logger=mlogger,
        default_root_dir=str(ckpt_dir),
        accelerator=accelerator,
        devices=devices,
    )

    trainer.fit(model, train_loader, val_loader)

    # ===== SAVE CHECKPOINT =====
    last_ckpt = ckpt_dir / "last.ckpt"
    trainer.save_checkpoint(str(last_ckpt))

    mlogger.experiment.log_artifact(
        run_id=mlogger.run_id,
        local_path=str(last_ckpt),
    )

    print("Обучение завершено. Чекпоинт:", last_ckpt)


if __name__ == "__main__":
    main()
