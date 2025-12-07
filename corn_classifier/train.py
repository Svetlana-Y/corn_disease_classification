import subprocess
from pathlib import Path

import mlflow
import pytorch_lightning as pl
import torch
# import logging
# logging.getLogger("pytorch_lightning").setLevel(logging.DEBUG)

# hydra imports for programmatic compose
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


def load_config_from_repo_configs(repo_root: Path) -> DictConfig:
    """Load hydra config from <repo_root>/configs using initialize_config_dir (accepts absolute path)."""
    configs_dir = repo_root / "configs"
    if not configs_dir.exists():
        raise FileNotFoundError(f"Config directory not found: {configs_dir}")

    # initialize_config_dir accepts an absolute path; use it as context manager
    # It will not require config_path to be relative.
    with initialize_config_dir(config_dir=str(configs_dir)):
        cfg = compose(config_name="train/default.yaml")
    return cfg


def normalize_cfg(cfg: DictConfig) -> DictConfig:
    """If config has shape { train: { ... } } -> return cfg.train, else return cfg."""
    if "train" in cfg and isinstance(cfg.train, DictConfig):
        return cfg.train
    return cfg


def ensure_abs_paths(cfg: DictConfig, repo_root: Path) -> None:
    """Make data_root and ckpt_dir absolute, relative to repo_root if they are relative."""
    if hasattr(cfg, "dataset") and hasattr(cfg.dataset, "data_root"):
        data_root = Path(cfg.dataset.data_root)
        if not data_root.is_absolute():
            cfg.dataset.data_root = str(repo_root / data_root)
    if hasattr(cfg, "output") and hasattr(cfg.output, "ckpt_dir"):
        ckpt_dir = Path(cfg.output.ckpt_dir)
        if not ckpt_dir.is_absolute():
            cfg.output.ckpt_dir = str(repo_root / ckpt_dir)


def main():
    # Рабочая директория, откуда запущена команда — считаем это корнем репозитория
    repo_root = Path.cwd()

    # Загружаем конфиг
    try:
        raw_cfg = load_config_from_repo_configs(repo_root)
    except Exception as e:
        print("Ошибка при загрузке конфига hydra:", e)
        raise

    cfg = normalize_cfg(raw_cfg)
    print("Используем конфиг:")
    print(OmegaConf.to_yaml(cfg))

    # Исправляем относительные пути на абсолютные (от репо)
    ensure_abs_paths(cfg, repo_root)

    # Попытка получить данные
    try:
        dvc_pull_or_download()
    except Exception as e:
        print("Не удалось получить данные через dvc/kaggle:", e)
        raise

    # Проверяем секцию логирования
    if not (hasattr(cfg, "logging") and hasattr(cfg.logging, "mlflow_uri")):
        raise RuntimeError("Конфиг не содержит раздел logging.mlflow_uri")

    # MLflow logger
    mlflow.set_tracking_uri(cfg.logging.mlflow_uri)
    mlogger = MLFlowLogger(tracking_uri=cfg.logging.mlflow_uri, experiment_name=cfg.logging.experiment_name)

    # Datasets / Loaders
    train_ds = CornDataset(split="train", cfg=cfg)
    val_ds = CornDataset(split="val", cfg=cfg)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.dataset.batch_size,
        num_workers=cfg.dataset.num_workers,
        shuffle=True,
    )
    val_loader = DataLoader(val_ds, batch_size=cfg.dataset.batch_size, num_workers=cfg.dataset.num_workers)
    # Model
    model = CornLitModel(cfg)

    # CKPT dir
    ckpt_dir = Path(cfg.output.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    trainer = pl.Trainer(
        max_epochs=cfg.train.max_epochs,
        logger=mlogger,
        default_root_dir=str(ckpt_dir),
        devices=1 if torch.cuda.is_available() else None,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        enable_progress_bar=True,  
    )

    trainer.fit(model, train_loader, val_loader)
    last = ckpt_dir / "last.ckpt"
    trainer.save_checkpoint(str(last))

    with mlflow.start_run():
        mlflow.log_param("git_commit", get_git_commit())
        mlflow.log_artifact(str(last))
        mlflow.log_metrics({                                        
            "final_val_acc": float(trainer.callback_metrics.get("val_acc", 0)),
            "final_val_loss": float(trainer.callback_metrics.get("val_loss", 0)),
        })
    print("Train finished. checkpoint:", last)


if __name__ == "__main__":
    main()