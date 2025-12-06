from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from .models.archs import get_backbone


@hydra.main(config_path="../configs", config_name="export/default.yaml")
def main(cfg: DictConfig):
    ckpt_path = Path(cfg.export.ckpt_path)
    onnx_path = Path(cfg.export.onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    if not ckpt_path.exists():
        print("Чекпойнт не найден:", ckpt_path)
        return

    # Загружаем модель
    sd = torch.load(ckpt_path, map_location="cpu")
    model = get_backbone(cfg.model.backbone, pretrained=False, num_classes=cfg.model.num_classes)
    # пытаемся извлечь state_dict
    if "state_dict" in sd:
        model.load_state_dict(
            {k.replace("model.", ""): v for k, v in sd["state_dict"].items() if k.startswith("model.") or True}
        )
    else:
        model.load_state_dict(sd)
    model.eval()

    dummy = torch.randn(1, 3, cfg.dataset.img_size, cfg.dataset.img_size)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=12,
    )
    print("Saved ONNX to", onnx_path)


if __name__ == "__main__":
    main()
