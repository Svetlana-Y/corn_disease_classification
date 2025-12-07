from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from .models.lit_model import CornLitModel

@hydra.main(config_path="../configs/export", config_name="default.yaml", version_base=None)
def main(cfg: DictConfig):

    ckpt_path = Path(cfg.export.ckpt_path)
    onnx_path = Path(cfg.export.onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not ckpt_path.exists():
        print("Чекпойнт не найден:", ckpt_path)
        return
    
    checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    model_cfg_dict = checkpoint['hyper_parameters']['cfg']
    model_cfg = OmegaConf.create(model_cfg_dict)
    model_cfg.dataset.img_size = cfg.dataset.img_size
    
    lit_model = CornLitModel(model_cfg)

    state_dict = checkpoint['state_dict']
    new_state_dict = {k.replace('model.', '').replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    lit_model.load_state_dict(new_state_dict, strict=False)

    model = lit_model.model.eval().to("cpu")

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
