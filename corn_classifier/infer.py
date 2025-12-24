from pathlib import Path

import hydra
import numpy as np
import onnxruntime as ort
from omegaconf import DictConfig
from PIL import Image

CLASS_NAMES = [
    "blight",
    "common_rust",
    "gray_leaf_spot",
    "healthy",
]


@hydra.main(config_path="../configs/infer", config_name="default.yaml")
def main(cfg: DictConfig):
    onnx_path = Path(cfg.infer.onnx_path)
    if not onnx_path.exists():
        print("ONNX модель не найдена:", onnx_path)
        return

    input_path = Path(cfg.infer.input)
    if not input_path.exists():
        print("Укажите файл через infer.input=path/to/img.jpg")
        return

    img = Image.open(input_path).convert("RGB").resize((cfg.dataset.img_size, cfg.dataset.img_size))

    arr = np.array(img).astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)
    arr = np.expand_dims(arr, 0)

    session = ort.InferenceSession(str(onnx_path))
    input_name = session.get_inputs()[0].name
    out = session.run(None, {input_name: arr})

    pred_idx = int(np.argmax(out[0], axis=1)[0])

    class_names = getattr(cfg.dataset, "class_names", CLASS_NAMES)
    pred_name = class_names[pred_idx]

    print(f"Predicted class: {pred_idx} ({pred_name})")


if __name__ == "__main__":
    main()
