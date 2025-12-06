from pathlib import Path

import hydra
import numpy as np
import onnxruntime as ort
from omegaconf import DictConfig
from PIL import Image


@hydra.main(config_path="../configs", config_name="infer/default.yaml")
def main(cfg: DictConfig):
    onnx_path = Path(cfg.infer.onnx_path)
    if not onnx_path.exists():
        print("ONNX модель не найдена:", onnx_path)
        return

    input_path = cfg.infer.input
    if not input_path:
        print("Укажите файл через infer.input=path/to/img.jpg")
        return

    img = Image.open(input_path).convert("RGB").resize((cfg.dataset.img_size, cfg.dataset.img_size))
    arr = np.array(img).astype(np.float32) / 255.0
    # HWC -> CHW
    arr = arr.transpose(2, 0, 1)
    arr = np.expand_dims(arr, 0)

    session = ort.InferenceSession(str(onnx_path))
    input_name = session.get_inputs()[0].name
    out = session.run(None, {input_name: arr})
    pred = np.argmax(out[0], axis=1)[0]
    print("Predicted class:", int(pred))


if __name__ == "__main__":
    main()
