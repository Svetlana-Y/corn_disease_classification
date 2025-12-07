from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torch
from torchvision import transforms


class CornDataset(Dataset):
    def __init__(self, split: str = "train", cfg=None):
        root = Path(cfg.dataset.data_root) if cfg is not None else Path("data/raw")
        self.img_size = cfg.dataset.img_size if cfg is not None else 224
        folder = root / split
        if not folder.exists():
            folder = root
        self.samples = list(folder.rglob("*.jpg")) + list(folder.rglob("*.png"))
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            # нормализация простая
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return max(1, len(self.samples))

    def __getitem__(self, idx):
        path = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        label = torch.tensor(0, dtype=torch.long)
        return img, label