import pytorch_lightning as pl
import timm
import torch
import torch.nn.functional as F
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)


class CornLitModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()

        self.num_classes = cfg.model.num_classes
        self.lr = cfg.train.lr
        backbone = cfg.model.backbone.lower()

        # ================= BACKBONE =================
        if backbone == "rexnet":
            self.model = timm.create_model(
                "rexnet_200",
                pretrained=True,
                num_classes=self.num_classes,
            )

        elif backbone == "vit":
            self.model = timm.create_model(
                "vit_base_patch16_224",
                pretrained=True,
                num_classes=self.num_classes,
            )

        else:
            raise ValueError(f"Unsupported backbone '{cfg.model.backbone}'. " "Use: rexnet | vit")

        # ================= METRICS =================
        self.train_acc = MulticlassAccuracy(num_classes=self.num_classes)

        self.val_acc = MulticlassAccuracy(num_classes=self.num_classes)
        self.val_precision = MulticlassPrecision(num_classes=self.num_classes, average="macro")
        self.val_recall = MulticlassRecall(num_classes=self.num_classes, average="macro")
        self.val_f1 = MulticlassF1Score(num_classes=self.num_classes, average="macro")

    # ================= FORWARD =================
    def forward(self, x):
        return self.model(x)

    # ================= TRAIN =================
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        preds = torch.argmax(logits, dim=1)
        self.train_acc(preds, y)

        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_acc, on_epoch=True, prog_bar=True)

        return loss

    # ================= VALIDATION =================
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        preds = torch.argmax(logits, dim=1)

        self.val_acc(preds, y)
        self.val_precision(preds, y)
        self.val_recall(preds, y)
        self.val_f1(preds, y)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc, on_epoch=True, prog_bar=True)
        self.log("val_precision", self.val_precision, on_epoch=True)
        self.log("val_recall", self.val_recall, on_epoch=True)
        self.log("val_f1", self.val_f1, on_epoch=True, prog_bar=True)

    # ================= OPTIMIZER =================
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
