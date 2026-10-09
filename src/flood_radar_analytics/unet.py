"""The U-Net: training, inference, persistence (everything that needs PyTorch).

A convolutional segmentation network (ResNet-34 encoder, ImageNet-pretrained,
via segmentation_models_pytorch) that sees spatial context — shorelines,
field patterns, speckle texture — not just one pixel's value. Two input
variants: radar only ('s1') and radar + optical ('s1s2').

Never import this module in a process that also uses LightGBM (see the
package docstring).
"""

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from . import config
from .data import load_chip, model_input
from .evaluation import Predictor, evaluate


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class ChipDataset(Dataset):
    """Chips as (input, label) tensors. With `augment`: random 256-px crop,
    90-degree rotation and flip (labels move with the pixels)."""

    def __init__(self, names: list[str], variant: str, augment: bool = False, crop: int = 256,
                 seed: int = config.RANDOM_SEED):
        self.items = []
        for name in names:
            chip = load_chip(name, with_s2=(variant == "s1s2"))
            self.items.append((model_input(chip, variant), chip.label.astype(np.float32)))
        self.augment, self.crop = augment, crop
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = self.items[i]
        if self.augment:
            top, left = self.rng.integers(0, config.CHIP_SIZE - self.crop + 1, size=2)
            x = x[:, top:top + self.crop, left:left + self.crop]
            y = y[top:top + self.crop, left:left + self.crop]
            k = int(self.rng.integers(4))
            x, y = np.rot90(x, k, axes=(1, 2)), np.rot90(y, k)
            if self.rng.random() < 0.5:
                x, y = x[:, :, ::-1], y[:, ::-1]
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y))


def build_unet(variant: str, pretrained: bool = True) -> nn.Module:
    import segmentation_models_pytorch as smp

    return smp.Unet(encoder_name="resnet34", encoder_weights="imagenet" if pretrained else None,
                    in_channels=len(config.INPUT_CHANNELS[variant]), classes=1)


class BceDiceLoss(nn.Module):
    """BCE + soft Dice on valid pixels: Dice keeps the 9.5% water class from
    being swamped by easy land pixels; BCE keeps early gradients well-behaved."""

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        valid = target != config.NODATA_LABEL
        if not valid.any():
            return logits.sum() * 0.0
        logits, target = logits[valid], target[valid]
        bce = nn.functional.binary_cross_entropy_with_logits(logits, target)
        prob = torch.sigmoid(logits)
        dice = 1 - (2 * (prob * target).sum() + 1) / (prob.sum() + target.sum() + 1)
        return bce + dice


def predict_probability(net: nn.Module, x: np.ndarray, dev: torch.device | None = None, tile: int = 512) -> np.ndarray:
    """Water probability for a normalised (C, H, W) input of any size, in
    overlapping tiles (reflect-padded), keeping each tile's centre."""
    dev = dev or device()
    net.eval()
    _, height, width = x.shape
    margin = 32
    step = tile - 2 * margin
    out = np.zeros((height, width), dtype=np.float32)
    padded = np.pad(x, ((0, 0), (margin, margin + tile), (margin, margin + tile)), mode="reflect")
    with torch.no_grad():
        for top in range(0, height, step):
            for left in range(0, width, step):
                window = padded[:, top:top + tile, left:left + tile]
                prob = torch.sigmoid(net(torch.from_numpy(np.ascontiguousarray(window))[None].to(dev)))[0, 0].cpu().numpy()
                h, w = min(step, height - top), min(step, width - left)
                out[top:top + h, left:left + w] = prob[margin:margin + h, margin:margin + w]
    return out


def unet_predictor(net: nn.Module, variant: str, threshold: float = 0.5) -> Predictor:
    dev = device()
    return lambda chip: predict_probability(net, model_input(chip, variant), dev) >= threshold


def train_unet(train_names: list[str], valid_names: list[str], variant: str, epochs: int = 30,
               batch_size: int = 8, learning_rate: float = 3e-4, log=print) -> tuple[nn.Module, list[dict]]:
    """Train on random 256-px crops; keep the epoch with the best validation IoU."""
    torch.manual_seed(config.RANDOM_SEED)
    dev = device()
    net = build_unet(variant).to(dev)
    loader = DataLoader(ChipDataset(train_names, variant, augment=True), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = BceDiceLoss()
    history, best_iou, best_state = [], -1.0, None
    for epoch in range(epochs):
        net.train()
        losses = []
        for x, y in loader:
            x, y = x.to(dev), y.to(dev)
            optimizer.zero_grad()
            loss = loss_fn(net(x)[:, 0], y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        schedule.step()
        valid_iou = evaluate({"U-Net": unet_predictor(net, variant)}, valid_names,
                             with_s2=(variant == "s1s2"))["U-Net"].pooled.iou
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "valid_iou": valid_iou})
        log(f"  epoch {epoch + 1:>2}/{epochs}  loss {np.mean(losses):.4f}  valid IoU {valid_iou:.4f}")
        if valid_iou > best_iou:
            best_iou = valid_iou
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    return net, history


def save_unet(net: nn.Module, variant: str) -> None:
    path = config.model_path(f"unet_{variant}.pt")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), path)


def load_unet(variant: str) -> nn.Module:
    path = config.model_path(f"unet_{variant}.pt")
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Train first: `python scripts/train.py`.")
    net = build_unet(variant, pretrained=False)
    net.load_state_dict(torch.load(path, map_location="cpu"))
    return net.to(device())
