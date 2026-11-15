"""Multi-view 2D CNN training and inference for '3D Mesh Quality Control'.

Supports both key competition presets:
- v5:  tf_efficientnetv2_s.in21k_ft_in1k @ 512x768, pos_weight<=25.0 (scored 13.711/20.0)
- v11: tf_efficientnetv2_m.in21k_ft_in1k @ 512x768, pos_weight<=5.0, label smoothing 0.05, AMP + grad accum
"""

import argparse
from contextlib import nullcontext
import os
from pathlib import Path
import random

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import f1_score
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm


DEFECTS = [
    "abstract",
    "artifacts",
    "intersection",
    "lowpoly",
    "noisy",
    "open",
    "partial",
    "scale",
    "set",
    "simple",
]
TARGETS = DEFECTS + ["quality"]

PRESETS = {
    "v5": {
        "model_name": "tf_efficientnetv2_s.in21k_ft_in1k",
        "batch_size": 8,
        "grad_accum": 1,
        "pos_weight_cap": 25.0,
        "label_smoothing": 0.0,
        "drop_rate": 0.0,
        "amp": False,
    },
    "v11": {
        "model_name": "tf_efficientnetv2_m.in21k_ft_in1k",
        "batch_size": 4,
        "grad_accum": 2,
        "pos_weight_cap": 5.0,
        "label_smoothing": 0.05,
        "drop_rate": 0.3,
        "amp": True,
    },
}


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


class MeshDataset(Dataset):
    """Loads 1536x1024 stitched 6-view renders for multi-label defect classification."""

    def __init__(
        self,
        df: pd.DataFrame,
        img_dir: Path,
        transform=None,
        is_test: bool = False,
        label_smoothing: float = 0.0,
    ):
        self.df = df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.is_test = is_test
        self.label_smoothing = label_smoothing

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        item_id = str(row["item_id"])
        img_path = self.img_dir / f"{item_id}.png"

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (1536, 1024))

        if self.transform is not None:
            image = self.transform(image)

        if self.is_test:
            return image, item_id

        labels = row[TARGETS].values.astype(np.float32)
        if self.label_smoothing > 0.0:
            labels = labels * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        return image, torch.tensor(labels)


class MeshQualityModel(nn.Module):
    def __init__(self, model_name: str, pretrained: bool = True, drop_rate: float = 0.0):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=len(TARGETS),
            drop_rate=drop_rate,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def calculate_competition_metric(y_true: np.ndarray, y_pred_probs: np.ndarray, thresholds: np.ndarray) -> float:
    y_pred = (y_pred_probs > thresholds).astype(int)
    f1_quality = f1_score(y_true[:, 10], y_pred[:, 10], zero_division=0)
    f1_defects = f1_score(y_true[:, :10], y_pred[:, :10], average="weighted", zero_division=0)
    return float(10.0 * f1_quality + 10.0 * f1_defects)


def optimize_thresholds(y_true: np.ndarray, y_pred_probs: np.ndarray) -> np.ndarray:
    best_thresholds = np.full(len(TARGETS), 0.5)
    for i in range(len(TARGETS)):
        best_thresh, best_f1 = 0.5, 0.0
        for thresh in np.arange(0.1, 0.9, 0.05):
            preds = (y_pred_probs[:, i] > thresh).astype(int)
            score = f1_score(y_true[:, i], preds, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_thresh = float(thresh)
        best_thresholds[i] = best_thresh
    return best_thresholds


def get_transforms(img_height: int, img_width: int):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_tf = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        normalize,
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
        normalize,
    ])
    return train_tf, val_tf


def train_model(args: argparse.Namespace, cfg: dict, device: torch.device) -> tuple[nn.Module, Path, Path]:
    train_df = pd.read_csv(args.train_csv)
    shuffled = train_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    split_idx = int(0.8 * len(shuffled))
    train_split = shuffled.iloc[:split_idx]
    val_split = shuffled.iloc[split_idx:]

    train_tf, val_tf = get_transforms(args.img_height, args.img_width)
    train_ds = MeshDataset(train_split, args.train_dir, train_tf, label_smoothing=cfg["label_smoothing"])
    val_ds = MeshDataset(val_split, args.train_dir, val_tf, label_smoothing=cfg["label_smoothing"])

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=args.num_workers)

    pos_counts = train_df[TARGETS].sum()
    neg_counts = len(train_df) - pos_counts
    pos_weights = np.clip((neg_counts / pos_counts).values, 1.0, cfg["pos_weight_cap"])
    pos_weights_tensor = torch.tensor(pos_weights, dtype=torch.float32, device=device)
    print("Calibrated pos_weights:", np.round(pos_weights, 2))

    model = MeshQualityModel(cfg["model_name"], pretrained=True, drop_rate=cfg["drop_rate"]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    use_amp = cfg["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    ckpt_path = args.output_dir / f"best_model_{args.preset}.pth"
    thresh_path = args.output_dir / f"best_thresholds_{args.preset}.npy"
    best_metric = 0.0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        for step, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Train]")):
            images, labels = images.to(device), labels.to(device)
            amp_ctx = torch.amp.autocast("cuda") if use_amp else nullcontext()
            with amp_ctx:
                outputs = model(images)
                loss = criterion(outputs, labels) / cfg["grad_accum"]

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % cfg["grad_accum"] == 0 or (step + 1) == len(train_loader):
                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item() * cfg["grad_accum"]

        model.eval()
        val_loss = 0.0
        all_probs, all_labels = [], []
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Val]"):
                images, labels = images.to(device), labels.to(device)
                amp_ctx = torch.amp.autocast("cuda") if use_amp else nullcontext()
                with amp_ctx:
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                val_loss += loss.item()
                all_probs.append(torch.sigmoid(outputs).cpu().numpy())
                all_labels.append((labels > 0.5).float().cpu().numpy())

        scheduler.step()
        probs_np = np.vstack(all_probs)
        labels_np = np.vstack(all_labels)
        thresholds = optimize_thresholds(labels_np, probs_np)
        val_metric = calculate_competition_metric(labels_np, probs_np, thresholds)

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Train Loss: {train_loss / len(train_loader):.4f} | "
            f"Val Loss: {val_loss / max(1, len(val_loader)):.4f} | "
            f"Val Metric: {val_metric:.4f}/20.0"
        )

        if val_metric > best_metric:
            best_metric = val_metric
            torch.save(model.state_dict(), ckpt_path)
            np.save(thresh_path, thresholds)
            print(f"Saved new best checkpoint ({val_metric:.4f}) to {ckpt_path}")

    return model, ckpt_path, thresh_path


def run_inference(args: argparse.Namespace, cfg: dict, device: torch.device, ckpt_path: Path, thresh_path: Path) -> None:
    model = MeshQualityModel(cfg["model_name"], pretrained=False, drop_rate=cfg["drop_rate"]).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    thresholds = np.load(thresh_path)
    print("Loaded optimized thresholds:", np.round(thresholds, 2))

    test_df = pd.read_csv(args.test_csv)
    _, val_tf = get_transforms(args.img_height, args.img_width)
    test_ds = MeshDataset(test_df, args.test_dir, val_tf, is_test=True)
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=args.num_workers)

    use_amp = cfg["amp"] and device.type == "cuda"
    test_preds, item_ids = [], []

    with torch.no_grad():
        for images, ids in tqdm(test_loader, desc="Predicting Test (2-View H-Flip TTA)"):
            images = images.to(device)
            amp_ctx = torch.amp.autocast("cuda") if use_amp else nullcontext()
            with amp_ctx:
                probs_normal = torch.sigmoid(model(images))
                probs_flipped = torch.sigmoid(model(torch.flip(images, dims=[-1])))
            avg_probs = ((probs_normal + probs_flipped) / 2.0).cpu().numpy()
            test_preds.extend((avg_probs > thresholds).astype(int))
            item_ids.extend(ids)

    sub_df = pd.DataFrame(test_preds, columns=TARGETS)
    sub_df.insert(0, "item_id", item_ids)

    # Deterministic quality rule: quality=1 iff zero defects are predicted
    sub_df["quality"] = (sub_df[DEFECTS].sum(axis=1) == 0).astype(int)
    sub_df = sub_df[["item_id"] + TARGETS]

    out_csv = args.output_dir / f"submission_{args.preset}.csv"
    sub_df.to_csv(out_csv, index=False)
    print(f"Saved submission to {out_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 2: 3D Mesh Quality Control")
    parser.add_argument("--preset", choices=["v5", "v11"], default="v5", help="Model configuration preset")
    parser.add_argument("--train-csv", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--test-csv", type=Path, default=Path("data/test.csv"))
    parser.add_argument("--train-dir", type=Path, default=Path("data/train"))
    parser.add_argument("--test-dir", type=Path, default=Path("data/test"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--img-height", type=int, default=512)
    parser.add_argument("--img-width", type=int, default=768)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-train", action="store_true", help="Skip training and run inference from saved checkpoint")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg = PRESETS[args.preset]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | Preset: {args.preset} ({cfg['model_name']})")

    if args.skip_train:
        ckpt_path = args.output_dir / f"best_model_{args.preset}.pth"
        thresh_path = args.output_dir / f"best_thresholds_{args.preset}.npy"
    else:
        _, ckpt_path, thresh_path = train_model(args, cfg, device)

    run_inference(args, cfg, device, ckpt_path, thresh_path)


if __name__ == "__main__":
    main()
