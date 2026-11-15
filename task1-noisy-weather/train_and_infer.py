"""5-Fold EfficientNet-V2-S training and TTA inference for 'Noisy Weather Outside'."""

import argparse
import copy
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


LABEL_TO_FOLDER = {0: "rain", 1: "fog", 2: "snow"}


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class WeatherDataset(Dataset):
    """Dataset for noisy weather images organized into class subfolders for train."""

    def __init__(self, df: pd.DataFrame, img_dir: Path, transform=None, is_test: bool = False, image_size: int = 224):
        self.data = df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.is_test = is_test
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        img_name = str(self.data.loc[idx, "id"])
        if self.is_test:
            img_path = self.img_dir / img_name
            label = -1
        else:
            label = int(self.data.loc[idx, "label"])
            img_path = self.img_dir / LABEL_TO_FOLDER[label] / img_name

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (self.image_size, self.image_size))

        if self.transform is not None:
            image = self.transform(image)

        return image, label, img_name


def get_transforms(image_size: int):
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.TrivialAugmentWide(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.2)),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])
    tta_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        normalize,
    ])
    return train_tf, val_tf, tta_tf


def build_model(device: torch.device) -> nn.Module:
    model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 3)
    return model.to(device)


def train_fold(
    fold: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    df: pd.DataFrame,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[float, Path]:
    print(f"\n{'=' * 40}\nFold {fold + 1}/{args.n_splits}\n{'=' * 40}")
    train_tf, val_tf, _ = get_transforms(args.image_size)

    train_ds = WeatherDataset(df.iloc[train_idx], args.train_dir, transform=train_tf, image_size=args.image_size)
    val_ds = WeatherDataset(df.iloc[val_idx], args.train_dir, transform=val_tf, image_size=args.image_size)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_model(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_f1 = 0.0
    best_weights = copy.deepcopy(model.state_dict())

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels, _ in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_ds)

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels, _ in val_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        val_f1 = f1_score(all_labels, all_preds, average="macro")
        scheduler.step()
        print(f"Epoch {epoch + 1:02d}/{args.epochs} | Train Loss: {epoch_loss:.4f} | Val Macro F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_weights = copy.deepcopy(model.state_dict())

    ckpt_path = args.output_dir / f"best_model_fold_{fold}.pth"
    torch.save(best_weights, ckpt_path)
    print(f"--> Fold {fold + 1} Best Val Macro F1: {best_f1:.4f} (saved to {ckpt_path})")
    return best_f1, ckpt_path


def generate_ensemble_submission(model_paths: list[Path], args: argparse.Namespace, device: torch.device) -> None:
    print("\nRunning 5-Fold + 2-View TTA Ensemble Inference...")
    test_df = pd.read_csv(args.test_csv)
    _, val_tf, tta_tf = get_transforms(args.image_size)

    loader_normal = DataLoader(
        WeatherDataset(test_df, args.test_dir, transform=val_tf, is_test=True, image_size=args.image_size),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    loader_tta = DataLoader(
        WeatherDataset(test_df, args.test_dir, transform=tta_tf, is_test=True, image_size=args.image_size),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    ensemble_probs = torch.zeros((len(test_df), 3), device=device)
    softmax = nn.Softmax(dim=1)

    for fold, ckpt_path in enumerate(model_paths):
        print(f"Loading checkpoint fold {fold + 1}: {ckpt_path}")
        model = build_model(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        model.eval()

        with torch.no_grad():
            for loader in (loader_normal, loader_tta):
                start_idx = 0
                for inputs, _, _ in loader:
                    inputs = inputs.to(device)
                    probs = softmax(model(inputs))
                    end_idx = start_idx + inputs.size(0)
                    ensemble_probs[start_idx:end_idx] += probs
                    start_idx = end_idx

    ensemble_probs /= len(model_paths) * 2
    final_preds = torch.argmax(ensemble_probs, dim=1).cpu().numpy()

    sub_df = pd.DataFrame({"id": test_df["id"], "label": final_preds})
    out_csv = args.output_dir / "submission.csv"
    sub_df.to_csv(out_csv, index=False)
    print(f"Saved ensemble submission to {out_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 1: Noisy Weather Outside (5-Fold EfficientNet-V2-S)")
    parser.add_argument("--train-csv", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--test-csv", type=Path, default=Path("data/test.csv"))
    parser.add_argument("--train-dir", type=Path, default=Path("data/train"))
    parser.add_argument("--test-dir", type=Path, default=Path("data/test"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-train", action="store_true", help="Skip training and run inference from saved folds")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.skip_train:
        model_paths = [args.output_dir / f"best_model_fold_{f}.pth" for f in range(args.n_splits)]
    else:
        train_df = pd.read_csv(args.train_csv)
        skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
        fold_scores, model_paths = [], []
        for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df["label"])):
            best_f1, ckpt_path = train_fold(fold, train_idx, val_idx, train_df, args, device)
            fold_scores.append(best_f1)
            model_paths.append(ckpt_path)
        print(f"\n5-Fold CV Average Macro F1: {np.mean(fold_scores):.4f}")

    generate_ensemble_submission(model_paths, args, device)


if __name__ == "__main__":
    main()
