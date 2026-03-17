import os
import json
import math
import time
import random
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision
from torchvision import transforms

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from tqdm import tqdm

# ------------------------------
# Reproducibility
# ------------------------------

def set_seed(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False  # True can slow down a lot
    torch.backends.cudnn.benchmark = True


# ------------------------------
# Dataset loaders
# ------------------------------

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif", ".webp"}


def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS


class FolderDataset(Dataset):
    def __init__(self, root: str, split: str, img_size: int = 384):
        split_dir = Path(root) / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Expected directory {split_dir} to exist for split='{split}'.")
        self.samples = []
        self.class_to_idx = {}
        for idx, cls in enumerate(sorted([d.name for d in split_dir.iterdir() if d.is_dir()])):
            self.class_to_idx[cls] = idx
            for img_path in (split_dir / cls).rglob('*'):
                if img_path.is_file() and is_image(img_path):
                    self.samples.append((str(img_path), idx))
        if not self.samples:
            raise RuntimeError(f"No images found under {split_dir}")
        self.transforms = get_transforms(split, img_size)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        img = self.transforms(img)
        return img, label


class CSVDataset(Dataset):
    def __init__(self, csv_path: str, split: str, img_size: int = 384, root: Optional[str] = None):
        df = pd.read_csv(csv_path)
        if 'filepath' not in df.columns or 'label' not in df.columns or 'split' not in df.columns:
            raise ValueError("CSV must have columns: filepath,label,split")
        self.df = df[df['split'].str.lower() == split.lower()].copy()
        if self.df.empty:
            raise RuntimeError(f"No rows with split={split} in {csv_path}")
        self.root = Path(root) if root else None
        # build classes
        classes = sorted(self.df['label'].unique().tolist())
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.df['label_idx'] = self.df['label'].map(self.class_to_idx)
        self.transforms = get_transforms(split, img_size)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        p = Path(row['filepath'])
        if self.root and not p.is_absolute():
            p = self.root / p
        img = Image.open(p).convert('RGB')
        img = self.transforms(img)
        return img, int(row['label_idx'])


# ------------------------------
# Transforms & Augmentations
# ------------------------------

def get_transforms(split: str, img_size: int):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if split.lower() == 'train':
        return transforms.Compose([
            transforms.Resize(int(img_size * 1.1)),
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            normalize,
        ])


# ------------------------------
# Model
# ------------------------------

def build_model(num_classes: int) -> nn.Module:
    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, num_classes)
    return model


# ------------------------------
# Training utilities
# ------------------------------

class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.sum = 0.0
        self.count = 0
    def update(self, val, n=1):
        self.sum += float(val) * n
        self.count += n
    @property
    def avg(self):
        return self.sum / max(1, self.count)


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int, weights=None):
    if weights is not None:
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)


def compute_class_weights(labels: List[int], num_classes: int):
    counts = np.bincount(labels, minlength=num_classes)
    inv = 1.0 / np.clip(counts, 1, None)
    weights = inv / inv.sum() * num_classes
    return weights


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    loss_meter = AverageMeter()
    correct = 0
    total = 0
    for imgs, labels in tqdm(loader, desc="train", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=True):
            logits = model(imgs)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss_meter.update(loss.item(), imgs.size(0))
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
    acc = correct / max(1, total)
    return loss_meter.avg, acc


def evaluate(model, loader, device, num_classes: int):
    model.eval()
    all_labels = []
    all_probs = []
    all_preds = []
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="eval", leave=False):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1)
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_preds.append(logits.argmax(1).cpu().numpy())
    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    try:
        # macro AUROC (one-vs-rest)
        auroc = roc_auc_score(y_true, y_prob, multi_class="ovr")
    except Exception:
        auroc = float("nan")
    cm = confusion_matrix(y_true, y_pred)
    return report, auroc, cm


def save_checkpoint(state: dict, is_best: bool, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state, output_dir / 'last.pt')
    if is_best:
        torch.save(state, output_dir / 'best.pt')


# ------------------------------
# Grad-CAM (simple)
# ------------------------------

class GradCAM:
    def __init__(self, model: nn.Module, target_layer: str = 'layer4'):
        self.model = model.eval()
        self.target_acts = None
        self.target_grads = None
        layer = dict([*model.named_modules()])[target_layer]
        layer.register_forward_hook(self.fwd_hook)
        layer.register_full_backward_hook(self.bwd_hook)

    def fwd_hook(self, m, i, o):
        self.target_acts = o.detach()
    def bwd_hook(self, m, gi, go):
        self.target_grads = go[0].detach()

    def __call__(self, img_tensor: torch.Tensor, class_idx: Optional[int] = None):
        # img_tensor: (1,3,H,W) normalized
        logits = self.model(img_tensor)
        if class_idx is None:
            class_idx = logits.argmax(1).item()
        score = logits[0, class_idx]
        self.model.zero_grad(set_to_none=True)
        score.backward(retain_graph=True)
        weights = self.target_grads.mean(dim=(2,3), keepdim=True)  # GAP over H,W
        cam = (weights * self.target_acts).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        cam -= cam.min()
        cam /= (cam.max() + 1e-6)
        return cam, class_idx


def overlay_cam_on_image(img: Image.Image, cam: np.ndarray, alpha: float = 0.35) -> Image.Image:
    import cv2
    img_np = np.array(img.convert('RGB'))
    cam_resized = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))
    heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (heatmap * alpha + img_np * (1 - alpha)).astype(np.uint8)
    return Image.fromarray(overlay)


# ------------------------------
# Prediction API
# ------------------------------

class Predictor:
    def __init__(self, weights_path: str, class_names: List[str], device: Optional[str] = None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.model = build_model(num_classes=len(class_names))
        state = torch.load(weights_path, map_location=self.device)
        if 'model' in state:
            self.model.load_state_dict(state['model'])
        else:
            self.model.load_state_dict(state)
        self.model.to(self.device).eval()
        self.class_names = class_names
        self.tf = get_transforms('val', img_size=384)

    @torch.inference_mode()
    def predict_image(self, image_path: str):
        img = Image.open(image_path).convert('RGB')
        t = self.tf(img).unsqueeze(0).to(self.device)
        logits = self.model(t)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        idx = int(np.argmax(probs))
        return {
            'pred_class': self.class_names[idx],
            'pred_index': idx,
            'probs': {c: float(p) for c, p in zip(self.class_names, probs)}
        }

    def gradcam_image(self, image_path: str, save_path: Optional[str] = None):
        img = Image.open(image_path).convert('RGB')
        t = self.tf(img).unsqueeze(0).to(self.device)
        cam = GradCAM(self.model)  # default target layer: layer4
        cam_map, idx = cam(t)
        blended = overlay_cam_on_image(img, cam_map, alpha=0.35)
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            blended.save(save_path)
        return blended, idx


# ------------------------------
# Orchestration (train / eval / predict)
# ------------------------------

def load_splits(args):
    if args.mode == 'folders':
        train_ds = FolderDataset(args.data_root, 'train', args.img_size)
        val_ds = FolderDataset(args.data_root, 'val', args.img_size)
        test_ds = None
        try:
            test_ds = FolderDataset(args.data_root, 'test', args.img_size)
        except Exception:
            pass
        class_to_idx = train_ds.class_to_idx
    elif args.mode == 'csv':
        train_ds = CSVDataset(args.csv_path, 'train', args.img_size, args.data_root)
        val_ds = CSVDataset(args.csv_path, 'val', args.img_size, args.data_root)
        test_ds = None
        try:
            test_ds = CSVDataset(args.csv_path, 'test', args.img_size, args.data_root)
        except Exception:
            pass
        class_to_idx = train_ds.class_to_idx
    else:
        raise ValueError("mode must be 'folders' or 'csv'")
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    return train_ds, val_ds, test_ds, class_to_idx, idx_to_class


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds, val_ds, test_ds, class_to_idx, idx_to_class = load_splits(args)
    num_classes = len(class_to_idx)

    # class weights / sampler for imbalance
    train_labels = [y for _, y in getattr(train_ds, 'samples', [])] if isinstance(train_ds, FolderDataset) else train_ds.df['label_idx'].tolist()
    class_weights = compute_class_weights(train_labels, num_classes)
    sample_weights = np.array([class_weights[y] for y in train_labels], dtype=np.float32)

    train_loader = make_loader(train_ds, args.batch_size, shuffle=False, num_workers=args.workers, weights=sample_weights)
    val_loader = make_loader(val_ds, args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader = make_loader(test_ds, args.batch_size, shuffle=False, num_workers=args.workers) if test_ds else None

    model = build_model(num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_val = -1.0
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'meta.json').write_text(json.dumps({
        'class_to_idx': class_to_idx,
        'idx_to_class': idx_to_class,
        'img_size': args.img_size
    }, indent=2))

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        print(f"train loss: {tr_loss:.4f} | acc: {tr_acc:.4f}")

        rep, auroc, cm = evaluate(model, val_loader, device, num_classes)
        val_acc = rep['accuracy'] if 'accuracy' in rep else np.mean([rep[str(i)]['recall'] for i in range(num_classes)])
        print(f"val acc: {val_acc:.4f} | AUROC (macro): {auroc:.4f}")

        is_best = val_acc > best_val
        best_val = max(best_val, val_acc)

        save_checkpoint({'model': model.state_dict(), 'epoch': epoch, 'val_acc': val_acc}, is_best, output_dir)
        scheduler.step()

        # early stopping patience
        if args.patience > 0 and (epoch - np.argmax([best_val])) > args.patience:
            print("Early stopping triggered.")
            break

    # final evaluation
    best_ckpt = output_dir / 'best.pt'
    state = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(state['model'] if 'model' in state else state)

    print("\nValidation metrics (best checkpoint):")
    rep, auroc, cm = evaluate(model, val_loader, device, num_classes)
    print(json.dumps({'accuracy': rep.get('accuracy', None), 'macro_auroc': auroc}, indent=2))

    if test_loader:
        print("\nTest metrics:")
        rep_t, auroc_t, cm_t = evaluate(model, test_loader, device, num_classes)
        print(json.dumps({'accuracy': rep_t.get('accuracy', None), 'macro_auroc': auroc_t}, indent=2))


def predict_cli(args):
    meta_path = Path(args.weights).parent / 'meta.json'
    meta = json.loads(meta_path.read_text())
    class_names = [meta['idx_to_class'][str(i)] for i in range(len(meta['idx_to_class']))]
    pred = Predictor(args.weights, class_names)
    out = pred.predict_image(args.image_path)
    print(json.dumps(out, indent=2))


def gradcam_cli(args):
    meta_path = Path(args.weights).parent / 'meta.json'
    meta = json.loads(meta_path.read_text())
    class_names = [meta['idx_to_class'][str(i)] for i in range(len(meta['idx_to_class']))]
    pred = Predictor(args.weights, class_names)
    img, idx = pred.gradcam_image(args.image_path, args.save_cam)
    print(json.dumps({'target_class': class_names[idx], 'saved_to': args.save_cam}, indent=2))


# ------------------------------
# Main (argparse)
# ------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Medical Image Disease Classifier (PyTorch)')
    parser.add_argument('--mode', choices=['folders', 'csv'], default='folders', help='Dataset mode')
    parser.add_argument('--data_root', type=str, default='./dataset', help='Root folder for images (or base for CSV filepaths)')
    parser.add_argument('--csv_path', type=str, default='./labels.csv', help='Path to labels.csv if mode=csv')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Where to save checkpoints & metadata')
    parser.add_argument('--img_size', type=int, default=384)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--patience', type=int, default=0, help='Early stopping patience (0 disables)')
    # Predict / Grad-CAM
    parser.add_argument('--predict', action='store_true')
    parser.add_argument('--gradcam', action='store_true')
    parser.add_argument('--weights', type=str, help='Path to checkpoint .pt')
    parser.add_argument('--image_path', type=str, help='Path to single image for prediction/gradcam')
    parser.add_argument('--save_cam', type=str, default=None, help='Where to save Grad-CAM overlay image')

    args = parser.parse_args()

    if args.predict:
        if not args.weights or not args.image_path:
            raise SystemExit('--predict requires --weights and --image_path')
        predict_cli(args)
    elif args.gradcam:
        if not args.weights or not args.image_path:
            raise SystemExit('--gradcam requires --weights and --image_path')
        gradcam_cli(args)
    else:
        train(args)
