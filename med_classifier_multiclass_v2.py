import os
import argparse
import pandas as pd
from PIL import Image
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

class ChestXrayDataset(Dataset):
    def __init__(self, df, class_to_idx, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["filepath"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = self.class_to_idx[row["label"]]
        return img, label

def compute_class_weights(train_df, class_names):
    counts = Counter(train_df["label"].tolist())
    total = sum(counts.values())
    weights = []
    for c in class_names:
        # inverse frequency
        w = total / (len(class_names) * counts.get(c, 1))
        weights.append(w)
    return torch.tensor(weights, dtype=torch.float32)

def train_model(model, dataloaders, criterion, optimizer, device, num_epochs, output_dir, class_names):
    best_acc = 0.0
    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = os.path.join(output_dir, "model.pth")   # 🔹 always save as model.pth

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        for phase in ["train", "val"]:
            if phase not in dataloaders:
                continue

            model.train() if phase == "train" else model.eval()
            running_loss, running_corrects, total = 0.0, 0, 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels).item()
                total += labels.size(0)

            epoch_loss = running_loss / max(total, 1)
            epoch_acc = running_corrects / max(total, 1)
            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names
                }, ckpt_path)
                print("✅ Best model saved")

    print(f"\nTraining complete. Best val Acc: {best_acc:.4f}")
    print(f"Model checkpoint: {ckpt_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_path", required=True)
    ap.add_argument("--output_dir", default="outputs_multi")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()

    df = pd.read_csv(args.csv_path)
    # class names sorted for reproducibility
    class_names = sorted(df["label"].unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    print(f"🔹 Classes ({len(class_names)}): {class_names}")

    transforms_map = {
        "train": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ]),
        "val": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ]),
    }

    datasets = {
        split: ChestXrayDataset(df[df["split"] == split], class_to_idx, transform=transforms_map[split])
        for split in ["train", "val"]
        if (df["split"] == split).any()
    }
    dataloaders = {
        split: DataLoader(datasets[split], batch_size=args.batch_size, shuffle=True, num_workers=0)
        for split in datasets.keys()
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model = model.to(device)

    # class weights for imbalance (optional but helpful)
    if "train" in datasets:
        class_weights = compute_class_weights(df[df["split"] == "train"], class_names).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_model(
        model, dataloaders, criterion, optimizer, device,
        num_epochs=args.epochs, output_dir=args.output_dir, class_names=class_names
    )

if __name__ == "__main__":
    main()
