import os
import argparse
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim

# ---------------- Dataset ----------------
class MedicalImageDataset(Dataset):
    def __init__(self, csv_path, split, transform=None):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df["split"] == split]
        self.transform = transform

        # Drop rows if file doesn’t exist
        self.df = self.df[self.df["filepath"].apply(os.path.exists)].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["filepath"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = 0 if row["label"] == "NORMAL" else 1
        return img, label

# ---------------- Training Loop ----------------
def train_model(model, dataloaders, criterion, optimizer, device, num_epochs):
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        for phase in ["train", "val"]:
            if phase not in dataloaders:
                continue
            model.train() if phase == "train" else model.eval()
            running_loss, running_corrects = 0, 0
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
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
    return model

# ---------------- Main ----------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    datasets = {split: MedicalImageDataset(args.csv_path, split, transform=transform)
                for split in ["train", "val"]}

    # Drop empty splits
    datasets = {k: v for k, v in datasets.items() if len(v) > 0}
    dataloaders = {split: DataLoader(datasets[split], batch_size=args.batch_size, shuffle=True)
                   for split in datasets}

    if not dataloaders:
        raise RuntimeError("❌ No data found. Check your CSV and images folder.")

    model = models.resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    train_model(model, dataloaders, criterion, optimizer, device, args.epochs)

    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.output_dir, "model.pth"))
    print("✅ Training complete. Model saved!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default="labels_chestxray.csv")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    main(args)
