import os
import pandas as pd

# Path to your images folder
images_dir = "images"  # adjust if needed

# Collect image paths
filepaths = []
labels = []

for root, dirs, files in os.walk(images_dir):
    for file in files:
        if file.endswith((".png", ".jpg", ".jpeg")):
            filepath = os.path.abspath(os.path.join(root, file))
            filepaths.append(filepath)
            
            # assume folder name = label
            label = os.path.basename(root)
            labels.append(label)

# Make DataFrame
df = pd.DataFrame({
    "filepath": filepaths,
    "label": labels
})

# Add train/val/test splits (80/10/10)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
n = len(df)
train_end = int(0.8 * n)
val_end = int(0.9 * n)

df["split"] = ["train"] * train_end + ["val"] * (val_end - train_end) + ["test"] * (n - val_end)

# Save CSV
df.to_csv("labels.csv", index=False)

print("✅ labels.csv created successfully!")
print(df.head())
