# rebuild_labels.py
import os
import pandas as pd
import random

# Parameters
image_root = "images"   # adjust if your folder has a different name
output_csv = "labels_fixed.csv"

# Collect all image file paths
valid_exts = [".png", ".jpg", ".jpeg"]
all_files = []

for root, _, files in os.walk(image_root):
    for f in files:
        if os.path.splitext(f)[1].lower() in valid_exts:
            all_files.append(os.path.join(root, f))

print(f"✅ Found {len(all_files)} image files under '{image_root}'")

# Assign random splits
splits = ["train", "val", "test"]
data = []
for f in all_files:
    split = random.choices(splits, weights=[0.7, 0.2, 0.1])[0]
    # Label: take parent folder name as class
    label = os.path.basename(os.path.dirname(f))
    data.append([f, label, split])

# Save to CSV
df = pd.DataFrame(data, columns=["filepath", "label", "split"])
df.to_csv(output_csv, index=False)
print(f"✅ CSV written: {output_csv}")
print(df.head())
