# make_labels_multiclass.py
import os
import argparse
import pandas as pd

def make_csv(data_dir, output_csv):
    rows = []
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            continue
        for label in os.listdir(split_dir):
            label_dir = os.path.join(split_dir, label)
            if not os.path.isdir(label_dir):
                continue
            for fname in os.listdir(label_dir):
                if fname.lower().endswith((".jpeg", ".jpg", ".png")):
                    rows.append({
                        "filepath": os.path.join(label_dir, fname),
                        "label": label,
                        "split": split
                    })
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"✅ {output_csv} created with {len(df)} rows")
    print(df["label"].value_counts())
    print(df.head())

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="cxr_multi")
    ap.add_argument("--output_csv", default="labels_multiclass.csv")
    args = ap.parse_args()
    make_csv(args.data_dir, args.output_csv)
