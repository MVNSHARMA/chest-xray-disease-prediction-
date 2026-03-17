# build_multidisease_dataset.py
import os
import re
import argparse
import random
import shutil
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png"}

def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS

def mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def safe_copy(src: Path, dst_dir: Path, prefix: str = ""):
    mkdir(dst_dir)
    base = f"{prefix}{src.name}"
    dst = dst_dir / base
    if not dst.exists():
        shutil.copy2(src, dst)
        return dst
    # de-duplicate
    stem, ext = os.path.splitext(base)
    i = 1
    while True:
        cand = dst_dir / f"{stem}_{i}{ext}"
        if not cand.exists():
            shutil.copy2(src, cand)
            return cand
        i += 1

def gather_pneumonia(pneu_root: Path, out_root: Path):
    print(f"• Using pneumonia dataset at: {pneu_root}")
    for split in ["train", "val", "test"]:
        for lbl in ["NORMAL", "PNEUMONIA"]:
            src = pneu_root / split / lbl
            if not src.exists():
                continue
            dst = out_root / split / lbl
            for img in src.rglob("*"):
                if img.is_file() and is_image(img):
                    safe_copy(img, dst, prefix="pn_")
    print("✓ Copied NORMAL & PNEUMONIA")

def gather_tb(tb_root: Path, out_root: Path):
    print(f"• Using TB dataset at: {tb_root}")
    # Heuristics: find TB-positive folders by name
    tb_dirs = [p for p in tb_root.rglob("*") if p.is_dir() and re.search(r"(tb|tuber)", p.name, re.I)]
    if not tb_dirs:
        # fallback: if dataset uses a 'Tuberculosis' class folder exactly
        tb_dirs = [p for p in tb_root.rglob("*") if p.is_dir() and p.name.lower() in {"tuberculosis", "tb"}]

    tb_imgs_by_split = {"train": [], "val": [], "test": []}

    # Try to respect any existing splits in the TB dataset
    split_keywords = {"train": "train", "val": "val|valid|validation", "test": "test"}
    found_any_split = False
    for tb_dir in tb_dirs:
        # find split by parent path name
        parent_str = str(tb_dir.parent).lower()
        assigned = False
        for sname, pattern in split_keywords.items():
            if re.search(pattern, parent_str):
                for img in tb_dir.rglob("*"):
                    if img.is_file() and is_image(img):
                        tb_imgs_by_split[sname].append(img)
                found_any_split = True
                assigned = True
                break
        if not assigned:
            # will handle later if no splits detected
            pass

    if not found_any_split:
        # No splits? Gather all TB images, then split 80/10/10
        all_tb = []
        for tb_dir in tb_dirs:
            for img in tb_dir.rglob("*"):
                if img.is_file() and is_image(img):
                    all_tb.append(img)
        random.seed(42)
        random.shuffle(all_tb)
        n = len(all_tb)
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)
        tb_imgs_by_split["train"] = all_tb[:n_train]
        tb_imgs_by_split["val"] = all_tb[n_train:n_train+n_val]
        tb_imgs_by_split["test"] = all_tb[n_train+n_val:]
    # Copy TB images
    for split in ["train", "val", "test"]:
        dst = out_root / split / "TUBERCULOSIS"
        for img in tb_imgs_by_split[split]:
            safe_copy(img, dst, prefix="tb_")
    print("✓ Copied TUBERCULOSIS")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pneumonia_root", default="chest_xray", help="Path to Kaggle Pneumonia dataset")
    ap.add_argument("--tb_root", required=True, help="Path to TB dataset root")
    ap.add_argument("--out_dir", default="cxr_multi", help="Output unified dataset")
    args = ap.parse_args()

    pneu_root = Path(args.pneumonia_root)
    tb_root = Path(args.tb_root)
    out_root = Path(args.out_dir)

    # Prepare folder tree
    for split in ["train", "val", "test"]:
        for lbl in ["NORMAL", "PNEUMONIA", "TUBERCULOSIS"]:
            mkdir(out_root / split / lbl)

    gather_pneumonia(pneu_root, out_root)
    gather_tb(tb_root, out_root)

    # quick counts
    print("\nSummary:")
    for split in ["train", "val", "test"]:
        line = []
        for lbl in ["NORMAL", "PNEUMONIA", "TUBERCULOSIS"]:
            cnt = sum(1 for _ in (out_root / split / lbl).glob("*"))
            line.append(f"{lbl}={cnt}")
        print(f"- {split}: " + ", ".join(line))

if __name__ == "__main__":
    main()
