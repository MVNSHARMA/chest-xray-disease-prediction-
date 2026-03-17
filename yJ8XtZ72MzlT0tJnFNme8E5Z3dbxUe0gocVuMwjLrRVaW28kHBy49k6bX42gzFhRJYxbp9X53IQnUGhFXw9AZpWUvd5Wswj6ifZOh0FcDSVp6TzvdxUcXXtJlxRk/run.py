import os
import shutil
import pandas as pd

# Paths
TBX11K_IMAGES = "TBX11K/images"
TBX11K_LABELS = "TBX11K/labels.csv"     # TBX11K class labels
TBXATT_LABELS = "TBX11K/tbxatt.csv"     # TBX-Att attribute annotations

OUTPUT_DIR = "TBX11K_subtypes"
CSV_OUTPUT = os.path.join(OUTPUT_DIR, "tbx11k_subtypes.csv")

# --- Step 1: Load datasets ---
df_main = pd.read_csv(TBX11K_LABELS)
df_att = pd.read_csv(TBXATT_LABELS)

# Merge on image filename
df = pd.merge(df_main, df_att, on="filename")

# --- Step 2: Define mapping function ---
def map_to_subtype(row):
    if row.get("cavitation", 0) == 1:
        return "TB_CAVITARY"
    elif row.get("miliary", 0) == 1:
        return "TB_MILIARY"
    elif row.get("pleural_effusion", 0) == 1:
        return "TB_PLEURAL"
    elif row.get("nodules", 0) == 1 or row.get("infiltration", 0) == 1 or row.get("consolidation", 0) == 1:
        return "TB_PULMONARY"
    elif row["label"] in ["healthy", "sick_but_non-tb"]:
        return "NON_TB"
    else:
        return "TB_UNCERTAIN"

df["subtype_label"] = df.apply(map_to_subtype, axis=1)

# --- Step 3: Prepare folders ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
for subtype in df["subtype_label"].unique():
    os.makedirs(os.path.join(OUTPUT_DIR, subtype), exist_ok=True)

# --- Step 4: Copy images ---
for _, row in df.iterrows():
    src = os.path.join(TBX11K_IMAGES, row["filename"])
    dst = os.path.join(OUTPUT_DIR, row["subtype_label"], row["filename"])
    if os.path.exists(src):
        shutil.copy(src, dst)

# --- Step 5: Save CSV ---
df.to_csv(CSV_OUTPUT, index=False)

print(f"✅ Subtype dataset created at {OUTPUT_DIR}")
