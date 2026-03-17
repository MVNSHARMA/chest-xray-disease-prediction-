import pandas as pd
import os

# Input CSV
input_csv = "labels.csv"
output_csv = "labels_clean.csv"

# Load CSV
df = pd.read_csv(input_csv)

# Ensure there's a filepath column
if "filepath" not in df.columns:
    raise ValueError("❌ CSV must have a 'filepath' column.")

# Keep only rows where the image exists
df["exists"] = df["filepath"].apply(lambda x: os.path.exists(x))
df_clean = df[df["exists"]].drop(columns=["exists"])

print(f"✅ Original rows: {len(df)}")
print(f"✅ Valid rows: {len(df_clean)}")
print(f"⚠️ Removed {len(df) - len(df_clean)} missing images")

# Save cleaned CSV
df_clean.to_csv(output_csv, index=False)
print(f"✅ Clean CSV saved as: {output_csv}")
