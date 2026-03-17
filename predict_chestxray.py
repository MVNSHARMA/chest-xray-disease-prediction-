import argparse
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import pandas as pd
from tqdm import tqdm

# ----------------------------
# Load Model
# ----------------------------
def load_model(model_path, num_classes):
    model = models.resnet18(weights=None)  # no pretrained weights
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model

# ----------------------------
# Prediction Function
# ----------------------------
def predict_image(model, image_path, class_names, transform):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
        predicted_class = class_names[preds.item()]

    return predicted_class

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, help="Path to a single image")
    parser.add_argument("--folder_path", type=str, help="Path to a folder of images for batch prediction")
    parser.add_argument("--model_path", type=str, default="outputs/model.pth",
                        help="Path to trained model (default: outputs/model.pth)")
    parser.add_argument("--output_csv", type=str, default="batch_predictions.csv",
                        help="CSV filename for batch results")
    args = parser.parse_args()

    # Classes in Chest X-Ray dataset
    class_names = ["NORMAL", "PNEUMONIA"]

    # Load model
    model = load_model(args.model_path, num_classes=len(class_names))

    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # ----------------------------
    # Single image mode
    # ----------------------------
    if args.image_path:
        prediction = predict_image(model, args.image_path, class_names, transform)
        print(f"✅ Prediction for {os.path.basename(args.image_path)}: {prediction}")

    # ----------------------------
    # Batch mode
    # ----------------------------
    elif args.folder_path:
        results = []
        for fname in tqdm(os.listdir(args.folder_path), desc="Predicting"):
            fpath = os.path.join(args.folder_path, fname)
            if os.path.isfile(fpath) and fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                pred = predict_image(model, fpath, class_names, transform)
                results.append({"filename": fname, "prediction": pred})

        # Save to CSV
        df = pd.DataFrame(results)
        df.to_csv(args.output_csv, index=False)
        print(f"✅ Batch predictions saved to {args.output_csv}")
    else:
        print("⚠️ Please provide either --image_path or --folder_path")
