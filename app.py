import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

# ------------------------------
# Flask setup
# ------------------------------
app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ------------------------------
# Load Model
# ------------------------------
def load_model():
    model_path = "outputs_multi/model_multiclass.pth"
    checkpoint = torch.load(model_path, map_location="cpu")

    # ✅ Load class names from checkpoint
    if "class_names" in checkpoint:
        class_names = checkpoint["class_names"]
    else:
        class_names = ["NORMAL", "PNEUMONIA", "TUBERCULOSIS"]  # fallback

    # ✅ Define model
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model, class_names


model, class_names = load_model()

# ------------------------------
# Image Preprocessing
# ------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ------------------------------
# Routes
# ------------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "No file uploaded", 400

    file = request.files["file"]
    if file.filename == "":
        return "No selected file", 400

    filepath = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
    file.save(filepath)

    # Load and preprocess image
    img = Image.open(filepath).convert("RGB")
    img = transform(img).unsqueeze(0)

    # Model prediction
    with torch.no_grad():
        outputs = model(img)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, pred = torch.max(probabilities, 0)
        prediction = class_names[pred.item()]
        confidence_percent = round(confidence.item() * 100, 2)

    # ✅ Send result to result.html
    return render_template("result.html", filename=file.filename, prediction=prediction, confidence=confidence_percent)


# ------------------------------
# Run
# ------------------------------
if __name__ == "__main__":
    app.run(debug=True)
