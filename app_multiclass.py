import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image

# --------------------------
# Flask Config
# --------------------------
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# --------------------------
# Model Loading
# --------------------------
MODEL_PATH = "outputs/model_multiclass.pth"

# Define the same architecture as training
num_classes = 2  # 🔹 Change if you add more diseases later
device = torch.device("cpu")

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Class names (must match training CSV labels order)
class_names = ["NORMAL", "PNEUMONIA"]  

# --------------------------
# Image Transform
# --------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)
        confidence = torch.softmax(outputs, dim=1)[0][predicted].item()
    return class_names[predicted.item()], confidence

# --------------------------
# Routes
# --------------------------
@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            # Run prediction
            label, confidence = predict_image(filepath)
            return render_template("result.html", label=label, confidence=confidence, filename=filename)
    return render_template("index.html")

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return redirect(url_for("static", filename="uploads/" + filename))

# --------------------------
# Run Flask
# --------------------------
if __name__ == "__main__":
    app.run(debug=True, port=5001)  # use 5001 if 5000 is busy
