import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from flask import Flask, request, render_template, redirect
from werkzeug.utils import secure_filename
from PIL import Image

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

MODEL_PATH = os.environ.get("MODEL_PATH", "outputs_multi/model_multiclass.pth")

# Load checkpoint to get class names & weights
ckpt = torch.load(MODEL_PATH, map_location="cpu")
class_names = ckpt["class_names"]

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict_image(image_path):
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].tolist()
        idx = int(torch.argmax(logits, dim=1).item())
    return class_names[idx], probs[idx], list(zip(class_names, probs))

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect("/")
        f = request.files["file"]
        if f.filename == "":
            return redirect("/")
        filename = secure_filename(f.filename)
        path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        f.save(path)
        label, conf, dist = predict_image(path)
        # sort full distribution high->low
        dist.sort(key=lambda t: t[1], reverse=True)
        return render_template("result.html", label=label, confidence=conf, dist=dist, filename=filename)
    return render_template("index.html", classes=class_names)

if __name__ == "__main__":
    app.run(debug=True, port=5001)
