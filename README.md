# AI Chest X-ray Diagnostics

This project is an AI-powered Medical Assistant designed to classify chest X-ray images, helping medical professionals and individuals detect signs of common respiratory diseases. It utilizes a Deep Learning model to analyze radiography patterns and provides immediate feedback alongside confidence scores.

## Features

- **Multi-Class Detection:** Detects and classifies X-rays into three categories: `NORMAL`, `PNEUMONIA`, and `TUBERCULOSIS`.
- **Confidence Scoring:** Calculates prediction probabilities to provide transparency in the AI's assessment (e.g., Pneumonia: 96%).
- **Modern User Interface:** A streamlined, responsive, and aesthetically pleasing 'Glassmorphism' web design for easy drag-and-drop interactions.
- **Medical Disclaimers:** Built-in safeguards making it clear that the tool is intended for educational and preliminary screening only.

## Architecture

The project is built on the following technologies:
- **Backend:** Python and Flask (`app.py`).
- **Deep Learning Framework:** PyTorch & Torchvision.
- **Model Architecture:** ResNet-18 (`models.resnet18`). The final fully connected layer is modified to output 3 distinct classes, and the model is fine-tuned on the provided chest X-ray datasets. Pre-trained weights (`ResNet18_Weights.DEFAULT`) are utilized for transfer learning to ensure high contextual accuracy.
- **Image Processing:** Python Imaging Library (`PIL`) and `torchvision.transforms` are used to resize input images to 224x224 and normalize them to ImageNet standards before prediction.
- **Frontend:** HTML5, CSS3, and Jinja2 templating (`templates/index.html`, `templates/result.html`).

## Dataset

The model is trained on combined datasets of chest X-rays, including but not limited to open-source Pneumonia and Tuberculosis datasets. Data preprocessing pipelines (`build_multidisease_dataset.py`, `clean_labels.py`) handle combining and preparing the data for PyTorch DataLoaders.

## Project Structure

- `app.py`: The main Flask application server that loads the model and handles inference.
- `med_classifier_multiclass.py`: The core training script containing the `ChestXrayDataset` class and the logic to train the ResNet-18 model.
- `templates/`: Directory containing the frontend HTML UI.
- `outputs_multi/`: Directory where the trained model weights (`model_multiclass.pth`) and class names are saved.
- `chest_xray/` & `cxr_multi/`: Image dataset directories.

## Setup and Installation

### Prerequisites
- Python 3.8+
- PyTorch

### Installation

1. **Clone the repository.**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: Ensure you install the appropriate version of PyTorch for your system from the [official PyTorch website](https://pytorch.org/).)*
3. **Run the Application:**
   ```bash
   python app.py
   ```
4. **Access the UI:**
   Open your browser and navigate to `http://127.0.0.1:5000/`.

## Important Medical Disclaimer ⚠️

This tool utilizes Artificial Intelligence to analyze patterns in chest radiography. It is designed for educational and preliminary screening purposes **ONLY**. It is **NOT** a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider regarding any medical condition or before making any healthcare decisions. False positives and false negatives may occur.
