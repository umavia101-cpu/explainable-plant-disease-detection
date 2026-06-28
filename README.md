# Explainable Plant Disease Detection

Detect plant diseases from leaf images using deep learning, with explainability.

## Aim

Classify leaf images as healthy or diseased, compare different models, and use
Grad-CAM to explain predictions.

## Planned Workflow

1. Explore the dataset.
2. Preprocess images.
3. Train and compare models (basic CNN, EfficientNet, MobileNet, ViT).
4. Add Grad-CAM explainability.
5. Build a Streamlit demo.

## Setup

TensorFlow has no wheels for Python 3.14, so the ML environment uses Python 3.12.

```bash
# Create and activate a virtual environment (Python 3.11 or 3.12)
python3.12 -m venv .venv
source .venv/bin/activate

# Install dependencies (pinned versions)
pip install -r requirements.txt
```

On Apple Silicon, `tensorflow-metal` is included so training runs on the GPU.

## How to Run

```bash
# Dataset exploration (downloads PlantVillage from TensorFlow Datasets on first run)
python data_exploration.py

# Training and demo (added later)
python train.py
streamlit run app.py
```

## Current Progress

Progress completed this week:

- Repository and beginner-friendly project structure set up
- Python 3.12 virtual environment created with pinned dependencies
- Dataset loading switched to TensorFlow Datasets (`plant_village`) — no manual download needed
- Dataset exploration script (`data_exploration.py`) written and run end-to-end

Dataset exploration findings:

- 38 classes, 54,303 images in total
- All images are 256x256 (uniform size)
- 15,083 healthy images vs 39,220 diseased images
- Classes are imbalanced (counts range from ~275 to ~1,645 images per class)
- Outputs saved in `outputs/`: class counts (CSV), class distribution chart, sample images

Next steps:

- Basic preprocessing: resize images to the model input size and normalise pixel values
- Prepare a train / validation / test split
- Handle class imbalance (e.g. class weights or augmentation)
- Build and train a basic CNN as the first baseline model
