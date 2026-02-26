IndustrialAI Crack Classification – Submission

Candidate: Yuvraj Singh
IndustrialAI Crack Classification 

1. Overview

This submission contains a deep learning-based binary classification system for detecting cracks on stamped industrial metal surfaces.

The model is implemented using PyTorch and leverages transfer learning with EfficientNet-B0 pretrained on ImageNet.

The objective is to classify image patches into:

Defect (Crack)

No Defect (Non-Crack)

2. Contents of Submission

This zip file contains:

report.pdf
crack_detection_submission.ipynb
best_model.pth
README.txt

report.pdf → Technical report describing methodology and experiments

crack_detection_submission.ipynb → Complete reproducible training and evaluation pipeline

best_model.pth → Trained model weights

README.txt → Usage instructions

3. Environment Requirements

Python 3.10+

PyTorch

torchvision

numpy

matplotlib

scikit-learn

Recommended:

GPU support (tested on NVIDIA P100)

4. Dataset Structure
Training Dataset

The training dataset does not follow a class-folder structure.

Labels are assigned using filename numeric ranges as specified in the assignment instructions.

A custom dataset loader inside the notebook handles:

Filename parsing

Label mapping

Image loading

Transformations

Validation Dataset

Validation dataset follows this structure:

val/
    defect/
    no_defect/

It is loaded using torchvision.datasets.ImageFolder.

5. How to Run
Option 1: Run Notebook (Recommended)

Open crack_detection_submission.ipynb

Ensure dataset path is correctly set

Run all cells sequentially

Training and validation will execute automatically

Best model will be saved as best_model.pth

Option 2: Inference Using Saved Model

To load trained model weights:

import torch
from torchvision import models
import torch.nn as nn

model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)

model.load_state_dict(torch.load("best_model.pth"))
model.eval()

Preprocess input images using the same validation transformations described in the report before inference.

6. Training Configuration

Model: EfficientNet-B0 (ImageNet pretrained)

Loss Function: CrossEntropyLoss

Optimizer: Adam

Learning Rate: 1e-5

Batch Size: 32

Input Size: 224 × 224

Device: GPU (if available)

7. Reproducibility

All hyperparameters are defined inside the notebook.

The training pipeline is modular and easy to modify.

Best model checkpoint is saved automatically.

Validation accuracy is computed at the end of each epoch.

8. Notes

The model was fully fine-tuned.

Data augmentation was applied during training only.

Validation was performed without augmentation to ensure fair evaluation.


End of README
