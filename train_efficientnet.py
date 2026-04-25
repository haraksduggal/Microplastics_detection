"""
EfficientNet-B0 — Microplastic Detection (Baseline 2)
======================================================
Same pipeline as ResNet50 scripts for fair comparison.
No Grad-CAM heatmaps.

Usage:
    python train_efficientnet.py

Requirements:
    pip install torch torchvision matplotlib scikit-learn seaborn tqdm
"""

import os
import copy
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import CosineAnnealingLR

from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc as sk_auc
)

# ============================================================
# CONFIG
# ============================================================
DATASET_DIR = "/Users/haraksduggal/Downloads/microplastics/dataset"
OUTPUT_DIR = "/Users/haraksduggal/Downloads/microplastics/results_efficientnet"

BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-4
WEIGHT_DECAY = 1e-4
MIXUP_ALPHA = 0.2
NUM_WORKERS = 0
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = (
    "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
print(f"Device: {DEVICE}")
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. DATA LOADING
# ============================================================
print("\n[1/5] Loading dataset...")

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(f"{DATASET_DIR}/train", transform=train_transform)
val_dataset   = datasets.ImageFolder(f"{DATASET_DIR}/val",   transform=val_test_transform)
test_dataset  = datasets.ImageFolder(f"{DATASET_DIR}/test",  transform=val_test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

class_names = train_dataset.classes
print(f"  Classes: {class_names}")
print(f"  Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")


# ============================================================
# 2. EfficientNet-B0 MODEL
# ============================================================
print("\n[2/5] Building EfficientNet-B0 model...")

class EfficientNetB0(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        effnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

        # Freeze early layers (features 0-5, keep 6-8 trainable)
        for i, block in enumerate(effnet.features):
            if i < 5:
                for param in block.parameters():
                    param.requires_grad = False

        self.features = effnet.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # Custom classifier (EfficientNet-B0 outputs 1280 channels)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


model = EfficientNetB0(num_classes=2).to(DEVICE)

total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Total params:     {total_params:,}")
print(f"  Trainable params: {trainable_params:,}")


# ============================================================
# 3. LOSS, OPTIMIZER, SCHEDULER
# ============================================================

class_counts = [0, 0]
for _, label in train_dataset.samples:
    class_counts[label] += 1
weights = torch.tensor([max(class_counts) / c for c in class_counts], dtype=torch.float32).to(DEVICE)
print(f"  Class counts: {dict(zip(class_names, class_counts))}")
print(f"  Loss weights: {weights.tolist()}")

criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                       lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)


# ============================================================
# 4. MIXUP
# ============================================================

def mixup_data(x, y, alpha=0.2):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0)).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    return mixed_x, y, y[idx], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ============================================================
# 5. TRAINING LOOP
# ============================================================
print(f"\n[3/5] Training for {EPOCHS} epochs...")

history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
best_val_acc = 0.0
best_model_wts = None

for epoch in range(EPOCHS):
    # --- Train ---
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
    for images, labels in pbar:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        mixed_images, targets_a, targets_b, lam = mixup_data(images, labels, MIXUP_ALPHA)

        optimizer.zero_grad()
        outputs = model(mixed_images)
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (lam * preds.eq(targets_a).sum().item()
                    + (1 - lam) * preds.eq(targets_b).sum().item())
        total += labels.size(0)

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    train_loss = running_loss / total
    train_acc = correct / total

    # --- Validate ---
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            val_correct += preds.eq(labels).sum().item()
            val_total += labels.size(0)

    val_loss = val_loss / val_total
    val_acc = val_correct / val_total

    scheduler.step()

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)

    print(f"  Epoch {epoch+1:2d}/{EPOCHS}  |  "
          f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}  |  "
          f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}  "
          f"{'*** BEST ***' if val_acc > best_val_acc else ''}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_wts = copy.deepcopy(model.state_dict())

# Load best weights
model.load_state_dict(best_model_wts)
torch.save(best_model_wts, f"{OUTPUT_DIR}/efficientnet_b0_best.pth")
print(f"\n  Best val accuracy: {best_val_acc:.4f}")
print(f"  Model saved to {OUTPUT_DIR}/efficientnet_b0_best.pth")


# ============================================================
# 6. PLOT TRAINING CURVES
# ============================================================
print("\n[4/5] Saving training curves...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(history["train_loss"], label="Train Loss", linewidth=2)
ax1.plot(history["val_loss"],   label="Val Loss",   linewidth=2)
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
ax1.set_title("Training & Validation Loss — EfficientNet-B0"); ax1.legend(); ax1.grid(True, alpha=0.3)

ax2.plot(history["train_acc"], label="Train Acc", linewidth=2)
ax2.plot(history["val_acc"],   label="Val Acc",   linewidth=2)
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
ax2.set_title("Training & Validation Accuracy — EfficientNet-B0"); ax2.legend(); ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/training_curves_efficientnet.png", dpi=300, bbox_inches="tight")
plt.close()
print("  Saved training_curves_efficientnet.png")


# ============================================================
# 7. TEST SET EVALUATION
# ============================================================
print("\n[5/5] Evaluating on test set...")

model.eval()
all_preds, all_labels, all_probs = [], [], []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs  = np.array(all_probs)

test_acc = np.mean(all_preds == all_labels)
report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
print(f"\n  Test Accuracy: {test_acc:.4f}\n")
print(report)

with open(f"{OUTPUT_DIR}/classification_report_efficientnet.txt", "w") as f:
    f.write(f"EfficientNet-B0 — Microplastic Detection\n")
    f.write(f"=========================================\n\n")
    f.write(f"Test Accuracy: {test_acc:.4f}\n\n")
    f.write(report)
print("  Saved classification_report_efficientnet.txt")

# --- Confusion Matrix ---
cm = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges",
            xticklabels=class_names, yticklabels=class_names, ax=ax,
            annot_kws={"size": 16})
ax.set_xlabel("Predicted", fontsize=13)
ax.set_ylabel("Actual", fontsize=13)
ax.set_title("Confusion Matrix — EfficientNet-B0", fontsize=14)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/confusion_matrix_efficientnet.png", dpi=300, bbox_inches="tight")
plt.close()
print("  Saved confusion_matrix_efficientnet.png")

# --- ROC Curve ---
fpr, tpr, _ = roc_curve(all_labels, all_probs)
roc_auc = sk_auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
ax.set_xlabel("False Positive Rate", fontsize=13)
ax.set_ylabel("True Positive Rate", fontsize=13)
ax.set_title("ROC Curve — EfficientNet-B0", fontsize=14)
ax.legend(fontsize=12); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/roc_curve_efficientnet.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"  Saved roc_curve_efficientnet.png (AUC = {roc_auc:.4f})")


# ============================================================
# 8. FINAL SUMMARY
# ============================================================
print(f"\n{'='*55}")
print(f"  TRAINING COMPLETE — EfficientNet-B0")
print(f"{'='*55}")
print(f"  Model:         EfficientNet-B0")
print(f"  Best Val Acc:  {best_val_acc:.4f}")
print(f"  Test Acc:      {test_acc:.4f}")
print(f"  AUC-ROC:       {roc_auc:.4f}")
print(f"{'='*55}")
print(f"  Files saved to: {OUTPUT_DIR}/")
print(f"    - efficientnet_b0_best.pth")
print(f"    - training_curves_efficientnet.png")
print(f"    - confusion_matrix_efficientnet.png")
print(f"    - roc_curve_efficientnet.png")
print(f"    - classification_report_efficientnet.txt")
print(f"{'='*55}")

# --- Quick Comparison if other results exist ---
print(f"\n  MODEL COMPARISON TABLE (for your paper):")
print(f"  {'Model':<25s} {'Test Acc':>10s} {'AUC':>8s}")
print(f"  {'-'*45}")
print(f"  {'EfficientNet-B0':<25s} {test_acc:>10.4f} {roc_auc:>8.4f}")

cbam_report = Path("/Users/haraksduggal/Downloads/microplastics/results/classification_report.txt")
vanilla_report = Path("/Users/haraksduggal/Downloads/microplastics/results_vanilla/classification_report_vanilla.txt")

for path, name in [(cbam_report, "ResNet50 + CBAM"), (vanilla_report, "ResNet50 (vanilla)")]:
    if path.exists():
        with open(path) as f:
            for line in f:
                if "Test Accuracy" in line:
                    acc = float(line.strip().split(":")[1].strip())
                    print(f"  {name:<25s} {acc:>10.4f}     —")

print(f"  {'-'*45}")
print(f"\n  All three models trained! Paper comparison ready.")
