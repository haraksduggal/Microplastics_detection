"""
ResNet50 + CBAM — Microplastic Detection
==========================================
- ResNet50 pretrained (ImageNet) + CBAM attention
- Mixup augmentation
- Weighted cross-entropy
- Cosine annealing LR
- Grad-CAM heatmaps
- Full evaluation (confusion matrix, ROC, classification report)

Usage:
    python train_resnet50_cbam.py

Requirements:
    pip install torch torchvision matplotlib scikit-learn seaborn tqdm pillow
"""

import os
import copy
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
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
# CONFIG — UPDATE DATASET_DIR AND OUTPUT_DIR
# ============================================================
DATASET_DIR = "/Users/haraksduggal/Downloads/microplastics/dataset"
OUTPUT_DIR = "/Users/haraksduggal/Downloads/microplastics/results"

BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-4
WEIGHT_DECAY = 1e-4
MIXUP_ALPHA = 0.2
NUM_WORKERS = 0        # 0 is safest on Mac
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
print("\n[1/6] Loading dataset...")

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
# 2. CBAM MODULE
# ============================================================

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        return x * self.sigmoid(self.conv(combined))


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


# ============================================================
# 3. ResNet50 + CBAM MODEL
# ============================================================
print("\n[2/6] Building ResNet50 + CBAM model...")

class ResNet50_CBAM(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # Freeze early layers (conv1 + layer1 + layer2)
        for name, param in resnet.named_parameters():
            if "layer3" not in name and "layer4" not in name and "fc" not in name:
                param.requires_grad = False

        self.conv1   = resnet.conv1
        self.bn1     = resnet.bn1
        self.relu    = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1  = resnet.layer1
        self.layer2  = resnet.layer2
        self.layer3  = resnet.layer3
        self.layer4  = resnet.layer4
        self.avgpool = resnet.avgpool

        # CBAM after layer3 (1024 ch) and layer4 (2048 ch)
        self.cbam3 = CBAM(1024)
        self.cbam4 = CBAM(2048)

        # Custom classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.cbam3(self.layer3(x))
        x = self.cbam4(self.layer4(x))
        x = torch.flatten(self.avgpool(x), 1)
        x = self.classifier(x)
        return x


model = ResNet50_CBAM(num_classes=2).to(DEVICE)

total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Total params:     {total_params:,}")
print(f"  Trainable params: {trainable_params:,}")


# ============================================================
# 4. LOSS, OPTIMIZER, SCHEDULER
# ============================================================

# Weighted loss (handles any remaining class imbalance)
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
# 5. MIXUP
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
# 6. TRAINING LOOP
# ============================================================
print(f"\n[3/6] Training for {EPOCHS} epochs...")

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

        # Mixup
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
torch.save(best_model_wts, f"{OUTPUT_DIR}/resnet50_cbam_best.pth")
print(f"\n  Best val accuracy: {best_val_acc:.4f}")
print(f"  Model saved to {OUTPUT_DIR}/resnet50_cbam_best.pth")


# ============================================================
# 7. PLOT TRAINING CURVES
# ============================================================
print("\n[4/6] Saving training curves...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(history["train_loss"], label="Train Loss", linewidth=2)
ax1.plot(history["val_loss"],   label="Val Loss",   linewidth=2)
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
ax1.set_title("Training & Validation Loss"); ax1.legend(); ax1.grid(True, alpha=0.3)

ax2.plot(history["train_acc"], label="Train Acc", linewidth=2)
ax2.plot(history["val_acc"],   label="Val Acc",   linewidth=2)
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
ax2.set_title("Training & Validation Accuracy"); ax2.legend(); ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/training_curves.png", dpi=300, bbox_inches="tight")
plt.close()
print("  Saved training_curves.png")


# ============================================================
# 8. TEST SET EVALUATION
# ============================================================
print("\n[5/6] Evaluating on test set...")

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
        all_probs.extend(probs[:, 1].cpu().numpy())  # prob of class 1

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs  = np.array(all_probs)

# Classification report
test_acc = np.mean(all_preds == all_labels)
report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
print(f"\n  Test Accuracy: {test_acc:.4f}\n")
print(report)

# Save report
with open(f"{OUTPUT_DIR}/classification_report.txt", "w") as f:
    f.write(f"ResNet50 + CBAM — Microplastic Detection\n")
    f.write(f"========================================\n\n")
    f.write(f"Test Accuracy: {test_acc:.4f}\n\n")
    f.write(report)
print("  Saved classification_report.txt")

# --- Confusion Matrix ---
cm = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names, ax=ax,
            annot_kws={"size": 16})
ax.set_xlabel("Predicted", fontsize=13)
ax.set_ylabel("Actual", fontsize=13)
ax.set_title("Confusion Matrix — ResNet50 + CBAM", fontsize=14)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.close()
print("  Saved confusion_matrix.png")

# --- ROC Curve ---
fpr, tpr, _ = roc_curve(all_labels, all_probs)
roc_auc = sk_auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
ax.set_xlabel("False Positive Rate", fontsize=13)
ax.set_ylabel("True Positive Rate", fontsize=13)
ax.set_title("ROC Curve — ResNet50 + CBAM", fontsize=14)
ax.legend(fontsize=12); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/roc_curve.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"  Saved roc_curve.png (AUC = {roc_auc:.4f})")


# ============================================================
# 9. GRAD-CAM VISUALIZATION
# ============================================================
print("\n[6/6] Generating Grad-CAM heatmaps...")

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        target = output[0, class_idx]
        target.backward()

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        # Normalize
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx


def show_gradcam(img_path, model, target_layer, transform, class_names, save_path):
    """Generate and save Grad-CAM for a single image"""
    from PIL import Image
    import torch.nn.functional as F

    img = Image.open(img_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(DEVICE)

    gradcam = GradCAM(model, target_layer)
    cam, pred_class = gradcam.generate(input_tensor)

    # Resize CAM to image size
    cam_resized = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize((224, 224)))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original
    img_resized = img.resize((224, 224))
    axes[0].imshow(img_resized)
    axes[0].set_title("Original Image", fontsize=13)
    axes[0].axis("off")

    # Heatmap
    axes[1].imshow(cam_resized, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap", fontsize=13)
    axes[1].axis("off")

    # Overlay
    axes[2].imshow(img_resized)
    axes[2].imshow(cam_resized, cmap="jet", alpha=0.5)
    axes[2].set_title(f"Overlay — Pred: {class_names[pred_class]}", fontsize=13)
    axes[2].axis("off")

    plt.suptitle("Grad-CAM Visualization — ResNet50 + CBAM", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# Generate Grad-CAMs for 4 test images (2 per class)
target_layer = model.cbam4  # attention layer — most interesting for visualization
gradcam_count = 0

for cls_idx, cls_name in enumerate(class_names):
    cls_dir = Path(DATASET_DIR) / "test" / cls_name
    images = sorted(cls_dir.glob("*"))[:2]  # first 2 images per class

    for img_idx, img_path in enumerate(images):
        save_path = f"{OUTPUT_DIR}/gradcam_{cls_name}_{img_idx}.png"
        try:
            show_gradcam(img_path, model, target_layer, val_test_transform,
                         class_names, save_path)
            print(f"  Saved gradcam_{cls_name}_{img_idx}.png")
            gradcam_count += 1
        except Exception as e:
            print(f"  Warning: Grad-CAM failed for {img_path.name}: {e}")

if gradcam_count == 0:
    print("  No Grad-CAM images generated (check test set paths)")


# ============================================================
# 10. FINAL SUMMARY
# ============================================================
print(f"\n{'='*55}")
print(f"  TRAINING COMPLETE — RESULTS SUMMARY")
print(f"{'='*55}")
print(f"  Model:         ResNet50 + CBAM")
print(f"  Best Val Acc:  {best_val_acc:.4f}")
print(f"  Test Acc:      {test_acc:.4f}")
print(f"  AUC-ROC:       {roc_auc:.4f}")
print(f"{'='*55}")
print(f"  Files saved to: {OUTPUT_DIR}/")
print(f"    - resnet50_cbam_best.pth   (model weights)")
print(f"    - training_curves.png      (loss & accuracy plots)")
print(f"    - confusion_matrix.png     (test set)")
print(f"    - roc_curve.png            (test set)")
print(f"    - classification_report.txt")
print(f"    - gradcam_*.png            (attention heatmaps)")
print(f"{'='*55}")
print(f"\n  All done! Ready for your paper.")
