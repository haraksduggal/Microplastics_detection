<div align="center">

# 🔬 Microplastics Detection

### Machine Learning-Based Detection of Marine Microplastics Using Attention-Enhanced Deep Learning

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<p align="center">
  <em>An attention-enhanced deep learning framework for automated microplastic detection<br>with implications for human health and environmental sustainability</em>
</p>

---

<img src="https://img.shields.io/badge/SDG_3-Good_Health-4C9F38?style=flat-square" alt="SDG 3"/>
<img src="https://img.shields.io/badge/SDG_6-Clean_Water-26BDE2?style=flat-square" alt="SDG 6"/>
<img src="https://img.shields.io/badge/SDG_14-Life_Below_Water-0A97D9?style=flat-square" alt="SDG 14"/>

</div>

---

## 📖 About

Marine microplastic pollution is one of the most pressing environmental and public health challenges of the 21st century. These microscopic plastic fragments (< 5mm) have been detected in marine ecosystems, drinking water, food chains, and even human biological tissues.

Traditional identification methods like **FTIR** and **Raman Spectroscopy** are expensive, slow, and require specialized expertise. This project presents an **automated microplastic detection framework** using deep learning and transfer learning, making large-scale environmental monitoring feasible.

### ✨ Key Contributions

- **CBAM-Enhanced ResNet50** — Integrates Convolutional Block Attention Module for focused feature extraction on microplastic particles
- **Multi-Architecture Comparison** — Comprehensive evaluation across 4 model implementations
- **Cross-Framework Validation** — PyTorch and TensorFlow implementations on identical data
- **Grad-CAM Interpretability** — Visual explanations confirming attention on particle-relevant regions
- **Sustainability Aligned** — Contributes to UN SDGs 3, 6, and 14

---

## 🏗️ Architecture

```
┌─────────┐   ┌──────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐
│  Input   │──▶│  Conv1   │──▶│ Layer1  │──▶│ Layer2  │──▶│ Layer3  │
│ 224×224  │   │ BN+ReLU  │   │(frozen) │   │(frozen) │   │ 1024ch  │
└─────────┘   └──────────┘   └─────────┘   └─────────┘   └────┬────┘
                                                               │
                                                          ┌────▼────┐
                                                          │  CBAM   │
                                                          │ 1024ch  │
                                                          └────┬────┘
                                                               │
┌─────────┐   ┌──────────┐   ┌─────────┐   ┌─────────┐   ┌────▼────┐
│ Output  │◀──│ FC Head  │◀──│ AvgPool │◀──│  CBAM   │◀──│ Layer4  │
│ 2 class │   │2048→512→2│   │ Global  │   │ 2048ch  │   │ 2048ch  │
└─────────┘   └──────────┘   └─────────┘   └─────────┘   └─────────┘

CBAM Module: Feature Map F → Channel Attention → Spatial Attention → Refined F''
```

---

## 📊 Results

### Model Comparison

| Model | Framework | Test Acc. | F1-Score | AUC-ROC | Parameters | Final Loss |
|:------|:----------|:---------:|:--------:|:-------:|:----------:|:----------:|
| ResNet50 | PyTorch | 100.00% | 1.00 | 1.00 | 25.0M | 0.1099 |
| **ResNet50 + CBAM** | **PyTorch** | **100.00%** | **1.00** | **1.00** | **25.2M** | **0.1014** |
| EfficientNet-B0 | PyTorch | 100.00% | 1.00 | 1.00 | 4.7M | 0.1159 |
| ResNet50 (TF) | TensorFlow | 95.25% | 0.95 | 0.97 | 23.6M | 0.2700 |

### Key Findings

> 🎯 **ResNet50 + CBAM** achieves the **lowest training loss** (0.1014) — indicating the most confident, best-calibrated predictions
>
> ⚡ **EfficientNet-B0** matches performance with **5.4× fewer parameters** — ideal for edge deployment
>
> 🔄 **Cross-framework validation** reveals implementation-level impact on model performance

---

## 📁 Project Structure

```
Microplastics_detection/
│
├── 📂 dataset/
│   ├── train/
│   │   ├── microplastic/          # 1,600 images
│   │   └── non_microplastic/      # 1,600 images
│   ├── val/
│   │   ├── microplastic/          # 200 images
│   │   └── non_microplastic/      # 200 images
│   └── test/
│       ├── microplastic/          # 200 images
│       └── non_microplastic/      # 200 images
│
├── 📂 results/
│   ├── resnet50_cbam_best.pth     # Best model weights
│   ├── training_curves.png        # Loss & accuracy plots
│   ├── confusion_matrix.png       # Test set confusion matrix
│   ├── roc_curve.png              # ROC curve with AUC
│   ├── classification_report.txt  # Precision, recall, F1
│   └── gradcam_*.png              # Grad-CAM heatmaps
│
├── 📂 results_vanilla/            # Vanilla ResNet50 results
├── 📂 results_efficientnet/       # EfficientNet-B0 results
│
├── 🐍 augment_mp.py               # Image augmentation (781 → 2000)
├── 🐍 split_dataset.py            # Train/val/test split
├── 🐍 train_resnet50_cbam.py      # ResNet50 + CBAM training
├── 🐍 train_resnet50_vanilla.py   # Vanilla ResNet50 training
├── 🐍 train_efficientnet.py       # EfficientNet-B0 training
│
├── 📄 microplastic_ieee_paper.tex # IEEE format research paper
└── 📄 README.md
```

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install torch torchvision matplotlib scikit-learn seaborn tqdm pillow numpy
```

### Step 1 — Augment Microplastic Images

Expand 781 original microplastic images to 2,000 using microscopy-appropriate augmentations:

```bash
# Update INPUT_DIR in augment_mp.py to your image folder
python augment_mp.py
```

**Augmentations applied:** rotation, flipping, brightness/contrast jitter, Gaussian blur, noise injection, random crop-resize, sharpening (2–4 per image).

### Step 2 — Split Dataset

Create balanced train/val/test splits (80/10/10):

```bash
# Update paths in split_dataset.py
python split_dataset.py
```

### Step 3 — Train Models

```bash
# Train ResNet50 + CBAM (primary model)
python train_resnet50_cbam.py

# Train vanilla ResNet50 (baseline)
python train_resnet50_vanilla.py

# Train EfficientNet-B0 (efficiency comparison)
python train_efficientnet.py
```

> **Training time:** ~25 min per model on Apple M-series (MPS) | ~15 min on NVIDIA GPU

### Step 4 — View Results

All results are automatically saved to their respective `results/` directories:
- Training curves (loss & accuracy)
- Confusion matrices
- ROC curves with AUC scores
- Classification reports
- Grad-CAM heatmaps (ResNet50 models)

---

## 🧠 CBAM: How It Works

The **Convolutional Block Attention Module** applies two sequential attention mechanisms:

**Channel Attention** — *"What features to focus on"*
```
M_c(F) = σ(MLP(AvgPool(F)) + MLP(MaxPool(F)))
```

**Spatial Attention** — *"Where to focus"*
```
M_s(F') = σ(Conv7×7([AvgPool(F'); MaxPool(F')]))
```

CBAM adds only **~213K parameters** (+0.9%) to ResNet50 while providing:
- More focused attention on particle regions
- Faster convergence during training
- Better-calibrated prediction confidence

---

## 🔍 Grad-CAM Visualization

Grad-CAM heatmaps reveal **where** the model looks when making predictions:

| | CBAM Model | Vanilla Model |
|:---:|:---:|:---:|
| **Attention Pattern** | Concentrated on particles | Diffuse across image |
| **Background Suppression** | Strong | Weak |
| **Scientific Validity** | Focuses on morphology | Relies on context |

---

## 📋 Dataset Details

| Property | Value |
|:---------|:------|
| **Total Images** | 4,000 (2,000 per class) |
| **Original Microplastic** | 781 images (augmented to 2,000) |
| **Original Non-Microplastic** | 5,000 images (subsampled to 2,000) |
| **Image Size** | 224 × 224 pixels |
| **Microplastic Source** | Laboratory petri dish captures |
| **Non-Microplastic Source** | IFCB flow cytometry imaging |
| **Split Ratio** | 80% train / 10% val / 10% test |

---

## ⚙️ Training Configuration

| Hyperparameter | PyTorch Models | TensorFlow Model |
|:---------------|:--------------:|:----------------:|
| **Epochs** | 30 | 20 |
| **Optimizer** | Adam | Adam |
| **Learning Rate** | 1×10⁻⁴ | 1×10⁻⁴ |
| **Weight Decay** | 1×10⁻⁴ | — |
| **Scheduler** | Cosine Annealing | ReduceLROnPlateau |
| **Mixup α** | 0.2 | — |
| **Loss** | Weighted CE | Weighted CE |
| **Hardware** | Apple MPS | NVIDIA T4 (Colab) |

---

## 📝 Research Paper

The complete IEEE-format research paper is included as `microplastic_ieee_paper.tex`. It covers:

- Comprehensive literature review (31 references, Chicago style)
- Full methodology with CBAM mathematical formulation
- Four-model comparative analysis
- Training convergence and parameter efficiency study
- Grad-CAM interpretability analysis
- Cross-framework (PyTorch vs TensorFlow) validation
- Implications for human health and UN SDGs
- Honest limitations and 8 future work directions

---

## 🔮 Future Work

- 🔬 **Same-domain validation** with unified imaging protocols
- 🏷️ **Multi-class morphotype classification** (fiber, fragment, film, pellet, foam)
- 📡 **Spectral-visual data fusion** combining CNN features with FTIR/Raman data
- 📱 **Edge deployment** via quantization and pruning for portable devices
- 🎯 **Object detection** using YOLOv8 for particle-level localization and counting
- 🎨 **Generative augmentation** using GANs/diffusion models for synthetic training data
- 🧪 **Polymer identification** through multi-task learning
- 📈 **Longitudinal monitoring** integration with automated sampling stations

---

## 🛡️ Limitations

The microplastic and non-microplastic images originate from **distinct imaging modalities** (petri dish photography vs. flow cytometry), which may allow models to leverage imaging-domain features rather than particle morphology alone. This is transparently documented in the research paper. The framework is validated as a **laboratory pre-screening tool**, with same-domain evaluation identified as a priority for future work.

---

## 📚 Key References

1. Woo et al. (2018) — *CBAM: Convolutional Block Attention Module* — ECCV
2. He et al. (2016) — *Deep Residual Learning* — CVPR
3. Tan & Le (2019) — *EfficientNet* — ICML
4. Selvaraju et al. (2017) — *Grad-CAM* — ICCV
5. Jambeck et al. (2015) — *Plastic waste inputs into the ocean* — Science

> Full bibliography with 31 Chicago-style references available in the research paper.

---

## 👩‍💻 Author

**Haraks Duggal**
**Harpreet Singh**
**Jovan Kooner**
---

<div align="center">

*Built with 🧪 science and 💻 deep learning for a cleaner planet*

**If this project helped you, consider giving it a ⭐**

</div>
