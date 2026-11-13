# 🔍 TruthLens: AI-Generated vs Real Image Detection

A Deep Learning-based image classifier that detects whether an image is a **real photograph** or **AI-generated**, built with Convolutional Neural Networks (CNN), TensorFlow/Keras and Streamlit. Users upload an image and get a real-time prediction with a confidence score.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)

---

## 📌 Project Overview

AI-generated images are getting harder to tell apart from real photographs. In this project I trained a CNN from scratch on the **CIFAKE** dataset for binary classification (Real vs AI-generated) and deployed it in a Streamlit app.

This was my first practical CNN project. I built a baseline, analyzed where it failed, improved it with a deeper architecture, and then tested it on images from outside the dataset to understand its real-world limits.

## 🎯 Project Objectives

- Build a CNN for binary classification on RGB images.
- Apply normalization and data augmentation inside the model.
- Train with EarlyStopping, ModelCheckpoint and ReduceLROnPlateau.
- Evaluate with precision, recall and a confusion matrix.
- Compare a baseline with an improved model.
- Deploy the model in Streamlit and document its limitations honestly.

## 🛠️ Technologies & Tools

| Technology | Purpose |
|---|---|
| Python | Core language |
| TensorFlow & Keras | CNN development and training |
| NumPy, Matplotlib | Array operations and visualization |
| scikit-learn | Classification report, confusion matrix |
| Pillow | Image loading and resizing in the app |
| Streamlit | Web application |
| Git & GitHub | Version control |

## 📊 Dataset: CIFAKE

[CIFAKE: Real and AI-Generated Synthetic Images](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) from Kaggle.

- **Real images:** from CIFAR-10
- **Fake images:** generated with Stable Diffusion
- **Size:** 32 × 32 pixels, RGB
- **Classes:** `FAKE` = 0, `REAL` = 1
- **Training:** 100,000 images (90% train / 10% validation)
- **Test:** 20,000 images (10,000 per class, never used in training)

> The dataset is not included in this repository. Download it from Kaggle and place the `train/` and `test/` folders in the project root.

## 🧠 Model Architecture

### Model 1: Baseline CNN

```
Input (32x32x3) → Random Flip + Rotation (training only) → Rescaling (0-1)
→ Conv 32 → MaxPool → Conv 64 → MaxPool → Conv 128 → MaxPool
→ Flatten → Dense 128 → Dropout 0.5 → Dense 1 (Sigmoid)
```

### Model 2: Improved CNN (used in the app)

```
Input (32x32x3) → Random Flip (training only) → Rescaling (0-1)
→ Block 1: [Conv 32 → BatchNorm → ReLU] x2 → MaxPool → Dropout 0.25
→ Block 2: [Conv 64 → BatchNorm → ReLU] x2 → MaxPool → Dropout 0.25
→ Block 3: [Conv 128 → BatchNorm → ReLU] x2 → MaxPool → Dropout 0.25
→ Flatten → Dense 128 → Dropout 0.5 → Dense 1 (Sigmoid)
```

**Changes in Model 2:** BatchNormalization after every convolution, two Conv layers per block, Dropout after each block, random rotation removed, plus ReduceLROnPlateau and longer EarlyStopping patience.

Normalization and augmentation are built into the models, so the saved `.keras` files handle preprocessing themselves.

### Training configuration

| Setting | Model 1 | Model 2 |
|---|---|---|
| Optimizer / Loss | Adam / Binary Crossentropy | Adam / Binary Crossentropy |
| Batch size | 64 | 64 |
| Max epochs | 20 | 30 |
| EarlyStopping patience | 3 | 5 |
| ReduceLROnPlateau | No | Yes |
| Stopped at epoch | 8 | 27 |
| Best epoch | 5 | 22 |

## 📈 Model Performance

Evaluated on the 20,000-image CIFAKE test set.

| Metric | Model 1 | Model 2 |
|---|---|---|
| **Test accuracy** | 86.02% | **97.27%** |
| Best validation accuracy | 86.45% | 97.63% |
| FAKE precision / recall | 0.79 / 0.98 | 0.97 / 0.97 |
| REAL precision / recall | 0.98 / 0.74 | 0.97 / 0.97 |

**Confusion matrices**

| Model 1 | Predicted FAKE | Predicted REAL |
|---|---|---|
| **Actual FAKE** | 9839 | 161 |
| **Actual REAL** | 2635 | 7365 |

| Model 2 | Predicted FAKE | Predicted REAL |
|---|---|---|
| **Actual FAKE** | 9730 | 270 |
| **Actual REAL** | 276 | 9724 |

**Key observations**

- Model 1 was **biased towards FAKE**: it labelled 26% of real images as fake. Model 2 removed this bias (errors are now almost equal: 270 vs 276).
- Model 1 overfit early: its best epoch was 5, after which training accuracy kept rising while validation accuracy dropped.
- Validation (97.6%) and test (97.3%) accuracy are close, so Model 2 generalizes well **within CIFAKE**.
- Several changes were made together, so I cannot say which one contributed most.

## ⚙️ Prediction Pipeline (Streamlit App)

1. Upload a JPG, JPEG or PNG image.
2. Convert to RGB, center-crop to a square and resize to 32 × 32.
3. Reshape to `(1, 32, 32, 3)` and run the model.
4. The model outputs the probability of REAL (FAKE probability is `1 - p`).
5. The app shows the predicted class, a confidence score and probability bars.

No division by 255 is done in the app, because the `Rescaling` layer is already inside the model.

## 📸 Screenshots

> Add your screenshots here, for example:
>
> `![App screenshot](screenshots/app.png)`
> `![Confusion matrix](screenshots/confusion_matrix.png)`

## ⚠️ Limitations

The 97.3% accuracy applies to **CIFAKE, not to all images.**

- **Fails on camera photos.** On high-resolution real photos taken with a camera, the model often predicted **AI-generated**.
- **Dataset bias.** CIFAKE's real images come from CIFAR-10 (only 10 categories, already low-resolution), so the model likely learned the dataset's style rather than general features of real photographs.
- **Single generator.** All fake images are from Stable Diffusion, so other generators (Midjourney, DALL·E) may not be detected reliably.
- **Information loss.** Shrinking a large photo to 32 × 32 removes the fine textures that could reveal AI generation.
- **Confidence is not certainty.** The model can be confident and still wrong on unfamiliar images.

TruthLens is a **learning project**, not a reliable real-world AI-image detector.

## 🔮 Future Improvements

- Train on a higher-resolution, more diverse dataset with real camera photos and multiple generators.
- Use transfer learning (e.g. EfficientNetB0) with GPU training on Google Colab.
- Add Grad-CAM heatmaps to show what the model looks at.
- Run ablation experiments to find which change helped most.
- Deploy the app publicly (Streamlit Community Cloud or Hugging Face Spaces).

## 📂 Project Structure

```
TruthLens/
├── notebooks/
│   └── truthlens_notebook.ipynb   # data loading, training, evaluation
├── app.py                         # Streamlit application
├── truthlens_model.keras          # Model 1 (baseline)
├── truthlens_model_v2.keras       # Model 2 (used by app.py)
├── requirements.txt
├── .gitignore
└── README.md
```

## ▶️ How to Run

```bash
git clone https://github.com/YOUR_USERNAME/TruthLens.git
cd TruthLens
pip install -r requirements.txt     # Python 3.10 or 3.11 recommended
streamlit run app.py
```

The app opens at `http://localhost:8501`.

To retrain, download CIFAKE, place `train/` and `test/` in the project root and run the notebook. Training Model 2 took about 91 minutes on a laptop CPU.

## 📚 Key Learnings

- Building and training a CNN for binary image classification.
- Input pipelines, normalization and augmentation as model layers.
- Spotting overfitting from training curves and fixing it with Dropout, BatchNormalization and EarlyStopping.
- Evaluating with precision, recall and confusion matrices, not accuracy alone.
- Understanding dataset bias: high test accuracy does not guarantee real-world performance.
- Deploying a Keras model in Streamlit.

## 🙏 Acknowledgements

CIFAKE: Bird, J.J. and Lotfi, A. (2023). CIFAR-10: Krizhevsky, A. and Hinton, G.

## 👨‍💻 Author

**Ujjwal Tripathi**

BCA Graduate | Aspiring AI/ML Developer

Interested in Machine Learning, Deep Learning and Generative AI.