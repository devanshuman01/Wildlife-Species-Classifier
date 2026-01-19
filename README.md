# Wildlife Species Classifier 🦁🐯

A production-grade Deep Learning project that classifies distinct wildlife species using **Transfer Learning (MobileNetV2)** and **TensorFlow/Keras**.

## 📌 Project Overview
This project addresses the challenge of accurately identifying wildlife species from images. By leveraging a pre-trained **MobileNetV2** model, the system achieves high accuracy with efficient training, making it suitable for real-time applications.

### Key Features
* **Transfer Learning:** Utilizes MobileNetV2 (pre-trained on ImageNet) to extract robust features, significantly reducing training time compared to training from scratch.
* **Data Augmentation:** Implements real-time augmentation layers (`RandomFlip`, `RandomRotation`) to prevent overfitting and improve generalization on unseen data.
* **Production-Ready Preprocessing:** Includes rescaling layers `[-1, 1]` to match the specific requirements of the MobileNet architecture.
* **Optimized Performance:** Trained using the **Adam optimizer** for fast convergence.

## 🛠️ Tech Stack
* **Language:** Python
* **Framework:** TensorFlow / Keras
* **Architecture:** MobileNetV2 (CNN)
* **Libraries:** NumPy, Pandas, Matplotlib

## 📊 Model Architecture
The model consists of a custom head attached to the frozen MobileNetV2 base:
1.  **Input Layer:** (224, 224, 3)
2.  **Augmentation Block:** Random Horizontal Flip + Random Rotation (0.2)
3.  **Base Model:** MobileNetV2 (Weights frozen)
4.  **Pooling:** GlobalAveragePooling2D
5.  **Dropout:** 0.2 (To reduce overfitting)
6.  **Output:** Dense Layer (Softmax activation for 5 classes)

## 🚀 How to Run
1.  **Clone the repository:**
    ```bash
    https://github.com/devanshuman01/Wildlife-Species-Classifier
    ```
2.  **Open the Notebook:**
    Upload `Wildlife_Species_Classifier_CNN.ipynb` to [Google Colab](https://colab.research.google.com/).
3.  **Upload Data:**
    Ensure your dataset is uploaded to your Google Drive or Colab environment as per the notebook instructions.
4.  **Run All Cells:**
    Execute the cells to train the model and view the accuracy plots.

## 📈 Results
* **Accuracy:** Achieved ~99% Validation Accuracy (after 10 epochs).
* **Classes:** Trained to identify 5 specific wildlife categories.
