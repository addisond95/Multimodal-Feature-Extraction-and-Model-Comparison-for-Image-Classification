# Multimodal-Feature-Extraction-and-Model-Comparison-for-Image-Classification

## Overview
This project explores **feature engineering, model development, and comparative performance analysis** on the MNIST handwritten digits dataset. It demonstrates the end-to-end process of transforming raw image data into machine-learning-ready features and benchmarking multiple classification approaches.

## Objectives
- Apply **2D Discrete Cosine Transform (DCT)** for feature extraction and frequency-domain representation.  
- Perform **dimensionality reduction** through **eigen decomposition** and PCA-like projection.  
- Train and evaluate multiple models — **Random Forest**, **custom SVM implementation (QP-based)**, and a **Convolutional Neural Network (CNN)** built in **PyTorch**.  
- Compare performance, interpretability, and computational cost across traditional and deep learning methods.  

## Key Results
| Model | Description | Accuracy |
|--------|--------------|-----------|
| Random Forest | Trained on DCT-based 60-dimensional features | **42.3%** |
| Custom Linear SVM | Implemented from scratch using quadratic programming | **15.0%** |
| Convolutional Neural Network (CNN) | Two-layer CNN trained on raw 28×28 images | **97.4%** |

The CNN significantly outperformed traditional models by leveraging spatial hierarchies directly from pixel data, confirming the advantage of deep learning architectures for image recognition.

## Technical Stack
- **Languages & Libraries:** Python, NumPy, Pandas, Matplotlib, Seaborn, scikit-learn, PyTorch, CVXOPT  
- **Concepts Demonstrated:**  
  - Frequency-domain feature engineering using DCT  
  - Eigen decomposition and feature reduction  
  - Model comparison and evaluation (Random Forest, SVM, CNN)  
  - Visualization and interpretability through confusion matrices and metrics  

## Highlights
- **End-to-end pipeline:** from preprocessing and feature extraction to visualization and model evaluation.  
- Demonstrates **frequency-based feature learning** and **spatial feature learning** side-by-side.  
- Includes a **self-implemented linear SVM** trained via quadratic programming to illustrate core optimization principles.  
- Highlights the trade-offs between **accuracy, interpretability, and computational cost**.  

## Insights
- DCT-based directional features capture general structure but lose fine-grained spatial detail.  
- The Random Forest achieved moderate performance but was constrained by limited feature expressiveness.  
- The CNN achieved near-perfect accuracy by learning spatial hierarchies directly from raw image data.  
- Deep learning’s **inductive bias toward locality and translation invariance** makes it particularly suited for image recognition tasks.

## Potential Extensions
- Extend the DCT-based pipeline to **CIFAR-10** or custom grayscale datasets.  
- Integrate **hyperparameter optimization** for classical models.  
- Experiment with **hybrid models** combining frequency and convolutional features.  
- Visualize **CNN activations** and **feature maps** for interpretability.

## Author
**Addison DeSalvo**  
Johns Hopkins University — M.S. Data Science  
*Focus Areas: Swarm AI, Machine Learning, and Public Health Applications*
