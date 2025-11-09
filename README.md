# Multimodal-Feature-Extraction-and-Model-Comparison-for-Image-Classification

**Note:** This repository is currently being organized and cleaned. Some files and structure may change as documentation and code are updated.

## Overview
This project explores feature engineering, model development, and comparative performance analysis on the MNIST handwritten digits dataset. It demonstrates the end to end process of transforming raw image data into machine learning ready features and benchmarking multiple classification approaches.

## Objectives
- Apply 2D Discrete Cosine Transform (DCT) for feature extraction and frequency domain representation  
- Perform dimensionality reduction through eigen decomposition and PCA like projection  
- Train and evaluate multiple models including a Random Forest, custom SVM implementation (QP based), and a Convolutional Neural Network (CNN) built in PyTorch  
- Compare performance, interpretability, and computational cost across traditional and deep learning methods  

## Key Results
| Model | Description | Accuracy |
|--------|--------------|-----------|
| Random Forest | Trained on DCT based 60 dimensional features | 42.3 percent |
| Custom Linear SVM | Implemented from scratch using quadratic programming | 15.0 percent |
| Convolutional Neural Network (CNN) | Two layer CNN trained on raw 28 x 28 images | 97.4 percent |

The CNN significantly outperformed traditional models by leveraging spatial hierarchies directly from pixel data.

## Technical Stack
- Languages and Libraries: Python, NumPy, Pandas, Matplotlib, Seaborn, scikit learn, PyTorch, CVXOPT  
- Concepts Demonstrated:  
  - Frequency domain feature engineering using DCT  
  - Eigen decomposition and feature reduction  
  - Model comparison and evaluation  
  - Visualization and interpretability through confusion matrices and metrics  

## Highlights
- End to end pipeline from preprocessing and feature extraction to visualization and model evaluation  
- Demonstrates frequency based and spatial feature learning side by side  
- Includes a self implemented linear SVM trained via quadratic programming  
- Highlights trade offs between accuracy, interpretability, and computational cost  

## Insights
- DCT based features capture general structure but lose fine grained spatial detail  
- Random Forest performance was limited by feature expressiveness  
- CNN accuracy was highest due to its ability to learn spatial hierarchies directly from pixel data  
- Deep learning inductive biases make CNNs well suited for image recognition tasks  

## Potential Extensions
- Extend the DCT based pipeline to CIFAR 10 or custom grayscale datasets  
- Integrate hyperparameter optimization for classical models  
- Experiment with hybrid models combining frequency and convolutional features  
- Visualize CNN activations and feature maps for interpretability  

## Author
**Addison DeSalvo**  
Johns Hopkins University — M.S. Data Science  
Focus Areas: Swarm AI, Machine Learning, and Public Health Applications
