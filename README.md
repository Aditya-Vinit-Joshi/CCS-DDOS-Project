#  DDoS Attack Detection and Classification using Machine & Deep Learning

This project focuses on the accurate **detection and classification of Distributed Denial of Service (DDoS) attacks** using supervised learning algorithms. We employ classical machine learning models (Random Forest, XGBoost, K-Nearest Neighbors) and a Multi-Layer Perceptron (MLP) neural network to handle a multi-class classification problem. The dataset used simulates real-world DDoS scenarios and is sourced from Kaggle.

>  Project Files:
> - `CCS_Project_Part1.ipynb`: Data exploration, preprocessing, and traditional ML model training.
> - `CCS_Project_Part2.ipynb`: PCA optimization and MLP implementation.
> - `AdityaJoshi-AnayaDandekar-CS258-Project.pptx`: Final project presentation.

---

##  Table of Contents

- [Introduction](#introduction)
- [Motivation & Literature Survey](#motivation--literature-survey)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Models Implemented](#models-implemented)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Conclusion & Future Work](#conclusion--future-work)
- [Setup & Installation](#setup--installation)
- [References](#references)

---

##  Introduction

DDoS attacks aim to overwhelm networks or services by flooding them with traffic, causing denial of service to legitimate users. This project applies machine learning to detect and classify different types of DDoS attacks efficiently and accurately.

---

##  Motivation & Literature Survey

The base methodology stems from studies like:

- **SPIDER**: PCA + RNN-based robust detection system
- **Autoencoders**: Unsupervised anomaly detection
- **CNN-BiLSTM**: Temporal pattern detection
- **Hybrid Deep Models**: GANs + ResNet + AlexNet for high-accuracy multi-class detection

Challenges addressed:
- Detecting AND classifying attack types
- Handling high-dimensional data
- Improving accuracy through feature selection (PCA, correlation analysis)

---

##  Dataset

- **Source**: [Kaggle – DDoS Attack Network Logs by Jacob van Steyn](https://www.kaggle.com/datasets/jacobvs/ddos-attack-network-logs)
- **Format**: ARFF → Converted to CSV
- **Size**: ~2.1 million rows, 28 features
- **Classes**: Normal, UDP-Flood, Smurf, SIDDOS, HTTP-Flood

### Key Features

- Traffic metadata: source/destination IPs, port info, flow timestamps
- Labels indicate attack class
- Mimics real-world Intrusion Detection System (IDS) logs

---

###  Methodology

###  Data Preprocessing

- **Dropped Columns**: Identifiers, timestamps (e.g., `FID`, `SEQ_NUMBER`)
- **String Decoding**: Converted byte strings to readable strings
- **Label Encoding**: Applied to `PKT_TYPE`, `PKT_CLASS`
- **Scaling**: StandardScaler / MinMaxScaler
- **Dimensionality Reduction**:
  - **Correlation matrix** for redundant feature removal
  - **Principal Component Analysis (PCA)** to reduce computational complexity

---

##  Models Implemented

| Algorithm            | Notebook Part | Optimization Techniques    |
|---------------------|---------------|-----------------------------|
| Random Forest        | Part 1        | Basic, no PCA              |
| XGBoost              | Part 1 & 2    | PCA, Correlation           |
| K-Nearest Neighbors  | Part 1        | Scaled & Cleaned Data      |
| Multi-Layer Perceptron (MLP) | Part 2 | With and without PCA       |

###  Feature Engineering

- Dimensionality reduced using PCA
- Feature correlation considered to drop redundant columns
- One-hot encoding/label encoding applied where needed

---

##  Evaluation Metrics

- **Accuracy Score**
- **F1 Score** (macro/micro/weighted)
- **Confusion Matrix**
- **Classification Report**
- **Execution Time**

---

## 🏁 Results

| Model                | Accuracy (%) | Notes                              |
|---------------------|--------------|-------------------------------------|
| Random Forest        | ~95.2%       | Fast, good balance                 |
| XGBoost              | ~96.4%       | Best performer with PCA            |
| KNN                  | ~92.7%       | Sensitive to scaling               |
| MLP (neural network) | ~94.1%       | Performs well after PCA            |

>  **Observations**:
> - Smurf attack was hard to classify due to class imbalance
> - PCA improved execution time significantly without much loss in accuracy

---

##  Conclusion & Future Work

### Accomplishments
- Built an end-to-end ML pipeline for multi-class DDoS attack classification
- Applied PCA and correlation to reduce data noise
- Compared traditional ML and neural models

###  Future Scope
- Apply **SMOTE** to balance class distribution
- Perform **hyperparameter tuning**
- Use **deep CNN/LSTM** models for sequential data
- Integrate with **real-time traffic sniffers** (e.g., Scapy)
- Explore **Explainable AI** for security auditability

---

##  Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/ddos-attack-detection.git
cd ddos-attack-detection
