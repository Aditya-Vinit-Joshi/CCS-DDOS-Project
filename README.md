
# DDoS Attack Detection and Classification using Machine Learning

This repository contains the code and methodology for detecting and classifying Distributed Denial of Service (DDoS) attacks using traditional machine learning techniques. The code is implemented across two Jupyter notebooks and leverages dimensionality reduction techniques like PCA, alongside classifiers like Random Forest, XGBoost, and K-Nearest Neighbors.

## Repository Structure

```
.
├── CCS_Project_Part1.ipynb        # Initial data loading, cleaning, ML training
├── CCS_Project_Part2.ipynb        # Correlation, PCA, PCA-based ML retraining
├── README.md                      # Full project documentation
```

## Objective

The goal of this project is to:
- Detect and classify multiple types of DDoS attacks from network traffic logs
- Apply machine learning models and analyze their performance
- Improve computational efficiency and model accuracy using correlation analysis and PCA

## Dataset

- **Source**: [Kaggle - DDoS Attack Network Logs by Jacob van Steyn](https://www.kaggle.com/datasets/jacobvs/ddos-attack-network-logs)
- **Format**: ARFF (converted to DataFrame)
- **Size**: Approximately 2.1 million records with 28 features
- **Target Classes**: UDP-FLOOD, SMURF, SIDDOS, HTTP-FLOOD, and others

## Notebooks Overview

### 1. `CCS_Project_Part1.ipynb`

This notebook includes:

- Dataset download using `kagglehub`
- ARFF file loading with `scipy.io.arff`
- DataFrame creation
- Feature cleanup: dropping timestamps and identifiers
- Byte-string decoding to readable strings
- Label encoding for categorical features
- Data exploration with correlation heatmaps and distribution plots
- Model training:
  - RandomForestClassifier
  - XGBClassifier
  - KNeighborsClassifier
- Evaluation: Accuracy, Confusion Matrix, Classification Report, F1 Score

### 2. `CCS_Project_Part2.ipynb`

This notebook extends the first by:

- Performing correlation analysis to remove redundant features
- Applying PCA for dimensionality reduction
- Visualizing explained variance ratio to select components
- Retraining models on reduced data:
  - XGBClassifier with PCA
  - KNN with PCA
- Comparing model accuracy and computational efficiency

## Project Flow

1. **Data Loading**
```python
from scipy.io import arff
data, meta = arff.loadarff(path + '/final-dataset.arff')
df = pd.DataFrame(data)
```

2. **Data Preprocessing**
```python
df['PKT_TYPE'] = df['PKT_TYPE'].str.decode('utf-8')
df['PKT_CLASS'] = df['PKT_CLASS'].str.decode('utf-8')
df['PKT_TYPE'] = LabelEncoder().fit_transform(df['PKT_TYPE'])
df['PKT_CLASS'] = LabelEncoder().fit_transform(df['PKT_CLASS'])
```

3. **Exploratory Data Analysis**
```python
sns.heatmap(df.corr(), annot=False, cmap='coolwarm')
```

4. **Correlation and PCA**
```python
from sklearn.decomposition import PCA
pca = PCA(n_components=10)
reduced = pca.fit_transform(scaled_data)
```

5. **Model Training**
```python
model = RandomForestClassifier()
model.fit(X_train, y_train)
```

6. **Evaluation**
```python
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

## Installation and Setup

### 1. Clone this Repository

```bash
git clone https://github.com/your-username/ddos-attack-detection.git
cd ddos-attack-detection
```

### 2. Create Virtual Environment (Optional)

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost kagglehub scipy tensorflow
```

## Dependencies

This project requires the following Python packages:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- kagglehub
- scipy
- tensorflow (imported but not used)
- os
- time

You can install them via pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost kagglehub scipy tensorflow
```

## Results Summary

| Model                | Accuracy | Notes                                  |
|---------------------|----------|----------------------------------------|
| Random Forest        | ~95%     | Performs well without dimensionality reduction |
| XGBoost              | ~96%     | Highest accuracy, PCA improves speed   |
| K-Nearest Neighbors  | ~92%     | Sensitive to feature scaling           |

- PCA and correlation analysis improve computation without compromising accuracy.
- Some attack types (e.g., Smurf) show lower F1 due to class imbalance.

## Future Work

- Apply SMOTE to mitigate class imbalance
- Hyperparameter tuning via `GridSearchCV`
- Explore neural networks like MLP, CNN, or LSTM
- Real-time traffic monitoring integration with packet capture tools
- Use explainable AI (e.g., SHAP, LIME) for model transparency

## Authors

Aditya Joshi  
Anaya Dandekar  
MS Computer Science  
San Jose State University  

## License

This project is licensed under the MIT License.
