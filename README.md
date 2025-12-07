# Maritime Navigational Status Classification: Kattegat Strait AIS Data Analysis

A machine learning project for classifying vessel navigational status from Automatic Identification System (AIS) data collected from ships transiting the Kattegat Strait between January 1st and March 10th, 2022.

## Overview

This project develops and evaluates classification models to predict maritime navigational status (e.g., "Under way using engine", "At anchor", "Constrained by her draught") from AIS data. The dataset comprises 358,351 AIS records from vessels transiting the Kattegat Strait between January 1st and March 10th, 2022, providing a comprehensive view of maritime traffic patterns during this period.

The models use vessel motion features (speed, course, heading) and physical characteristics (length, width, draught, ship type) to classify navigational status. The project addresses a highly imbalanced multiclass classification problem (~92% of samples are "Under way using engine") and demonstrates the effectiveness of tree-based models (Random Forest and XGBoost) over linear models for this maritime domain application.

The temporal and geographic specificity of this dataset—focusing on a critical shipping route during a specific 69-day period in early 2022—makes this analysis particularly valuable for understanding navigational patterns in the Kattegat Strait region.

## Project Structure

```
Maritime-Navigational-Status-Classification/
├── data/
│   ├── ais_data.csv              # Raw AIS data: Kattegat Strait, Jan 1 - Mar 10, 2022 (358,351 samples)
│   └── ais_data_model_ready.csv   # Preprocessed data (after cleaning)
├── notebooks/
│   ├── 01_eda_preprocessing.ipynb    # Exploratory data analysis and preprocessing
│   └── 02_modeling_evaluation.ipynb   # Model training, evaluation, and comparison
├── src/
│   ├── preprocessing.py          # Data cleaning and preprocessing functions
│   └── train_models.py           # Model building functions
├── results/
│   ├── figures/                  # Visualization outputs
│   │   ├── eda & preprocessing/  # 18 EDA figures
│   │   └── modeling & evaluation/ # 5 model evaluation figures
│   │       ├── confusion_matrix_*.png (3 files)
│   │       └── feature_importance_*.png (2 files)
│   └── metrics/                  # Evaluation metrics and reports
│       ├── model_evaluation_metrics.csv
│       ├── validation_metrics.csv
│       ├── test_metrics.csv
│       ├── confusion_matrix_*.csv (3 files)
│       └── classification_report_*.txt (3 files)
├── report/                       # Final report
│   └── Rendered Notebooks/       # Rendered notebook outputs
│       ├── 01_eda_preprocessing.pdf
│       └── 02_modeling_evaluation.pdf
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Data

### Dataset Description

**Source**: Automatic Identification System (AIS) data  
**Geographic Scope**: Kattegat Strait  
**Temporal Scope**: January 1st - March 10th, 2022 (69 days)  
**Total Records**: 358,351 AIS observations  
**Final Dataset**: ~326,000 samples after preprocessing

The Kattegat Strait is a critical maritime passage connecting the Baltic Sea to the North Sea, making it one of the busiest shipping routes in Northern Europe. The data collected during this period captures diverse navigational behaviors across different vessel types, weather conditions, and operational contexts.

### Data Characteristics

- **Highly imbalanced classes**: ~92% of observations are "Under way using engine"
- **Diverse vessel types**: Cargo, Tanker, Fishing, Passenger, and specialized vessels
- **Multiple navigational statuses**: 10 distinct status classes after preprocessing
- **Rich feature set**: Motion characteristics (SOG, COG, heading) and vessel specifications (dimensions, type)

## Features

### Input Features (7 total)
- **Motion Features (3)**: 
  - Speed over Ground (SOG) - knots
  - Course over Ground (COG) - degrees
  - Heading - degrees
- **Vessel Specifications (4)**:
  - Length - meters
  - Width - meters
  - Draught - meters
  - Ship Type - categorical (one-hot encoded)

### Target Variable
- **Navigational Status**: 10 classes (after preprocessing)
  - At anchor
  - Constrained by her draught
  - Engaged in fishing
  - Moored
  - Power-driven vessel pushing ahead or towing alongside
  - Power-driven vessel towing astern
  - Reserved for future amendment [HSC]
  - Restricted maneuverability
  - Under way sailing
  - Under way using engine (majority class, ~92%)

## Installation

### Prerequisites
- Python 3.10+ (required for type hints using `|` syntax)

### Setup

1. **Clone the repository**:
```bash
git clone <repository-url>
cd Maritime-Navigational-Status-Classification
```

2. **Create a virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Complete Pipeline

1. **Exploratory Data Analysis and Preprocessing**:
   - Open `notebooks/01_eda_preprocessing.ipynb`
   - Run all cells to:
     - Perform exploratory data analysis
     - Apply preprocessing pipeline (outlier filtering, invalid class removal, rare class removal)
     - Generate preprocessed data: `data/ais_data_model_ready.csv`
   - Output: 18 figures in `results/figures/eda & preprocessing/`

2. **Model Training and Evaluation**:
   - Open `notebooks/02_modeling_evaluation.ipynb`
   - Run all cells to:
     - Load and split preprocessed data (64% train, 16% validation, 20% test)
     - Train three models: Logistic Regression, Random Forest, XGBoost
     - Evaluate on validation and test sets
     - Generate confusion matrices and feature importance plots
     - Save all metrics and reports
   - Outputs: 
     - Metrics in `results/metrics/`
     - Figures in `results/figures/modeling & evaluation/`

### Using the Preprocessing Pipeline Programmatically

```python
from src.preprocessing import prepare_ais_data

# Load and preprocess data end-to-end
X_train, X_val, X_test, y_train, y_val, y_test, preprocessor = prepare_ais_data(
    path="../data/ais_data.csv"
)

# Preprocessor is fitted and ready to transform new data
X_train_processed = preprocessor.transform(X_train)
```

### Building and Training Models

```python
from src.train_models import (
    build_logistic_regression_model,
    build_random_forest_model,
    build_xgboost
)
from sklearn.preprocessing import LabelEncoder

# Build models
lr_model = build_logistic_regression_model()
rf_model = build_random_forest_model()
xgb_model = build_xgboost()

# Train (after preprocessing features)
lr_model.fit(X_train_processed, y_train)
rf_model.fit(X_train_processed, y_train)

# XGBoost requires integer-encoded labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
xgb_model.fit(X_train_processed, y_train_encoded)
```

## Data Preprocessing Pipeline

The preprocessing pipeline (`src/preprocessing.py`) performs the following steps:

1. **Load Raw Data**: Reads CSV and drops identifier columns (MMSI, Unnamed: 0)
2. **Filter Physical Outliers**: Removes physically impossible values:
   - SOG: [0, 60] knots
   - COG & Heading: [0, 360] degrees
   - Width: (0, 80] meters
   - Length: (0, 400] meters
   - Draught: (0, 25] meters
3. **Clean Types**: Standardizes data types and handles missing values
4. **Drop Invalid Classes**: Removes "Unknown value" (data quality issue)
5. **Separate Features and Target**: Creates feature matrix X and target vector y
6. **Drop Rare Classes**: Removes classes with < 50 samples
7. **Stratified Split**: 64% train, 16% validation, 20% test (preserves class distribution)
8. **Build Preprocessor**: StandardScaler for numeric features, OneHotEncoder for categorical

## Models

Three models are evaluated:

1. **Logistic Regression**: Linear baseline with balanced class weights
2. **Random Forest**: 300 trees, balanced class weights, no max depth limit
3. **XGBoost**: 300 trees, learning rate 0.05, no max depth limit, subsample 0.8

All models use class weighting to handle the severe class imbalance.

## Results

### Model Performance Summary

#### Validation Set Performance

| Model | Accuracy | Macro-F1 | Weighted-F1 | Precision (Macro) | Recall (Macro) |
|-------|----------|----------|-------------|-------------------|----------------|
| Logistic Regression | 0.717 | 0.373 | 0.797 | 0.303 | 0.784 |
| Random Forest | **0.986** | **0.875** | **0.985** | **0.915** | **0.844** |
| XGBoost | 0.978 | 0.834 | 0.978 | 0.886 | 0.800 |

#### Test Set Performance

| Model | Accuracy | Macro-F1 | Weighted-F1 | Precision (Macro) | Recall (Macro) |
|-------|----------|----------|-------------|-------------------|----------------|
| Logistic Regression | 0.719 | 0.369 | 0.798 | 0.300 | 0.772 |
| Random Forest | **0.985** | **0.883** | **0.985** | **0.919** | **0.860** |
| XGBoost | 0.979 | 0.846 | 0.978 | 0.884 | 0.826 |

**Best Model**: **Random Forest** achieves the highest performance across all metrics, demonstrating excellent generalization with consistent performance between validation and test sets.

### Key Findings

1. **Tree-based models significantly outperform linear models**: Random Forest and XGBoost achieve ~98% accuracy vs. ~72% for Logistic Regression
2. **Macro-F1 scores reflect class imbalance challenge**: Even the best model (Random Forest) achieves 88.3% macro-F1, indicating difficulty with rare classes
3. **Feature importance aligns with domain knowledge**:
   - Speed over Ground (SOG) is the most important feature
   - Draught is highly important (especially for "Constrained by her draught" status)
   - Vessel specifications (length, width) contribute to classification
4. **Motion features are primary discriminators**: SOG, COG, and heading effectively distinguish between active and stationary navigational statuses
5. **Model generalization is strong**: Minimal overfitting observed (validation and test performance are very similar)

## Output Files

All evaluation metrics, confusion matrices, and classification reports are exported to `results/metrics/` for reproducibility:

- **Metrics CSV files**: `validation_metrics.csv`, `test_metrics.csv`, `model_evaluation_metrics.csv`
- **Confusion matrices**: `confusion_matrix_*_test.csv` (one per model)
- **Classification reports**: `classification_report_*_test.txt` (detailed per-class metrics)

Visualizations are saved to `results/figures/`:
- **EDA figures**: 18 figures covering data distributions, correlations, and per-class patterns
- **Model evaluation figures**: Confusion matrices and feature importance plots

## Dependencies

See `requirements.txt` for complete list. Key packages:
- `pandas>=2.0.0` - Data manipulation
- `numpy>=1.24.0` - Numerical operations
- `scikit-learn>=1.3.0` - Machine learning
- `xgboost>=2.0.0` - Gradient boosting
- `matplotlib>=3.7.0` - Plotting
- `seaborn>=0.12.0` - Statistical visualizations

## Reproducibility

- All random seeds are set to 42 for reproducibility
- Stratified splitting preserves class distribution across train/val/test
- All metrics and figures are saved to `results/` directory
- Preprocessing pipeline is deterministic

## Acknowledgments

- **AIS Data**: Ships transiting the Kattegat Strait between January 1st and March 10th, 2022: https://www.kaggle.com/datasets/eminserkanerdonmez/ais-dataset?select=ais_data.csv
- Maritime domain knowledge references
- Kattegat Strait maritime traffic analysis
