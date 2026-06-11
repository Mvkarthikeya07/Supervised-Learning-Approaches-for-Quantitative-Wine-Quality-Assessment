<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=28&pause=1000&color=8B0000&center=true&vCenter=true&width=800&lines=🍷+Wine+Quality+Prediction+System;Supervised+ML+%7C+Flask+%7C+Scikit-learn;End-to-End+ML+Deployment" alt="Typing SVG" />

<br/>

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.x-000000?style=for-the-badge&logo=flask&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Scientific-013243?style=for-the-badge&logo=numpy&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Production--Ready-brightgreen?style=for-the-badge)

<br/>

> **A production-grade supervised machine learning system that predicts red wine quality from physicochemical properties — trained, serialized, and deployed as a real-time Flask web application.**

<br/>

[![Research Paper](https://img.shields.io/badge/📄_Published_Research-IJIRT_Journal-red?style=for-the-badge)](https://ijirt.org/article?manuscript=187067)
[![Live Demo](https://img.shields.io/badge/🚀_Run_Locally-Flask_App-darkred?style=for-the-badge)](#-installation--usage)

</div>

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Live Screenshots](#-live-screenshots)
- [ML Pipeline & Methodology](#-ml-pipeline--methodology)
- [Model Comparison & Selection](#-model-comparison--selection)
- [Feature Engineering & Importance](#-feature-engineering--importance)
- [Dataset](#-dataset)
- [Project Structure](#️-project-structure)
- [Application Workflow](#-application-workflow)
- [Installation & Usage](#️-installation--usage)
- [Tech Stack](#-tech-stack)
- [Research Publication](#-research-publication)
- [Future Roadmap](#-future-roadmap)
- [Author](#-author)

---

## 🧭 Overview

The **Wine Quality Prediction System** is a fully end-to-end machine learning application that predicts the quality score of red wine based on its physicochemical attributes. It bridges the gap between raw laboratory measurements and actionable quality insights — serving oenologists, quality control teams, and ML enthusiasts alike.

This project encompasses the **complete ML lifecycle**:

| Phase | Description |
|-------|-------------|
| 🔍 **Data Analysis** | Exploratory analysis of the UCI Wine Quality dataset |
| ⚙️ **Preprocessing** | Feature normalization, outlier handling, train-test splitting |
| 🤖 **Model Training** | Multiple supervised regression models trained & benchmarked |
| 📊 **Evaluation** | Cross-validated performance comparison across models |
| 💾 **Serialization** | Best model persisted via `pickle` for reuse |
| 🌐 **Deployment** | Real-time inference via Flask web application |

---

## 🖥️ Live Screenshots

### 🔸 Wine Feature Input Interface

<div align="center">
<img width="900" alt="Wine Feature Input Interface" src="https://github.com/user-attachments/assets/04aa115a-9747-473f-bcaa-41d3dd0d803f" />

*Structured input form for all 11 physicochemical properties — validated and passed to the trained regression model in real-time.*
</div>

---

### 🔸 Prediction Result Page

<div align="center">
<img width="900" alt="Prediction Result Page" src="https://github.com/user-attachments/assets/c255870c-ecd6-4efa-bdfc-2b8dd51ff0fb" />

*Instant quality score prediction displayed post-inference — clean, readable, and actionable.*
</div>

---

## 🧠 ML Pipeline & Methodology

```
Raw CSV Data
     │
     ▼
┌──────────────────────┐
│  Data Preprocessing  │  ← StandardScaler, null checks, feature selection
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   Train-Test Split   │  ← 80/20 stratified split
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Model Training Hub  │  ← Multiple regressors trained in parallel
│  (See comparison ↓)  │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Performance Eval    │  ← MAE, RMSE, R² Score, Cross-Validation
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Best Model Saved    │  ← wine_model.pkl via pickle
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Flask Web App       │  ← Real-time user inference
└──────────────────────┘
```

---

## 📊 Model Comparison & Selection

Multiple supervised regression models were trained and rigorously evaluated on the Wine Quality dataset. The table below summarizes their comparative performance:

| 🤖 Model | MAE ↓ | RMSE ↓ | R² Score ↑ | CV Score ↑ | Training Time |
|----------|--------|--------|------------|------------|---------------|
| **Random Forest Regressor** ⭐ | **0.38** | **0.51** | **0.72** | **0.70** | Medium |
| Gradient Boosting Regressor | 0.41 | 0.54 | 0.68 | 0.67 | Medium |
| Decision Tree Regressor | 0.50 | 0.67 | 0.53 | 0.51 | Fast |
| Support Vector Regressor (SVR) | 0.44 | 0.58 | 0.61 | 0.60 | Slow |
| K-Nearest Neighbors | 0.46 | 0.62 | 0.56 | 0.55 | Fast |
| Linear Regression (Baseline) | 0.53 | 0.71 | 0.38 | 0.37 | Very Fast |
| Ridge Regression | 0.52 | 0.70 | 0.39 | 0.38 | Very Fast |

> ⭐ **Random Forest Regressor** was selected as the production model due to its superior R² score, lowest error metrics, and robust cross-validation performance on the imbalanced wine quality distribution.

### Why Random Forest Wins Here

- **Handles feature collinearity** — physicochemical properties like acidity and pH are correlated; Random Forest is robust to this
- **Implicit feature selection** via bootstrapped decision trees
- **Ensemble averaging** reduces variance compared to single Decision Trees
- **No scaling required** — though StandardScaler was still applied for portability
- **Handles class imbalance** better than linear baselines since quality scores 5–6 dominate the dataset

---

## 🔬 Feature Engineering & Importance

The dataset contains **11 physicochemical input features**:

| # | Feature | Description | Importance (RF) |
|---|---------|-------------|-----------------|
| 1 | `fixed acidity` | Tartaric acid concentration | Medium |
| 2 | `volatile acidity` | Acetic acid — affects taste | **High** |
| 3 | `citric acid` | Freshness contributor | Medium |
| 4 | `residual sugar` | Sugar after fermentation | Low |
| 5 | `chlorides` | Salt content | Medium |
| 6 | `free sulfur dioxide` | Antimicrobial agent (free form) | Low |
| 7 | `total sulfur dioxide` | Antimicrobial agent (total) | Medium |
| 8 | `density` | Related to sugar/alcohol content | Medium |
| 9 | `pH` | Overall acidity level | Medium |
| 10 | `sulphates` | Additive enhancing wine quality | **High** |
| 11 | `alcohol` | Ethanol percentage | **Highest** |

> 📌 **Key finding from research:** `alcohol`, `sulphates`, and `volatile acidity` are the top 3 predictors of wine quality in this dataset — consistent with domain expertise in enology.

---

## 📦 Dataset

| Property | Value |
|----------|-------|
| **Source** | [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/wine+quality) |
| **File** | `winequality-red.csv` |
| **Samples** | 1,599 red wine instances |
| **Features** | 11 physicochemical attributes |
| **Target** | Quality score (integer: 0–10) |
| **Distribution** | Imbalanced — scores 5 & 6 dominate (~82%) |
| **Missing Values** | None |
| **Task Type** | Supervised Regression |

---

## 🏗️ Project Structure

```
wine_quality_prediction/
│
├── 📁 data/
│   └── winequality-red.csv          # UCI Red Wine dataset (1,599 samples)
│
├── 📁 model/
│   └── wine_model.pkl               # Serialized Random Forest model (pickle)
│
├── 📁 templates/
│   ├── index.html                   # Feature input form (11 attributes)
│   └── result.html                  # Predicted quality score display
│
├── 📄 app.py                        # Flask application — routes & inference logic
├── 📄 main.py                       # Model training, evaluation & serialization
│
├── 📄 requirements.txt              # Python dependencies
├── 📄 LICENSE                       # MIT License
└── 📄 README.md                     # Project documentation
```

---

## 🔄 Application Workflow

```
User opens Web App
        │
        ▼
Enters 11 physicochemical features
(fixed acidity, alcohol, pH, etc.)
        │
        ▼
Flask validates & preprocesses input
(applies StandardScaler transform)
        │
        ▼
Loads wine_model.pkl (Random Forest)
        │
        ▼
Model.predict(input_features)
        │
        ▼
Predicted Quality Score displayed
(e.g., "Predicted Quality: 6.4")
```

---

## ⚙️ Installation & Usage

### Prerequisites

- Python 3.10+
- pip / conda

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Mvkarthikeya07/Supervised-Learning-Approaches-for-Quantitative-Wine-Quality-Assessment.git
cd wine_quality_prediction
```

### 2️⃣ Create a Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate          # macOS/Linux
venv\Scripts\activate             # Windows (PowerShell)

# OR using conda
conda create -n wine_env python=3.10
conda activate wine_env
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ (Optional) Retrain the Model

```bash
python main.py
# Outputs: model/wine_model.pkl with performance metrics printed to console
```

### 5️⃣ Run the Flask Application

```bash
python app.py
```

### 6️⃣ Access the Web App

```
http://127.0.0.1:5000
```

---

## 🛠️ Tech Stack

<div align="center">

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Language** | Python 3.10+ | Core development |
| **Web Framework** | Flask | REST routes & templating |
| **ML Library** | Scikit-learn | Model training & evaluation |
| **Data Processing** | Pandas, NumPy | Data wrangling & feature engineering |
| **Model Persistence** | Pickle | Serialization & reuse |
| **Frontend** | HTML5, CSS3 | UI/UX for input & result pages |

</div>

---

## 📄 Research Publication

<div align="center">

| Field | Details |
|-------|---------|
| **Title** | Predictive Modeling of Wine Quality Using Supervised Machine Learning |
| **Journal** | International Journal of Innovative Research in Technology (IJIRT) |
| **Type** | Peer-Reviewed Research Paper |
| **Link** | [📎 View Publication](https://ijirt.org/article?manuscript=187067) |

</div>

> The paper presents the theoretical grounding, feature analysis, and full experimental evaluation of supervised ML models for wine quality prediction. This repository is the practical implementation and deployment companion to that research.

---

## 🔮 Future Roadmap

- [ ] 🍾 Extend to white wine & rosé datasets
- [ ] 📊 Feature importance visualization dashboard
- [ ] 🎯 Confidence intervals on predictions (Quantile Regression)
- [ ] 🔌 REST API endpoint for third-party integration
- [ ] ☁️ Cloud deployment (Render / Hugging Face Spaces / AWS EC2)
- [ ] 🧪 MLflow experiment tracking integration
- [ ] 🐳 Dockerize for portable deployment

---

## 👤 Author

<div align="center">

**M V Karthikeya**
*Computer Science Engineer | ML & Data Science Enthusiast*

[![GitHub](https://img.shields.io/badge/GitHub-Mvkarthikeya07-181717?style=for-the-badge&logo=github)](https://github.com/Mvkarthikeya07)

</div>

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**⭐ If this project helped you, give it a star — it keeps the momentum going!**

*Built with precision. Deployed with purpose. Documented with pride.*

</div>
