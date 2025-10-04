# A World Away: Hunting for Exoplanets with AI

## 🚀 Clesp Co.

### 🧑‍🤝‍🧑 Team

-   **Burak** – Machine Learning, Python, Web (HTML + CSS + JS)
-   **Ali** – Machine Learning, Python, Web Development
-   **Eren** – AI Engineering, Prompt Engineering

### 🛰️ Project Overview

This repository contains our submission for the **NASA Space Apps Challenge 2025**. Our project is an AI-driven pipeline designed to detect and classify exoplanets using real astronomical data from three major NASA missions. Within the 48-hour hackathon, we aimed to complete the full cycle of data collection, model training, and web demo development.

### 📊 Datasets Used

We utilized publicly available datasets from three cornerstone NASA missions, which were processed and unified into a single machine learning-ready format:

-   **Kepler Objects of Interest (KOI)**
-   **TESS Objects of Interest (TOI)**
-   **K2 Planet Candidates (K2)**

### 🧪 Step-by-Step Technical Pipeline

Our entire technical process was implemented on the Kaggle platform.

**1. Library Imports**
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import warnings; warnings.filterwarnings('ignore')
```

**2. Data Loading & Unification**
- We loaded the three datasets and created a unified binary target column, is_planet, where 1 represents a confirmed planet or candidate and 0 represents a false positive.
```python
df_koi['is_planet'] = df_koi['koi_disposition'].apply(lambda x: 1 if x in ['CONFIRMED', 'CANDIDATE'] else 0)
df_toi['is_planet'] = df_toi['tfopwg_disp'].apply(lambda x: 1 if x in ['PC', 'KP'] else 0)
df_k2['is_planet'] = df_k2['disposition'].apply(lambda x: 1 if x in ['CONFIRMED', 'CANDIDATE'] else 0)
```

**3. Feature Selection & Merging**
- We selected key physical features common across all missions (e.g., orbital period, planet radius, stellar temperature) and standardized their column names. The datasets were then merged and cleaned.
```python
df_final = pd.concat([df_koi_final, df_toi_final, df_k2_final], ignore_index=True)
df_final.dropna(inplace=True)
df_final = df_final[df_final['planet_name'].notna()]
df_final.to_csv('final_planet_dataset_with_names.csv', index=False)
```

**4. Model Training and Evaluation**
- The final dataset was split into training and testing sets. We trained and evaluated three powerful classification models to find the best performer.
```python
models = {
    "LightGBM": lgb.LGBMClassifier(random_state=42),
    "RandomForest": RandomForestClassifier(random_state=42),
    "XGBoost": xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
}


for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"{name} Accuracy: {accuracy_score(y_test, y_pred):.4f}")
```

**5. Model Saving**
- The best-performing model was serialized and saved using joblib for deployment.
```python
joblib.dump(best_model_instance, 'planet_classifier.joblib')
```

**🌐 Deployment & Web Demo**
- We are developing a web interface for real-time classification. Users will be able to input transit data for a celestial object and receive a prediction from our model, indicating whether it's a potential exoplanet and with what confidence score.

## Frontend: HTML, CSS, JavaScript

## Backend (Planned): Flask API serving the planet_classifier.joblib model.

## Deployment: GitHub Pages + Render

**🛠️ Technologies**
- Data Science: Python, Pandas, Scikit-learn

- Machine Learning: LightGBM, XGBoost, RandomForest

- Platform: Kaggle Notebooks

- Web: Flask, HTML, CSS, JavaScript

## 🚀 Acknowledgements
- Thanks to NASA, Kaggle, and the Space Apps Challenge 2025 for making real-world science accessible and fostering global collaboration.

### 🛰️ Hackathon Experience

In NASA Space Apps 2025's 48-hour marathon:

1.  Data collection and preprocessing,
2.  Model design and training,
3.  Web demo development.

We will complete the steps.

### 📜 License

This project is licensed under the MIT License.
