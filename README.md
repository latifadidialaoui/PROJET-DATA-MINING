# Data Mining 

This project aims to detect fraudulent transactions using machine learning techniques and feature engineering. The dataset includes numerical, categorical, and text data. Advanced preprocessing methods, such as TF-IDF feature extraction, normalization, and categorical encoding, were applied to prepare the data for model training. Multiple machine learning models, including SVM and KNN, were trained to classify transactions as fraudulent or legitimate.

## Table of Contents
- [Project Description](#project-description)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Data Preprocessing](#data-preprocessing)
- [Feature Engineering](#feature-engineering)
- [Modeling](#modeling)
- [Results](#results)
- [How to Use](#how-to-use)

## Features
 - *Data Cleaning and Preprocessing*:
    - Removal of irrelevant columns (e.g., Credit_card_number, Expiry).
    - Encoding categorical features (e.g., Profession) using LabelEncoder.
    - Normalization of numerical columns (e.g., Income, Security_code).

- *Feature Extraction with TF-IDF*:
    - Text-based descriptions were processed using TF-IDF to extract the most relevant textual features.
  
- *Feature Selection*:
    - Feature importance ranking using SelectKBest with the Chi-Square test.
  
- *Machine Learning Models*:
    - Support Vector Machine (SVM).
    - K-Nearest Neighbors (KNN).
  
- *Evaluation*:
    - Classification metrics such as accuracy, precision, recall, and F1-score.
    - Visualization of confusion matrices for detailed insights.
 
## Technologies Used
   - *Python*: Core programming language for data processing and modeling.
   - *Jupyter Notebook*: Interactive development environment.
   - *Pandas*: Data manipulation and analysis.
   - *NumPy*: Numerical computations.
   - *Scikit-learn*: Machine learning algorithms, preprocessing, and evaluation.
   - *Matplotlib and Seaborn*: Data visualization.
   - *TF-IDF Vectorizer*: Feature extraction for text data.

## Data Preprocessing
 1. *Dataset Cleaning*:
   - Dropped irrelevant columns: Credit_card_number and Expiry.
   - Checked for missing values and handled them appropriately.

 2. *Encoding Categorical Data*:
   - Encoded the Profession column using LabelEncoder.

 3. *Scaling*:
   - Normalized the Income and Security_code columns using Min-Max scaling to bring them into the [0,1] range.

