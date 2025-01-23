# Data Mining 

This project aims to detect fraudulent transactions using machine learning techniques and feature engineering. The dataset includes numerical, categorical, and text data. Advanced preprocessing methods, such as TF-IDF feature extraction, were employed to extract relevant features from the text data. Following this, feature selection techniques, including embedded approaches like Lasso and Random Forest importance, were used to identify the most significant characteristics, enhancing the model's predictive accuracy. Multiple machine learning models, including SVM and KNN, were then trained to classify transactions as fraudulent or legitimate.

![image](https://github.com/user-attachments/assets/9cebadab-d6ca-49d8-90ef-b18d746bd2bf)

## Table of Contents
- [Project Description](#project-description)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Data Preprocessing](#data-preprocessing)
- [Feature Engineering](#feature-engineering)
- [Modeling](#modeling)
- [Results](#results)

## Features
 - *Data Cleaning and Preprocessing*:
    - Removal of irrelevant columns (e.g., Credit_card_number, Expiry).
    - Encoding categorical features (e.g., Profession) using LabelEncoder.
    - Normalization of numerical columns (e.g., Income, Security_code).

- *Feature Extraction with TF-IDF*:
    - Text-based descriptions were processed using TF-IDF to extract the most relevant textual features.
  
- *Feature Selection*:
    - Feature importance ranking using SelectKBest with the Chi-Square test.
    - Lasso regression for feature selection based on coefficient magnitude.
    - Random Forest importance ranking to identify significant features.
  
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
   - *Lasso and Random Forest*: Techniques for feature selection and importance ranking.

## Data Preprocessing
 1. *Dataset Cleaning*:
      - Dropped irrelevant columns: Credit_card_number and Expiry.
      - Checked for missing values and handled them appropriately.

 2. *Encoding Categorical Data*:
      - Encoded the Profession column using LabelEncoder.

 3. *Scaling*:
      - Normalized the Income and Security_code columns using Min-Max scaling to bring them into the [0,1] range.

## Feature Engineering
 1. *Text Feature Extraction with TF-IDF*:
      - A Description column was programmatically generated to simulate textual data.
      - Applied TF-IDF to extract the top 100 textual features.
        
 2. *Feature Selection*:
      - Selected the most important features using SelectKBest with the Chi-Square test.
      - Lasso regression for feature selection based on coefficient magnitude.
      - Random Forest importance ranking to identify significant features.
   

## Modeling
 - *Machine Learning Models*:
      - SVM and KNN were implemented to classify transactions as fraudulent or legitimate.

 - *Evaluation Metrics*:
      - Accuracy: Measure of overall correctness.
      - Precision: Ratio of true positives to predicted positives.
      - Recall: Ability to find all relevant cases.
      - F1-Score: Balance between precision and recall.

 - *Confusion Matrices*:
      - Visualized confusion matrices for a better understanding of the classification results.

## Results
 - *Key Metrics*:
      - SVM: Accuracy = XX%, Precision = XX%, Recall = XX%, F1-Score = XX%.
      - KNN: Accuracy = XX%, Precision = XX%, Recall = XX%, F1-Score = XX%.
        
Detailed results and visualizations are included in the project notebook.
