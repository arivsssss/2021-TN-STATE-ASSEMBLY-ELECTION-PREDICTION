# Tamil Nadu Elections 2021 Analysis & Machine Learning Project

This repository presents a comprehensive study of the Tamil Nadu Elections 2021 data through both data visualization and machine learning. The project is divided into two primary parts:

1. **Data Visualization & Exploratory Data Analysis (EDA):**  
   - Cleaning and preprocessing the election dataset.
   - Visualizing key metrics such as vote distributions, candidate demographics, and turnout patterns.
   - Engineering features like _Non_Voters_, _Non_Voters_Percentage_, and _Invalid_Votes_percentage_ to better understand voting dynamics.

2. **Machine Learning Models:**  
   - Developing several predictive models including XGBoost (with feature selection and as a classifier), K-Nearest Neighbors, Logistic Regression, and Decision Tree Classifier.
   - Evaluating model performance using accuracy, ROC-AUC, F1-score, confusion matrices, and cross-validation.
   - Comparing model results to derive insights on predicting election outcomes.

The insights and methods in this project are derived from detailed analyses and model experiments as outlined in the case study documents :contentReference[oaicite:0]{index=0} and :contentReference[oaicite:1]{index=1}.

## Table of Contents

- [Overview](#overview)
- [Data Description](#data-description)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Machine Learning Models](#machine-learning-models)
  - [XGBoost Models](#xgboost-models)
  - [K-Nearest Neighbors](#k-nearest-neighbors)
  - [Logistic Regression](#logistic-regression)
  - [Decision Tree Classifier](#decision-tree-classifier)
- [Results & Findings](#results--findings)
- [Requirements](#requirements)
- [Setup & Usage](#setup--usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Overview

This project explores the dynamics of the Tamil Nadu Elections 2021 by integrating detailed EDA with multiple machine learning approaches. The data visualization segment focuses on understanding the election data through cleaning, transformation, and plotting, while the machine learning segment develops and evaluates several predictive models to forecast election outcomes.

## Data Description

The dataset contains information on candidates and constituencies including:
- **Candidate Details:** Name, Sex, Age, Party, Position, and Votes.
- **Electoral Metrics:** Valid_Votes, Electors, Turnout_Percentage, and more.
- **Engineered Features:** Non_Voters, Non_Voters_Percentage, Invalid_Votes_percentage, and additional indicators (e.g., Winner/Lost status).

## Exploratory Data Analysis (EDA)

Key steps in the EDA process include:
- **Data Cleaning:**  
  - Removing columns with uniform values.
  - Handling null values (e.g., in the _Candidate_Type_ and _Party_Type_TCPD_ columns).
- **Feature Engineering:**  
  - Calculating additional metrics such as the number of non-voters and vote discrepancies.
- **Data Quality Checks:**  
  - Validating if candidates’ votes are within valid limits.
  - Checking consistency of key variables across the dataset.
- **Visualization:**  
  - Plotting distributions of votes, ages, and turnout percentages to reveal trends.

## Machine Learning Models

A series of models were implemented and evaluated on the election dataset after preprocessing and encoding categorical variables.

### XGBoost Models

- **XGBoost with Feature Selection:**  
  Used to identify important predictors with high performance metrics (accuracy and ROC-AUC scores above 97%).
- **XGBoost Classifier:**  
  Evaluated using cross-validation and confusion matrices to verify its robustness.

### K-Nearest Neighbors

- Applied on the encoded dataset, KNN faced challenges with class imbalance, reflected in lower ROC-AUC scores and a skewed confusion matrix.

### Logistic Regression

- Provided baseline performance with convergence warnings suggesting further data scaling may be required.
- Metrics indicated high accuracy for the majority class but lower F1-scores for the minority class.

### Decision Tree Classifier

- Offered a tree-based approach with intuitive splits.
- Model evaluation included confusion matrices and cross-validation to understand classification boundaries.

## Results & Findings

- **Data Cleaning & EDA:**  
  The removal of redundant columns and the engineering of new features led to clearer insights into voter behavior and candidate performance.
- **Model Performance:**  
  Among the models, XGBoost consistently achieved the best predictive performance, while simpler models like KNN struggled with imbalanced class distributions.
  ![image](https://github.com/user-attachments/assets/65269872-3505-4e1c-93a1-34a11cf5b0d4)

- **Output:**  
  Results on every model when a new foreign data is given.

![image](https://github.com/user-attachments/assets/5ed55be0-8291-4066-8dc3-3f1259e33238)
![image](https://github.com/user-attachments/assets/851d28df-ed5e-41e7-bcb3-7398d8e94d71)
![image](https://github.com/user-attachments/assets/757b3d0f-b143-4d1a-9983-391ca91ef02f)
![image](https://github.com/user-attachments/assets/b35c0a93-e8da-4d1b-aff1-7266146bb188)


  

## Requirements

- Python 3.x
- [pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)
- [scikit-learn](https://scikit-learn.org/)
- [XGBoost](https://xgboost.readthedocs.io/)
- [Matplotlib](https://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)

Install the required packages using:
```bash
pip install -r requirements.txt
