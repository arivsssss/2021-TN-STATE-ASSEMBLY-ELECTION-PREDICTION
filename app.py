import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import LabelBinarizer

#########################################
# 1. DATA LOADING & PREPROCESSING SETUP #
#########################################

# Load dataset
df = pd.read_csv("tnelectionsformlmodels.csv")

# Select the relevant columns
data = df[['Sex', 'Party', 'Age', 'Electors',
           'Constituency_Type', 'District_Name', 'Sub_Region', 'No_of_Candidates',
           'ENOP', 'Contested', 'Turncoat', 'Incumbent', 'Recontest',
           'Education_Qualification', 'Main_Profession', 'Second_Profession',
           'Result', 'Alliance', 'Votes']]

# Clean data: remove rows where Party is "NOTA" and convert Result to string
data = data[data['Party'] != 'NOTA']
data['Result'] = data['Result'].astype(str)

# Binarize the classification target
lb_style = LabelBinarizer()
y_result = lb_style.fit_transform(data['Result']).ravel()

# Prepare features by dropping the targets ('Result' and 'Votes')
X = data.drop(columns=['Result', 'Votes'])
# One-hot encode features using drop_first=True to mimic training
X_encoded = pd.get_dummies(X, drop_first=True)
# Save the training columns so user input can be aligned
training_columns = X_encoded.columns

#########################################
# 2. STREAMLIT SIDEBAR: USER INPUT      #
#########################################

st.title("Election Prediction App")
st.sidebar.header("Enter Feature Values")

# Define original features
features = ['Sex', 'Party', 'Age', 'Electors', 'Constituency_Type',
            'District_Name', 'Sub_Region', 'No_of_Candidates', 'ENOP',
            'Contested', 'Turncoat', 'Incumbent', 'Recontest',
            'Education_Qualification', 'Main_Profession', 'Second_Profession',
            'Alliance']

# Collect user inputs for each feature
user_input = {}
for feature in features:
    if df[feature].dtype == object:
        # For categorical features, get unique values. For Party, use filtered data.
        if feature == "Party":
            unique_vals = sorted(data[feature].dropna().unique().tolist())
        else:
            unique_vals = sorted(df[feature].dropna().unique().tolist())
        user_input[feature] = st.sidebar.selectbox(f"{feature}", unique_vals)
    else:
        # For numeric features, use number_input with a range and default value (median)
        min_val = float(df[feature].min())
        max_val = float(df[feature].max())
        default_val = float(df[feature].median())
        user_input[feature] = st.sidebar.number_input(f"{feature}", min_value=min_val, max_value=max_val, value=default_val)

st.write("### User Input Summary")
st.write(user_input)

# Convert user input to DataFrame and one-hot encode using the same settings as training
user_df = pd.DataFrame([user_input])
user_encoded = pd.get_dummies(user_df, drop_first=True)
# Reindex to ensure the same columns as the training data, filling missing ones with 0
user_encoded = user_encoded.reindex(columns=training_columns, fill_value=0)

#########################################
# 3. MODEL LOADING & PREDICTION          #
#########################################

# Specify the absolute folder path for your models
models_folder = r"E:\tn_election_2021\election_models"

# Create a "Predict" button so that predictions are only run on demand
if st.button("Predict"):

    st.write("## Classifier Predictions (Result)")
    # Classifier models have filenames containing '_f1_'
    classifier_files = [f for f in os.listdir(models_folder) if f.endswith(".pkl") and "_f1_" in f]

    if classifier_files:
        for file in classifier_files:
            model_path = os.path.join(models_folder, file)
            model = joblib.load(model_path)
            # Get prediction (assumes classifier outputs 0/1)
            pred = model.predict(user_encoded)[0]
            # Map numeric prediction back to original label using the LabelBinarizer
            predicted_label = lb_style.classes_[int(pred)]
            # Parse the filename to extract model name and F1 score
            try:
                parts = file.split("_f1_")
                model_name = parts[0].replace("_", " ")
                f1_score_val = parts[1].replace(".pkl", "")
            except Exception:
                model_name = file
                f1_score_val = "N/A"
            st.write(f"**{model_name} (F1: {f1_score_val})**: Predicted Result = {predicted_label}")
    else:
        st.write("No classifier models found.")

    st.write("## Regressor Predictions (Votes)")
    # Regressor models have filenames containing '_r2_'
    regressor_files = [f for f in os.listdir(models_folder) if f.endswith(".pkl") and "_r2_" in f]

    if regressor_files:
        for file in regressor_files:
            model_path = os.path.join(models_folder, file)
            model = joblib.load(model_path)
            # Predict vote count
            pred_votes = model.predict(user_encoded)[0]
            # Parse filename to extract model name and R2 score
            try:
                parts = file.split("_r2_")
                model_name = parts[0].replace("_", " ")
                r2_score_val = parts[1].replace(".pkl", "")
            except Exception:
                model_name = file
                r2_score_val = "N/A"
            st.write(f"**{model_name} (R2: {r2_score_val})**: Predicted Votes = {pred_votes:.2f}")
    else:
        st.write("No regressor models found.")
