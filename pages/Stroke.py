import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

st.markdown("# Stroke Prediction System")

# Function to read and preprocess the dataset
@st.cache_data()
def load_data():
    data = pd.read_csv('data-stroke.csv')
    # Assuming 'Yes'/'No' are in columns that need to be binary encoded
    data = data.replace({'Yes': 1, 'No': 0})
    data = data.dropna()
    
    return data

data = load_data()

# Feature selection for the form
feature_columns = ['hypertension', 'heart_disease', 'ever_married', 'age', 'avg_glucose_level', 'bmi']
symptoms = st.multiselect('Select your symptoms/conditions:', feature_columns[:-3])
age = st.slider("Age", 0, 100, 50)
avg_glucose_level = st.slider("Average Glucose Level", 0, 300, 100)
bmi = st.slider("BMI", 10, 50, 25)

# Make sure all columns that are supposed to be numeric are converted correctly
data[feature_columns] = data[feature_columns].apply(pd.to_numeric, errors='coerce')

# Train-test split
X = data[feature_columns]
y = data['stroke']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Initialize and train the logistic regression model
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# Prediction based on user input
if st.button("Predict"):
    # Creating an input vector based on user input
    input_data = {feature: 0 for feature in feature_columns}
    input_data.update({
        'hypertension': 1 if 'hypertension' in symptoms else 0,
        'heart_disease': 1 if 'heart_disease' in symptoms else 0,
        'ever_married': 1 if 'ever_married' in symptoms else 0,
        'age': age,
        'avg_glucose_level': avg_glucose_level,
        'bmi': bmi
    })
    input_vector = pd.DataFrame([input_data], columns=feature_columns)
    
    # Perform the prediction
    prediction = logreg.predict(input_vector)
    probability = logreg.predict_proba(input_vector)[0, 1]
    
    # Display results
    st.success(f"Based on your input, you are {'more' if prediction[0] == 1 else 'less'} likely to have a stroke.")
    st.info(f"Probability of having a stroke: {probability * 100:.2f}%")
    st.info(f"Accuracy of model: {logreg.score(X_test, y_test) * 100:.2f}%")



