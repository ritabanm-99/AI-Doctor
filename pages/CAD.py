import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

st.markdown("# Coronary Artery Disease (CAD) Prediction System")

symptoms = st.multiselect(
    'What are your symptoms/conditions/history?',
    ['Obesity', 'Airway disease', 'Thyroid Disease', 'Current Smoker', 'EX-Smoker'])

Gender = st.multiselect('What is your gender?',['Male','Female'])
# Function to read and preprocess the dataset

@st.cache_data()
def load_data():
    cad_data = pd.read_csv("CAD.csv")
    # Preprocessing steps
    cad_data['Cath'] = cad_data['Cath'].map({'Cad': 1, 'Normal': 0})
    cad_data['Sex'] = cad_data['Sex'].map({'Male': 1, 'Female': 0})
    # Using .replace() with dictionary is fine, but ensure it's done once
    
    cad_data.replace({'Y': 1, 'N': 0}, inplace=True)
    feature_columns = ['Obesity', 'Airway disease', 'Sex', 'Thyroid Disease', 'BP', 'PR','Age', 'BMI', 'Current Smoker', 'EX-Smoker']
    return cad_data, feature_columns

# Use the function to load data
cad_data, feature_columns = load_data()
cad_data['Age'] = st.slider("Age",0,100,50)
cad_data['BMI'] = st.slider("BMI",0,50,20)
cad_data['BP'] = st.slider("BP", 0,200,20)
cad_data['PR'] = st.slider("PR", 0,400,100)

# Make sure all columns that are supposed to be numeric are converted correctly
for col in feature_columns:
    if cad_data[col].dtype == 'object':
        cad_data[col] = pd.to_numeric(cad_data[col], errors='coerce')

# Drop NaN values that might have been introduced during conversion
cad_data.dropna(inplace=True)

# Split the data into training and testing sets
X = cad_data[feature_columns]
y = cad_data['Cath']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Train the model
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# Prediction based on user input
if st.button("Predict"):
    if not symptoms:
        st.error("Please select at least one symptom.")
    else:
        # Transform user input into model input format
        input_vector = {feature: 1 if feature in symptoms else 0 for feature in feature_columns}
        input_df = pd.DataFrame([input_vector])

        # Perform the prediction
        prediction = logreg.predict(input_df)
        probability = logreg.predict_proba(input_df)[0, 1]

        # Display results
        st.success("Based on your symptoms, you are {} likely to have CAD.".format("very" if prediction[0] <= 1 else "less"))
        st.info(f"Probability of having CAD: {(probability)*100:.2f}%")
        st.info(f"Accuracy of model: {logreg.score(X_test, y_test)*100:.2f}%")

# Optionally, add code to display the head of the dataset
# st.dataframe(cad_data.head())
