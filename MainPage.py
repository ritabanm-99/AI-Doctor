import streamlit as st
import pandas as pd
import numpy as np

#Title
st.image("Ai doctor.png", width = 150)
st.markdown(" # AI Doctor")



st.write("""This tool has been designed as an
alternative to WebMD and provides a simple UI 
to make predictions about CAD and Stroke in patients based on their symptoms. Additionally, the tool also has features for an LLM therapist.""")

st.write("""This tool has been developed for a course project and is not intended to be used for medical purposes.""")

st.write(""" Main Features: """)
st.write(""" 1. Coronary Artery Disease (CAD) Prediction System""")
st.write("""2. Stroke Prediction Prediction System""")
st.write("""3. Therapist Recommendation System""")


#Note for the therapist feature
st.write("To test the therapist feature, kindly shoot me an email at ritabanm@buffalo.edu to access the OpenAI API key if you don't have one. ")
