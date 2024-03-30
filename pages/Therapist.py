
import streamlit as st
from langchain.llms import OpenAI

st.title("AI Therapist")

openai_api_key = st.text_input("OpenAI API Key", type="password")

def generate_response(input_text):
    llm = OpenAI(temperature=0.7, openai_api_key=openai_api_key)
    st.info(llm(input_text))

with st.form("my_form"):
    text = st.text_area("Enter text:", "What are you feeling today?")
    submitted = st.form_submit_button("Submit")
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
    elif submitted:
        generate_response(text)

with st.expander("See API Key"):
    st.write(" OPEN AI API Key: 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")
