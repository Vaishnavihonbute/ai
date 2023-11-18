# app.py

import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('heart_disease_prediction_model.pkl')

def predict_heart_disease(features):
    return model.predict([features])[0]

def main():
    st.title('Heart Disease Prediction Web App')

    # Collect user input for prediction
    age = st.slider('Age', min_value=20, max_value=80, value=40)
    sex = st.radio('Gender', ['Female', 'Male'])
    # Add other input fields based on your features

    # Map selected gender to 0 (Female) or 1 (Male)
    sex_mapping = {'Female': 0, 'Male': 1}
    sex_encoded = sex_mapping[sex]

    # Create a dictionary with user input
    user_input = {
        'age': age,
        'sex': sex_encoded,
        # Add other features
    }

    # Display user input
    st.subheader('User Input:')
    st.write(user_input)

    # Make prediction
    if st.button('Predict Heart Disease'):
        prediction = predict_heart_disease(user_input)
        st.subheader('Prediction:')
        st.write(f'The predicted heart disease risk is: {prediction}')

if __name__ == '__main__':
    main()
