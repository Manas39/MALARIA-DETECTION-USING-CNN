import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import pickle
from io import BytesIO  
import streamlit as st
import pandas as pd

# Load model and label encoder
model = load_model('Malaria/mala1.h5')
with open('Malaria/label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

# Define class labels
class_labels = {0: 'Parasitized', 1: 'Uninfected'}

# Function to predict
def predict(img):
    img_3d = img.reshape(-1, 80, 80, 3) / 255.
    prediction = model.predict(img_3d)[0]
    predicted_label = class_labels[np.argmax(prediction)]
    return predicted_label, prediction

# Streamlit configuration
st.set_page_config(
    page_title="Malaria Detection App",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="collapsed",  # Hide the sidebar by default
)

# Custom styles for the app
st.markdown(
    """
    <style>
        .big-title {
            font-size: 3em;
            color: #333;
        }
        .upload-section {
            margin-top: 20px;
        }
        .predict-button {
            font-size: 1.5em;
            padding: 10px 20px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        .predict-button:hover {
            background-color: #45a049;
        }
        .result-section {
            margin-top: 30px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Main app layout
st.title("Malaria Detection Web App 🦠")
st.markdown("Welcome to the Malaria Detection App. Upload an image to check if it is infected with malaria.", unsafe_allow_html=True)

# Upload section
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png","jfif"], key="upload_file")
predict_button = st.button("Predict", key="predict_button", help="Click to make a prediction")

# Main app logic
if uploaded_file is not None:
    # Display uploaded image with fixed width
    st.image(uploaded_file, caption="Uploaded Image", width=400)

    # Make prediction
    if predict_button:
        with st.spinner("Making prediction..."):
            image_data = BytesIO(uploaded_file.read())

            # Convert the image to a numpy array
            img = image.img_to_array(image.load_img(image_data, target_size=(80, 80)))

            # Make prediction using your custom function
            predicted_label, prediction_prob = predict(img)

            # Display prediction result and probabilities to the right of the image
            st.markdown("<hr class='result-section' />", unsafe_allow_html=True)
            st.header("Prediction Result:")
            st.markdown(f"The image is classified as: **{predicted_label}**", unsafe_allow_html=True)

            # Display prediction probabilities in a table
            st.markdown("Prediction Probabilities:")
            prob_normalized = prediction_prob / np.sum(prediction_prob)  # Normalize probabilities to sum to 1
            prob_table = pd.DataFrame({"Class": [class_labels[label] for label in le.classes_], "Probability": prob_normalized})
            st.table(prob_table)