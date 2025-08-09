# streamlit_app.py

import streamlit as st
from PIL import Image
import os

from utils import load_model, transform_image, predict

st.set_page_config(page_title="Vehicle Classifier", layout="centered")
st.title("🚗 Vehicle Image Classifier")
st.write("Upload an image of a vehicle (Car, Motorcycle, Truck, Van) to classify it.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Classifying..."):
        model_path = os.path.join(os.path.dirname(__file__), 'vehicle_classifier.pth')
        model = load_model(model_path)
        img_tensor = transform_image(image)
        prediction = predict(model, img_tensor)

    st.success(f"🧠 Prediction: **{prediction}**")
