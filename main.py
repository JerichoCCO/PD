import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Plantar Pressure Distribution Classification System", layout="wide")

st.write(
    "<div style='text-align: center; font-size: 50px;'>Welcome to Plantar Pressure Distribution Classification System</div>",
    unsafe_allow_html=True,
)

def load_image(image_file):
    img = Image.open(image_file)
    img = img.resize((299, 299))  # Resize to (299, 299)
    img = np.array(img)
    if img.shape[-1] == 4:
        img = img[..., :3]
    img = img.reshape(1, 299, 299, 3)  # Match model input shape
    img = img.astype('float32')
    img /= 255.0
    return img

def predict(image, model, labels):
    img = load_image(image)
    result = model.predict(img)
    predicted_class = np.argmax(result, axis=1)
    return labels[predicted_class[0]]

try:
    model = load_model('xception_model_frozen_stabilized_2.h5')
except Exception as e:
    st.error(f"Error loading the model: {e}")

def load_labels(filename):
    try:
        with open(filename, 'r') as file:
            labels = file.readlines()
        labels = [label.strip() for label in labels]
        return labels
    except FileNotFoundError:
        st.error(f"Labels file '{filename}' not found.")
        return []

st.title("Foot Arch Classification")
test_image = st.file_uploader("Choose an Image:", type=["jpg", "jpeg", "png"])
if test_image is not None:
    try:
        st.image(test_image, width=300, caption="Uploaded Image")
        labels = load_labels("labels.txt")
        if st.button("Predict"):
            st.write("Predicting...")
            if labels:
                predicted_category = predict(test_image, model, labels)
                st.success(f"Predicted Foot Arch Category: {predicted_category}")
    except Exception as e:
        st.error(f"Error processing the uploaded image: {e}")
