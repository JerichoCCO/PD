import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Set page configuration without an icon
st.set_page_config(page_title="Plantar Pressure Distribution Classification System", layout="wide")

# Page title
st.write(
    "<div style='text-align: center; font-size: 50px;'>Welcome to Plantar Pressure Distribution Classification System</div>",
    unsafe_allow_html=True,
)

# Function to load and preprocess the image
def load_image(image_file):
    img = Image.open(image_file)
    img = img.resize((32, 32))
    img = np.array(img)
    if img.shape[-1] == 4:  # Check if the image has an alpha channel
        img = img[..., :3]  # Remove the alpha channel
    img = img.reshape(1, 32, 32, 3)
    img = img.astype('float32')
    img /= 255.0
    return img

# Function to make predictions
def predict(image, model, labels):
    img = load_image(image)
    result = model.predict(img)
    predicted_class = np.argmax(result, axis=1)
    return labels[predicted_class[0]]
