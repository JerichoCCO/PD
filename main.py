import streamlit as st
import numpy as np
from PIL import Image
# from tensorflow.keras.models import load_model  # Commenting out for now

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
    # result = model.predict(img)  # Commenting out for now
    # predicted_class = np.argmax(result, axis=1)  # Commenting out for now
    # return labels[predicted_class[0]]  # Commenting out for now
    return "Prediction functionality is disabled."

# Load the model
# try:
#     model = load_model('vehicle.h5')  # Commenting out for now
# except Exception as e:
#     st.error(f"Error loading the model: {e}")

# Function to load labels from a text file
def load_labels(filename):
    try:
        with open(filename, 'r') as file:
            labels = file.readlines()
        labels = [label.strip() for label in labels]
        return labels
    except FileNotFoundError:
        st.error(f"Labels file '{filename}' not found.")
        return []

# Add a file uploader and prediction logic
st.title("Model Prediction")
test_image = st.file_uploader("Choose an Image:")
if test_image is not None:
    st.image(test_image, width=300, caption="Uploaded Image")
    if st.button("Predict"):
        # Check if model is loaded (functionality disabled for now)
        st.write("Predicting... (Functionality Disabled)")
        labels = load_labels("labels.txt")
        if labels:
            predicted_category = predict(test_image, None, labels)
            st.success(f"Predicted Vehicle Category: {predicted_category}")
