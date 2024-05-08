import streamlit as st
import numpy as np
import cv2
import joblib
from skimage.feature import graycomatrix, graycoprops
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Function to extract image features
def extract_features(image):
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # RGB values (mean across each channel)
    red_mean = np.mean(image[:, :, 2])
    green_mean = np.mean(image[:, :, 1])
    blue_mean = np.mean(image[:, :, 0])

    # Texture features using GLCM (Gray-Level Co-Occurrence Matrix)
    glcm = graycomatrix(gray_image, distances=[1], angles=[0], symmetric=True, normed=True)

    contrast = graycoprops(glcm, 'contrast')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]

    return [red_mean, green_mean, blue_mean, contrast, energy, correlation, homogeneity]

# Function to load and preprocess the image
def process_image(uploaded_image):
    # Decode image using OpenCV
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Resize if needed (for standardization)
    image = cv2.resize(image, (256, 256))  # Resize to your required dimension

    # Extract features from the image
    features = extract_features(image)

    # Normalize the features
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform([features])

    return normalized_features

# Load the pre-trained XGBoost model
model = joblib.load("best_xgb_model_{'learning_rate': 0.1, 'max_depth': 10, 'n_estimators': 300, 'subsample': 0.8}.pkl")
model

# Streamlit App Layout
st.title("Fruit Freshness Classifier")
st.write("Upload an image to analyze if the fruit is fresh or rotten.")

# Image Upload
uploaded_image = st.file_uploader("Upload Image of the Fruit", type=["jpg", "jpeg", "png"])

# Predict and Display Results
if uploaded_image is not None:
    # Extract features and make predictions
    features = process_image(uploaded_image)
    prediction = model.predict(features)[0]
    prediction_label = "Fresh" if prediction == 1 else "Rotten"

    # Display Prediction Result
    st.write(f"The uploaded fruit is predicted to be **{prediction_label}**.")

    # Show uploaded image
    st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)

    # Display extracted feature values
    st.write("Extracted Features:")
    feature_names = ["Red", "Green", "Blue", "Contrast", "Energy", "Correlation", "Homogeneity"]
    feature_values = features[0]
    for name, value in zip(feature_names, feature_values):
        st.write(f"{name}: {value:.4f}")
