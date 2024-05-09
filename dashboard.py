import streamlit as st
import numpy as np
import pandas as pd
import cv2
import joblib
from skimage.feature import graycomatrix, graycoprops
import altair as alt
import plotly.express as px
import matplotlib.pyplot as plt

def create_fruit_mask(fruit_type, image):
    # Convert the image to HSV (Hue, Saturation, Value)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    lower_orange = np.array([15, 40, 80])
    upper_orange = np.array([100, 255, 255])

    lower_yellow = np.array([10, 10, 80])
    upper_yellow = np.array([60, 255, 255])

    lower_red = np.array([0, 40, 70])
    upper_red = np.array([360, 255, 255])

    # Create masks based on the defined ranges
    mask_orange = cv2.inRange(hsv_image, lower_orange, upper_orange)
    mask_yellow = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
    mask_red = cv2.inRange(hsv_image, lower_red, upper_red)

    if fruit_type == "apple":
        final_mask = mask_red
    elif fruit_type == "banana":
        final_mask = mask_yellow
    else:
        final_mask = mask_orange

    # Apply morphological operations to refine the mask
    kernel = np.ones((5, 5), np.uint8)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)

    # Visualize the mask with the original image
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax[0].set_title("Original Image")
    ax[0].axis('off')
    
    ax[1].imshow(final_mask, cmap='gray')
    ax[1].set_title("Mask Visualization")
    ax[1].axis('off')
    st.pyplot(fig)

    return final_mask

# Function to extract image features
def extract_features(fruit_type, image):

    # Get the mask for fruit regions
    mask = create_fruit_mask(fruit_type, image)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Get the total number of pixels
    total_pixels = image.shape[0] * image.shape[1]

    # Calculate the maximum possible sum
    max_possible_sum = total_pixels * 255

    # Mask the RGB channels
    red_channel = image[:, :, 2]
    green_channel = image[:, :, 1]
    blue_channel = image[:, :, 0]

    # Sum values only where the mask is valid (i.e., where the pixel is part of the fruit)
    red_sum = np.sum(red_channel[mask > 0])
    green_sum = np.sum(green_channel[mask > 0])
    blue_sum = np.sum(blue_channel[mask > 0])

    # Normalize and scale up
    scale_factor = 5 * 10**5
    red_cumulative = int((red_sum / max_possible_sum) * scale_factor)
    green_cumulative = int((green_sum / max_possible_sum) * scale_factor)
    blue_cumulative = int((blue_sum / max_possible_sum) * scale_factor)

    # Calculate contrast (picture intensity) - mean brightness in the grayscale image)
    gray_fruit = gray_image #[mask > 0]
    contrast = np.mean(gray_fruit)

    # Calculate GLCM properties for texture analysis
    glcm = graycomatrix(gray_fruit, distances=[1], angles=[0], symmetric=True, normed=True)

    homogeneity = float(graycoprops(glcm, 'homogeneity')[0, 0])

    return [red_cumulative, green_cumulative, blue_cumulative, contrast, homogeneity]

# Function to load and preprocess the image
def process_image(fruit_type, uploaded_image):
    # Decode image using OpenCV
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    image = cv2.resize(image, (256, 256))

    features = extract_features(fruit_type, image)

    return features

def make_donut(probability_value, input_text, input_color):
    if input_color == 'green':
        chart_color = ['#27AE60', '#E0F4E8']  # Green tones
    elif input_color == 'red':
        chart_color = ['#E74C3C', '#F4E0E0']  # Red tones
    else:
        raise ValueError("Invalid input color. Must be 'green' or 'red'.")

    # Create the data for the chart
    data = pd.DataFrame({
        'Topic': [input_text, ''],
        '% value': [probability_value, 100 - probability_value]
    })

    # Create the donut chart
    plot = alt.Chart(data).mark_arc(innerRadius=45, cornerRadius=25).encode(
        theta=alt.Theta(field='% value', type='quantitative'),
        color=alt.Color(field='Topic', type='nominal',
                        scale=alt.Scale(domain=[input_text, ''], range=chart_color),
                        legend=None)
    ).properties(width=130, height=130)

    # Add a central text inside the donut chart
    central_text = alt.Chart(pd.DataFrame({'value': [f'{probability_value:.2f}%']})).mark_text(
        align='center',
        color="black",
        font="Lato",
        fontSize=20,
        fontWeight=700,
        fontStyle="italic"
    ).encode(text='value:N')

    # Add a label below the donut chart
    label = alt.Chart(pd.DataFrame({'label': [input_text]})).mark_text(
        align='center',
        font="Lato",
        fontSize=15,
        fontWeight=500
    ).encode(text='label:N').properties(width=130)

    # Concatenate donut chart and the label vertically
    final_chart = alt.vconcat(plot + central_text, label).resolve_scale(color='independent')

    return final_chart

# Load the pre-trained XGBoost model and Random Forest model

xgb_model = joblib.load("best_xgb_model_{'gamma': 0.1, 'learning_rate': 0.1, 'max_depth': 15, 'n_estimators': 300, 'subsample': 0.8}.pkl")

rf_model = joblib.load("best_rf_model_{'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}.pkl")

# Streamlit App Layout

with st.sidebar:
    st.title('Fruits Freshness Classifier')
    model_list = ["XG Boost (Default)", "Random Forest"]
    selected_model = st.selectbox('Select a pre-trained model', model_list, index=0)

st.title(f"Fruit Freshness Classifier using {selected_model}")

st.subheader("Image-based Fruit Freshness Classifier")
fruit_list = ["apple","banana","orange"]
selected_fruit_type = st.selectbox('Select the type of fruit:', fruit_list, index=0)

st.write("Upload an image to analyze if the fruit is fresh or rotten.")

# Upload Image Function
uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

# Predict and Display Results
if uploaded_image is not None:

    # Extract features and make predictions
    feature_values = process_image(selected_fruit_type, uploaded_image)

    feature_names = ["Red", "Green", "Blue", "Contrast", "Homogeneity"]
    
    # Create a dictionary and convert the values to a NumPy array
    feature_dict = dict(zip(feature_names, feature_values))
    features_array = np.array(list(feature_dict.values()))

    # Reshape to ensure it's a 2D array with shape (1, 5)
    features_array = features_array.reshape(1, -1)

    if selected_model == model_list[0]:
        model = xgb_model
    else:
        model = rf_model

    prediction_probabilities = model.predict_proba(features_array)[0]

    donut_chart_fresh = make_donut(prediction_probabilities[1] * 100, 'Fresh', 'green')
    donut_chart_rotten = make_donut(prediction_probabilities[0] * 100, 'Rotten', 'red')

    st.write("The uploaded fruit is predicted to be:")

    combined_chart = alt.hconcat(donut_chart_fresh, donut_chart_rotten)

    # Use columns to center the combined chart in the Streamlit page
    col_center = st.columns([4, 8, 4])[1]  # Adjust the ratio for centering

    with col_center:
        st.altair_chart(combined_chart, use_container_width=True)
    # st.altair_chart(donut_chart_fresh | donut_chart_rotten)

    # Display Prediction Result
    # st.write(f"The uploaded fruit is predicted to be **{prediction_label}**.")

    # Display extracted feature values
    feature_names = ["Red", "Green", "Blue", "Contrast", "Homogeneity"]
    feature_values = features_array[0]
    features_data = pd.DataFrame({
        "Feature": feature_names,
        "Value": [f"{val:.4f}" if isinstance(val, float) else str(val) for val in feature_values]  # Formatting values
    })

    # Display the DataFrame as a table in Streamlit
    st.write("Extracted Features:")
    st.table(features_data)

st.subheader("Manual Fruit Freshness Classifier")
st.write("Manually enter feature values to analyze if the fruit is fresh or rotten.")

manual_selected_fruit_type = st.selectbox('Select the type of fruits:', fruit_list, index=0)

# Input fields for each feature
red_sum = st.number_input("Red")
green_sum = st.number_input("Green")
blue_sum = st.number_input("Blue")
contrast = st.number_input("Contrast", format="%.8f")
homogeneity = st.number_input("Homogeneity", format="%.8f")

if st.button("Predict Freshness"):

    # Extract features and make predictions
    manual_features = [red_sum, green_sum, blue_sum, contrast, homogeneity]

    feature_names = ["Red", "Green", "Blue", "Contrast", "Homogeneity"]
    
    # Create a dictionary and convert the values to a NumPy array
    feature_dict = dict(zip(feature_names, manual_features))
    features_array = np.array(list(feature_dict.values()))

    features_array = features_array.reshape(1, -1)

    if selected_model == model_list[0]:
        model = xgb_model
    else:
        model = rf_model
    prediction_probabilities = model.predict_proba(features_array)[0]

    donut_chart_fresh = make_donut(prediction_probabilities[1] * 100, 'Fresh', 'green')
    donut_chart_rotten = make_donut(prediction_probabilities[0] * 100, 'Rotten', 'red')

    st.write("The uploaded fruit is predicted to be:")

    st.altair_chart(donut_chart_fresh | donut_chart_rotten)

    # Display Prediction Result
    # st.write(f"The uploaded fruit is predicted to be **{prediction_label}**.")

    # Display extracted feature values
    feature_names = ["Red", "Green", "Blue", "Contrast", "Homogeneity"]
    feature_values = features_array[0]
    features_data = pd.DataFrame({
        "Feature": feature_names,
        "Value": [f"{val:.4f}" if isinstance(val, float) else str(val) for val in feature_values]  # Formatting values
    })

    # Display the DataFrame as a table in Streamlit
    st.write("Extracted Features:")
    st.table(features_data)