# fruits-freshness-classifier

[Dashboard Link](https://fruits-freshness-classifier.streamlit.app/) 

# Executive Summary

Problem Statement:

- It is difficult to identify the freshness of each of the fruits when the number of fruits is in large volume
- It is important to monitor the freshness of the fruits from time to time to ensure good customer satisfaction and retention

Scope:

- The dataset covers fruits specifically on apples, bananas, and oranges

Definitions:
- Total RGB Color Composition: The cumulative measurement of color distribution across the red, green, and blue channels within each image.
   - Red: Total Colour on the Red Channel
   - Green: Total colour on the Green Channel
   - Blue: Total colour on the Blue Channel
- Picture Intensity (Contrast): A component of texture analysis, indicating the degree of brightness or darkness in the image.
- Overall Energy: A measure of the aggregate visual intensity and vibrancy present in the image.
- Pixel Correlation: An assessment of the interdependence and relationship between adjacent pixels in the image.
- Homogeneity: An evaluation of the uniformity and consistency of visual elements throughout the image.

Methodology:

- Identify whether the fruit is fresh or rotten based on the RGB
- Use Accuracy, Precision, Recall (Sensitivity), F1, MCC and Kappa to evaluate the classification performance of the models
- Use RMSE, MSE, MAE to evaluate the regression performance of the models