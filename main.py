import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import tensorflow as tf

# Load the pre-trained model
model = load_model('mnist.h5')

# Function to preprocess the image
def preprocess_image(image):
    # Convert image to grayscale
    gray = tf.image.rgb_to_grayscale(image)
    # Resize image to 28x28
    resized = tf.image.resize(gray, [28, 28])
    # Reshape image to match model input shape
    reshaped = tf.reshape(resized, (1, 28, 28, 1))
    # Normalize pixel values
    normalized = reshaped / 255.0
    return normalized

# Streamlit UI
st.title("MNIST Digit Recognition")

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption='Uploaded Image.', use_column_width=True)

    # Preprocess the image
    image = tf.image.decode_image(uploaded_image.read(), channels=3)
    processed_image = preprocess_image(image)

    # Make prediction
    prediction = model.predict(processed_image)
    digit = np.argmax(prediction)

    st.write("Predicted Digit:", digit)

    # Plot the processed image
    # plt.imshow(tf.squeeze(processed_image), cmap='gray')
    # plt.axis('off')
    # st.pyplot()
