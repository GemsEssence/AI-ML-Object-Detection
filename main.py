import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow_hub as hub

# Load the model
model_path = "tf_image_detection_model.h5"
custom_objects = {'KerasLayer': hub.KerasLayer}
model = load_model(model_path, custom_objects=custom_objects)

# Define the categories
categories = ["Animal", "Fruit", "Human", "Object", "Vegetable"]

# Define the confidence threshold for predictions
confidence_threshold = 0.70  # Set the threshold (e.g., 70%)

# Function to predict the category and display the image with a rectangle
def predict_and_display(image_path):
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(224, 224))  # Match your model's input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize to range [0, 1]

    # Predict the category
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    predicted_category = categories[predicted_index]
    predicted_confidence = np.max(predictions)  # Get the confidence for the prediction

    # If the confidence is below the threshold, classify as "Unknown"
    if predicted_confidence < confidence_threshold:
        predicted_category = "Unknown"

    # Load the original image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Display the image with prediction and rectangle
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.title(f"Predicted: {predicted_category} (Confidence: {predicted_confidence:.2f})", fontsize=16)
    plt.axis("off")
    plt.show()

    return predicted_category, predicted_confidence

# Test with an image
image_path = "tiger.jpeg"
category, confidence = predict_and_display(image_path)
print(f"The predicted category is: {category} with confidence {confidence:.2f}")
