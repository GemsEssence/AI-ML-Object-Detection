import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt6.QtGui import QIcon, QFont, QFontDatabase, QPixmap, QImage
from PyQt6.QtCore import Qt
import tensorflow_hub as hub
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from image_detection_UI import Ui_Image_Detection  # Import the generated UI file

class Image_Detection(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.setup_connections()
        self.init_variables()


    def init_ui(self):
        # Set up the user interface
        self.ui = Ui_Image_Detection()
        self.ui.setupUi(self)


    def init_variables(self):
        # Load the model
        model_path = "tf_image_detection_model.h5"
        custom_objects = {'KerasLayer': hub.KerasLayer}
        self.model = load_model(model_path, custom_objects=custom_objects)

        # Define the categories
        self.categories = ["Animal", "Fruit", "Human", "Object", "Vegetable"]

        # Define the confidence threshold for predictions
        self.confidence_threshold = 0.70


    def setup_connections(self):
        self.ui.upload_img_btn.clicked.connect(self.load_image)
        self.ui.categories_btn.clicked.connect(self.show_pred_categories_page)
        self.ui.back_btn.clicked.connect(self.show_home_page)


    def load_image(self):
        # Open file dialog to select an image
        options = QFileDialog.Option.DontUseNativeDialog
        image_path, _ = QFileDialog.getOpenFileName(
            self, "Select an Image", "", "Images (*.png *.jpg *.jpeg *.bmp)", options=options
        )
        if image_path:
            # Load and display the image
            self.display_image(image_path)

            # Predict and display the category
            category, confidence = self.predict_category(image_path)
            self.ui.prediction_msg.setText(f"Predicted: {category} (Confidence: {confidence:.2f})")


    def display_image(self, image_path):
        # Read the image using OpenCV
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Convert the image to QImage
        height, width, channel = img.shape
        bytes_per_line = 3 * width
        q_image = QImage(img.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)

        # Display the image in the QLabel
        pixmap = QPixmap.fromImage(q_image)
        self.ui.image_label.setPixmap(pixmap.scaled(600, 400, Qt.AspectRatioMode.KeepAspectRatio))


    # Function to predict the category
    def predict_category(self, image_path):
        # Load and preprocess the image
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  
        img_array = img_array / 255.0  

        # Predict the category
        predictions = self.model.predict(img_array)
        predicted_index = np.argmax(predictions)
        predicted_category = self.categories[predicted_index]
        predicted_confidence = np.max(predictions)

        # If the confidence is below the threshold, classify as "Unknown"
        if predicted_confidence < self.confidence_threshold:
            predicted_category = "Unknown"

        return predicted_category, predicted_confidence


    def show_pred_categories_page(self):
        self.ui.stackedWidget.setCurrentWidget(self.ui.pred_category_page)


    def show_home_page(self):
        self.ui.stackedWidget.setCurrentWidget(self.ui.home_page)



if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon('icons/app_icon.svg'))
    window = Image_Detection()
    window.show()
    sys.exit(app.exec())
