# Image Detection with TensorFlow

This project is a Python-based image detection desktop application that uses TensorFlow and OpenCV to classify images into predefined categories. It leverages a pre-trained model and processes input images to predict their category with confidence scores.

## Features

- Classifies images into categories such as `Animal`, `Fruit`, `Human`, `Object`, and `Vegetable`.
- Displays images with predictions and confidence scores.
- Configurable confidence threshold for accurate results.
- Pre-trained TensorFlow model integration using TensorFlow Hub.

## Installation

1. **Clone this repository**:
   ```bash
   git clone https://github.com/GemsEssence/AI-ML-Object-Detection
   ```

2. **Create and activate a virtual environment**:

   - **On Windows**:
     ```bash
     python -m venv env
     env\Scripts\activate.bat
     ```

   - **On macOS and Linux**:
     ```bash
     python3 -m venv env
     source env/bin/activate
     ```

3. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Use Python 3.10 or 3.11**:
   Ensure that Python 3.10 or 3.11 is installed and used for this project to maintain compatibility with the dependencies.

## Usage

1. Ensure your TensorFlow model is saved as `tf_image_detection_model.h5` in the project directory.
2. Run the main script to classify the image:
   ```bash
   python main.py
   ```
3. Check what all category images can be predicted by clicking on `Predict Category` button.
3. Upload the image using `Upload Image` button.
4. The script will display the image with the predicted category and confidence score. If the confidence is below the threshold, the category will be labeled as `Unknown`.

## Project Structure

- **main.py**: Contains the main logic for initializing the app, setting up the UI, and integrating the model prediction. This file is the entry point of the project.
- **Object_Detection.ipynb**: Jupyter Notebook for interactive experimentation and visualization.
- **requirements.txt**: Lists all required Python libraries and their versions.
- **image_detection_UI.ui**: The PyQt6 UI design file used to define the GUI layout.
- **image_detection_UI.py**: Generated Python code from the .ui file, containing the GUI implementation logic.

## Requirements

The project depends on the following libraries:

- TensorFlow
- OpenCV
- Matplotlib
- Numpy
- TensorFlow Hub

For a complete list of dependencies, refer to the `requirements.txt` file.

## Example Output

When running the script, you will see the classified image displayed along with the prediction details. For instance:

- **Predicted Category**: Animal
- **Confidence**: 0.85

## Notes

- Make sure the input image is clear and suitable for classification.
- Adjust the `confidence_threshold` variable in the script to fine-tune classification sensitivity.

