# Part 2: Loading the Pre-trained Model and Providing User Interface

# Import necessary libraries
import os
import numpy as np
import cv2
import keras
from keras.models import load_model
from tkinter import Tk, filedialog, messagebox 
from PIL import Image, ImageFile  # Importing necessary classes from PIL to handle images # For file selection dialog and pop-up message box
from scipy.stats import entropy
import nibabel as nib  # Handle .nii and .nii.gz files
import pydicom  # Library to handle .dcm files

# Define constants
img_size = 224  # Resize images to 224x224 for EfficientNetB0
categories = ["brain", "heart", "breast", "limbs", "unknown"]  # Organ categories

# Load the Pre-trained Model
organ_model_path = "fine_tuned_organ_classification_model.keras"

if not os.path.exists(organ_model_path):
    raise FileNotFoundError("Model file not found. Please ensure the model is trained and saved correctly.")

model = load_model(organ_model_path)

# Predict the Organ from User-uploaded Image
def predict_new_image():
    # Open file dialog to select an image
    Tk().withdraw()  # Hide the root window
    filepath = filedialog.askopenfilename(title="Select an image file", filetypes=[("Image Files", "*.jpg *.jpeg *.png *.nii *.nii.gz *.dcm")])
    if filepath:
        if filepath.endswith(('.jpg', '.jpeg', '.png')):
            # Load and preprocess the selected image (jpg, jpeg, png)
            try:
                img = Image.open(filepath).convert('L')  # Load image in grayscale mode
                resized_array = img.resize((img_size, img_size))
                prepared_image = np.stack((np.array(resized_array),) * 3, axis=-1).reshape(-1, img_size, img_size, 3) / 255.0  # Normalize and add batch dimension
            except OSError as e:
                messagebox.showerror("Error", f"Unable to read image file {filepath}. Skipping...")
                return
        elif filepath.endswith(('.nii', '.nii.gz')) and not filepath.startswith("._"):
            # Load and preprocess the selected .nii or .nii.gz image
            try:
                img = nib.load(filepath).get_fdata()
                mid_slice = img[:, :, img.shape[2] // 2]  # Select middle slice
                resized_array = cv2.resize(mid_slice, (img_size, img_size))
                prepared_image = np.stack((resized_array,) * 3, axis=-1).reshape(-1, img_size, img_size, 3) / 255.0  # Normalize and add batch dimension
            except Exception as e:
                messagebox.showerror("Error", f"Error loading file {filepath}: {e}")
                return
        elif filepath.endswith('.dcm'):
            # Load and preprocess the selected .dcm image
            try:
                dicom = pydicom.dcmread(filepath)
                img = dicom.pixel_array
                resized_array = cv2.resize(img, (img_size, img_size))
                prepared_image = np.stack((resized_array,) * 3, axis=-1).reshape(-1, img_size, img_size, 3) / 255.0  # Normalize and add batch dimension
            except Exception as e:
                messagebox.showerror("Error", f"Error loading file {filepath}: {e}")
                return
        else:
            messagebox.showerror("Error", "Unsupported file format.")
            return
        
        # Make prediction
        prediction = model.predict(prepared_image)
        predicted_category = categories[int(np.argmax(prediction))]
        confidence_score = np.max(prediction)
        prediction_entropy = entropy(prediction[0])

        # If the confidence is low or the entropy is high, consider it as not an organ
        if confidence_score < 0.7:  # Example threshold for uncertainty
            messagebox.showinfo("Prediction Result", "The model is uncertain. The uploaded image might not be an organ.")
        else:
            messagebox.showinfo("Prediction Result", f"Predicted Organ: {predicted_category}")
    else:
        messagebox.showinfo("Information", "No file selected.")

# Run the prediction function when the script is executed
if __name__ == "__main__":
    predict_new_image()
