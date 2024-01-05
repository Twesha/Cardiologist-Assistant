import streamlit as st
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

from util import *

import warnings
warnings.filterwarnings('ignore')

# Define labels for your classes
labels_dict = {
    0: 'History of MI',
    1: 'Myocardial Infarction',
    2: 'Normal',
    3: 'Arrhythmia'
}

# Dictionary mapping model names to their file paths
model_paths = {
    'AlexNet': r'saved_models\alex_net_ecg_model.hd5',
    'LeNet5': r'saved_models/lenet5_ecg_model.hd5',
    "VGG-16": r'saved_models\vgg16_ecg_model.hd5',
    "GoogleNet": r"saved_models\googlenet_ecg_model.hd5",
    "ResNet": r'saved_models\resnet_ecg_model.hd5'
    # Add more models as needed
}

lifeStylePath = "lifestylechanges.xlsx"
lifeStyle_df = pd.read_excel(lifeStylePath) 

def load_model(model_path):
    # Load the model
    tf.keras.backend.clear_session()
    model = tf.keras.models.load_model(model_path)
    return model

def preprocess_image(img_path):
    # Load and preprocess the image
    uploaded_file = image_full_path()
    img = image.load_img(img_path, target_size=(512, 512))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize pixel values to be between 0 and 1
    return img_array

def predict_image(model, img_array):
    # Make a prediction
    predictions = model.predict(img_array)
    return predictions

def main():
    st.title("ECG Classification")

    # Allow the user to choose the model
    selected_model = st.selectbox("Select Model", list(model_paths.keys()))

    # Load the selected model
    print(f"Loading model: {selected_model}")
    model_path = model_paths[selected_model]
    model = load_model(model_path)

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")


    if uploaded_file is not None:
        # Display the uploaded image
        
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        # Preprocess the image
        
        img_array = preprocess_image(uploaded_file)

        # Make predictions
        predictions = predict_image(model, img_array)

        # Decode predictions
        top_class = np.argmax(predictions)
        top_confidence = np.max(predictions)

        # Display the top prediction
        st.subheader("Top Prediction:")
        st.subheader(f"Class            :      {labels_dict[top_class]}")
        

        if labels_dict[top_class] == 'Arrhythmia':
            st.subheader(f"Lifestyle Changes   :      {lifeStyle_df.loc[1, 'LIFESTYLE CHANGES']}")
        elif labels_dict[top_class] == 'Myocardial Infarction':
            st.subheader(f"Lifestyle Changes   :      {lifeStyle_df.loc[0, 'LIFESTYLE CHANGES']}")



if __name__ == "__main__":
    main()