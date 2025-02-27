# streamlit_app.py
import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import requests
import tempfile
import os

# ----------------- Classification Models -----------------

def download_model(model_url):
    """
    Downloads the model from the specified URL and saves it to a temporary file.
    
    Returns:
        str: Path to the downloaded temporary model file.
    """
    response = requests.get(model_url)
    response.raise_for_status()  # Ensure the request was successful
    temp_dir = tempfile.gettempdir()
    model_filename = os.path.basename(model_url)
    temp_path = os.path.join(temp_dir, model_filename)
    with open(temp_path, 'wb') as f:
        f.write(response.content)
    return temp_path

@st.cache_resource
def load_cnn_model(model_url):
    from scripts.cnn_model import SimpleCNN
    model_path = download_model(model_url)
    model = SimpleCNN(num_classes=3)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

@st.cache_resource
def load_alexnet_model(model_url):
    from torchvision import models
    import torch.nn as nn
    model_path = download_model(model_url)
    model = models.alexnet(pretrained=False)
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, 3)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# ----------------- YOLOv8 Model -----------------

# Uncomment and modify if you decide to use YOLOv8 in the future
# @st.cache_resource
# def load_yolov8_model(model_path):
#     from ultralytics import YOLO  # Import moved here
#     model = YOLO(model_path)
#     return model

# ----------------- Load All Models -----------------

def load_all_models():
    base_url = "https://rpalphastorage.blob.core.windows.net/models/"
    cnn_url = base_url + 'cnn_best.pth'
    alexnet_url = base_url + 'alexnet_best.pth'
    # yolov8_url = base_url + 'yolov8_best.pt'
    
    cnn_model = load_cnn_model(cnn_url)
    alexnet_model = load_alexnet_model(alexnet_url)
    # yolov8_model = load_yolov8_model(yolov8_url)
    
    # return cnn_model, alexnet_model, yolov8_model
    return cnn_model, alexnet_model

# ----------------- Helper Functions -----------------

def preprocess_image(image, img_size=224):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet means
                             std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)

def get_classification_prediction(model, image):
    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
        return preds.item()

# Mapping class indices to labels
CLASS_NAMES = ['unripe', 'ripe', 'overripe']

# ----------------- Streamlit UI -----------------

def main():
    st.set_page_config(page_title="🍌 Banana Classifier & Detector", layout="wide")
    st.title("🍌 Banana Classification and Detection Web App")

    # Load models
    with st.spinner("Loading models..."):
        # cnn_model, alexnet_model, yolov8_model = load_all_models()
        cnn_model, alexnet_model = load_all_models()
    st.success("Models loaded successfully!")

    # Image upload
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("")
        st.write("Classifying and Detecting...")

        # Convert image for classification
        image_for_classification = preprocess_image(image)

        # CNN Prediction
        cnn_pred = get_classification_prediction(cnn_model, image_for_classification)
        cnn_label = CLASS_NAMES[cnn_pred]

        # AlexNet Prediction
        alexnet_pred = get_classification_prediction(alexnet_model, image_for_classification)
        alexnet_label = CLASS_NAMES[alexnet_pred]

        # # YOLOv8 Prediction
        # yolov8_results = yolov8_model(image)
        # yolov8_annotated = yolov8_results[0].plot()

        # Display Classification Results
        st.write("### Classification Results")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Standard CNN:**", cnn_label)
        with col2:
            st.write("**AlexNet:**", alexnet_label)

        # # Display YOLOv8 Results
        # st.write("### YOLOv8 Detection")
        # st.image(yolov8_annotated, caption='YOLOv8 Detection', use_column_width=True)

        # Optionally, display detection tables
        # st.write("### YOLOv8 Detection Results")
        # df_yolov8 = yolov8_results[0].pandas().xyxy[0]
        # st.dataframe(df_yolov8)

if __name__ == "__main__":
    main()
