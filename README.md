# 🍌 Banana Classification and Detection System

A comprehensive machine learning project for classifying and detecting bananas in images using multiple deep learning models, including a Standard CNN, AlexNet, YOLOv8, and Detectron2. The system is deployed as a user-friendly web application using Streamlit, allowing users to upload images and receive classification and detection results in real-time.

## 📈 Table of Contents

- [📚 Overview](#overview)
- [🚀 Features](#features)
- [🔧 Project Structure](#project-structure)
- [🛠️ Setup and Installation](#setup-and-installation)
  - [1. Clone the Repository](#1-clone-the-repository)
  - [2. Create a Virtual Environment](#2-create-a-virtual-environment)
  - [3. Install Dependencies](#3-install-dependencies)
  - [4. Set Up Detectron2](#4-set-up-detectron2)
- [📂 Data Preparation](#data-preparation)
  - [1. Organize the Original Dataset](#1-organize-the-original-dataset)
  - [2. Data Augmentation](#2-data-augmentation)
    - [A. Classification Data Augmentation](#a-classification-data-augmentation)
    - [B. Detection Data Augmentation](#b-detection-data-augmentation)
- [🤖 Model Training](#model-training)
  - [1. Standard CNN](#1-standard-cnn)
  - [2. AlexNet](#2-alexnet)
  - [3. YOLOv8](#3-yolov8)
  - [4. Detectron2](#4-detectron2)
- [🗂️ Saving Trained Models](#saving-trained-models)
- [🌐 Deploying with Streamlit](#deploying-with-streamlit)
  - [1. Running the Streamlit App Locally](#1-running-the-streamlit-app-locally)
  - [2. Deploying to Streamlit Cloud](#2-deploying-to-streamlit-cloud)
- [📄 Requirements](#requirements)
- [📜 License](#license)
- [🙌 Contributing](#contributing)
- [📞 Contact](#contact)

---

## 📚 Overview

This project aims to develop a robust system for classifying and detecting bananas in images. Leveraging multiple deep learning models ensures high accuracy and reliability across various scenarios. The system is designed to be easily deployable as a web application, providing users with an intuitive interface for real-time image analysis.

## 🚀 Features

- **Image Classification**: Classify bananas into three categories:
  - **Unripe**
  - **Ripe**
  - **Overripe**

- **Object Detection**: Detect and locate bananas within any uploaded image.

- **Multiple Models**: Utilize different deep learning architectures to ensure accuracy and performance:
  - **Standard CNN**
  - **AlexNet**
  - **YOLOv8**
  - **Detectron2**

- **Web Deployment**: Deploy the system as a web application using Streamlit, enabling easy access and usability.

## 🔧 Project Structure

```
project/
├── dataset/
│   ├── original/
│   │   ├── unripe/
│   │   ├── ripe/
│   │   └── overripe/
│   ├── augmented/
│   │   ├── images/
│   │   │   ├── unripe/
│   │   │   ├── ripe/
│   │   │   └── overripe/
│   │   └── labels/
│   │       ├── unripe/
│   │       ├── ripe/
│   │       └── overripe/
│   ├── yolov8/
│   │   ├── images/
│   │   │   ├── train/
│   │   │   ├── val/
│   │   │   └── test/
│   │   └── labels/
│   │       ├── train/
│   │       ├── val/
│   │       └── test/
│   └── detectron2/
│       ├── annotations_train.json
│       └── annotations_val.json
├── models/
│   ├── cnn_best.pth
│   ├── alexnet_best.pth
│   ├── yolov8_best.pt
│   └── detectron2_output/
│       ├── model_final.pth
│       └── ...
├── scripts/
│   ├── data_augmentation.py
│   ├── detection_data_augmentation.py
│   ├── cnn_data_loader.py
│   ├── cnn_model.py
│   ├── cnn_train.py
│   ├── alexnet_train.py
│   ├── yolov8_train.py
│   ├── convert_yolo_to_coco.py
│   ├── detectron2_register.py
│   └── detectron2_train.py
├── streamlit_app.py
├── requirements.txt
└── setup_project_structure.ps1
```

> **Note:** 
> - `dataset/`: Contains all data-related folders, including original and augmented datasets for classification and detection.
> - `models/`: Stores trained model files.
> - `scripts/`: Holds all Python scripts for data augmentation, model training, and dataset conversion.
> - `streamlit_app.py`: The Streamlit web application script.
> - `requirements.txt`: Lists all project dependencies.
> - `setup_project_structure.ps1`: PowerShell script to automate project structure setup.

---

## 🛠️ Setup and Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/banana-classification-detection.git
cd banana-classification-detection
```

### 2. Create a Virtual Environment

It's recommended to use a virtual environment to manage dependencies.

```bash
# Install virtualenv if not already installed
pip install virtualenv

# Create a virtual environment named 'env'
virtualenv env

# Activate the virtual environment
# On Windows:
.\env\Scripts\activate

# On Unix or MacOS:
source env/bin/activate
```

### 3. Install Dependencies

Install the core dependencies listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

### 4. Set Up Detectron2

**Detectron2** requires specific installation steps based on your system's CUDA and PyTorch versions. Follow the instructions below based on your setup.

#### A. Determine Your System's Configuration

- **Python Version**: Ensure you're using Python 3.8 or higher.
- **CUDA Version**: Check your installed CUDA version by running:

  ```bash
  nvcc --version
  ```

  *If you don't have CUDA installed or are unsure, you can opt for a CPU-only installation.*

#### B. Install Detectron2 Separately

Based on your CUDA and PyTorch versions, run the appropriate command.

- **For CUDA 11.7 and PyTorch 2.0**

  ```bash
  pip install detectron2==0.6 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu117/torch2.0/index.html
  ```

- **For CUDA 11.6 and PyTorch 1.13**

  ```bash
  pip install detectron2==0.6 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu116/torch1.13/index.html
  ```

- **For CPU-Only Installation**

  ```bash
  pip install detectron2==0.6 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu0/index.html
  ```

> **Note:** Replace `0.6` with the latest compatible version of Detectron2 if available. Always refer to the [official Detectron2 installation guide](https://detectron2.readthedocs.io/en/latest/tutorials/install.html) for the most accurate instructions.

#### C. Verify the Installation

```bash
python -c "import detectron2; print(detectron2.__version__)"
```

If no errors occur and the version prints out, the installation was successful.

---

## 📂 Data Preparation

### 1. Organize the Original Dataset

Ensure your original dataset is organized into three categories: `unripe`, `ripe`, and `overripe`, each containing 1,000 images.

```
dataset/
├── original/
│   ├── unripe/
│   ├── ripe/
│   └── overripe/
```

### 2. Data Augmentation

Data augmentation enhances the diversity of your dataset, improving model robustness.

#### A. Classification Data Augmentation

Run the `data_augmentation.py` script to augment classification images.

```powershell
# Navigate to the scripts directory
cd scripts

# Execute the data augmentation script
python data_augmentation.py
```

#### B. Detection Data Augmentation

Run the `detection_data_augmentation.py` script to augment detection images and labels.

```powershell
# Ensure you have labeled data in YOLO format
# Execute the detection data augmentation script
python detection_data_augmentation.py
```

> **Note:** Ensure that your labels are in YOLO format (`class_id x_center y_center width height`). Adjust the `bbox_params` in the script if your format differs.

---

## 🤖 Model Training

Train each model using the provided scripts.

### 1. Standard CNN

Train a simple Convolutional Neural Network for classification.

```powershell
cd scripts
python cnn_train.py
```

> **Note:** This script trains the Standard CNN using the augmented classification dataset. The best model will be saved as `models/cnn_best.pth`.

### 2. AlexNet

Fine-tune a pre-trained AlexNet model for classification.

```powershell
cd scripts
python alexnet_train.py
```

> **Note:** This script trains the AlexNet model using the augmented classification dataset. The best model will be saved as `models/alexnet_best.pth`.

### 3. YOLOv8

Train the YOLOv8 model for object detection.

```powershell
cd scripts
python yolov8_train.py
```

> **Note:** This script trains the YOLOv8 model using the augmented detection dataset. The best model will be saved as `models/yolov8_best.pt`.

### 4. Detectron2

Train the Detectron2 model for object detection.

```powershell
cd scripts
python detectron2_register.py
python detectron2_train.py
```

> **Note:** Ensure that the Detectron2 dataset is properly registered before training. The trained model will be saved in `models/detectron2_output/`.

---

## 🗂️ Saving Trained Models

After training each model, ensure that you have the best-performing model saved for deployment.

- **Standard CNN**: Saved as `models/cnn_best.pth`
- **AlexNet**: Saved as `models/alexnet_best.pth`
- **YOLOv8**: Saved as `models/yolov8_best.pt`
- **Detectron2**: Saved in `models/detectron2_output/`

Ensure all these files are correctly stored in the `models/` directory.

---

## 🌐 Deploying with Streamlit

We'll create a Streamlit web application that allows users to upload an image and get predictions from all four models.

### 1. Running the Streamlit App Locally

```powershell
# Ensure you're in the project root directory
cd ..

# Activate the virtual environment if not already activated
.\env\Scripts\activate

# Run the Streamlit app
streamlit run streamlit_app.py
```

This command will launch the web app in your default browser. You can upload an image, and the app will display the detected bananas with their classifications.

### 2. Deploying to Streamlit Cloud

Once your Streamlit app is working locally, you can deploy it to a cloud platform for wider accessibility.

#### Steps:

1. **Push Your Code to GitHub**:
   - Create a GitHub repository and push your project files (`streamlit_app.py`, `models/`, etc.).

2. **Set Up `requirements.txt`**:
   Ensure all dependencies are listed in `requirements.txt`.

3. **Deploy**:
   - Go to [Streamlit Cloud](https://streamlit.io/cloud) and sign in.
   - Click on "New app" and select your GitHub repository.
   - Specify the branch and the `streamlit_app.py` file.
   - Click "Deploy."

> **Note:** Large model files like `yolov8_best.pt` might exceed free tier limits on Streamlit Cloud. In such cases, consider using cloud storage services (like AWS S3) to host the model and load it at runtime or explore other hosting options.

---

## 📄 Requirements

Below is the comprehensive `requirements.txt` file tailored to the project needs:

```plaintext
# Core Libraries
torch==2.0.0
torchvision==0.15.1
torchaudio==2.0.1
numpy==1.24.2
pandas==1.5.3
matplotlib==3.7.1
scikit-learn==1.2.2
opencv-python==4.7.0.72
Pillow==9.5.0
albumentations==1.3.0

# Deep Learning Models
ultralytics==8.0.0  # For YOLOv8

# Web Application
streamlit==1.25.0

# Utilities
tqdm==4.65.0

# Detectron2 Installation Placeholder
# Detectron2 requires a specific installation command based on your CUDA and PyTorch versions.
# Please install Detectron2 separately following the instructions below.
```

> **Important Note on Detectron2 Installation**

Detectron2 is a sophisticated library that requires compatibility between its version, your system's **CUDA** version, and your installed **PyTorch** version. Due to these dependencies, it's recommended **not** to include Detectron2 directly in the `requirements.txt`. Instead, follow the steps below to install it appropriately.

### 1. Determine Your System's Configuration

- **Python Version**: Ensure you're using a compatible Python version (typically Python 3.8 or higher).
- **CUDA Version**: Check your installed CUDA version. You can verify this by running:

  ```bash
  nvcc --version
  ```

  *If you don't have CUDA installed or are unsure, Detectron2 also offers CPU-only installations.*

### 2. Install Detectron2 Separately

Use the following commands based on your CUDA version. Replace the CUDA version in the URL with the one matching your system.

#### A. For CUDA 11.7 and PyTorch 2.0

```bash
pip install detectron2==0.6 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu117/torch2.0/index.html
```

#### B. For CUDA 11.6 and PyTorch 1.13

```bash
pip install detectron2==0.6 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu116/torch1.13/index.html
```

#### C. For CPU-Only Installation

```bash
pip install detectron2==0.6 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu0/index.html
```

> **Note:** Replace `0.6` with the latest compatible version of Detectron2 if a newer version is available. Always refer to the [official Detectron2 installation guide](https://detectron2.readthedocs.io/en/latest/tutorials/install.html) for the most accurate and up-to-date instructions.

### 3. Verify the Installation

After installation, verify that Detectron2 is correctly installed by running:

```python
import detectron2
print(detectron2.__version__)
```

If no errors occur and the version prints out, the installation was successful.

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 🙌 Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. **Fork the Project**
2. **Create your Feature Branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit your Changes** (`git commit -m 'Add some AmazingFeature'`)
4. **Push to the Branch** (`git push origin feature/AmazingFeature`)
5. **Open a Pull Request**

---

## 📞 Contact

Your Name - [@yourtwitter](https://twitter.com/yourtwitter) - your.email@example.com

Project Link: [https://github.com/yourusername/banana-classification-detection](https://github.com/yourusername/banana-classification-detection)

---

> **Happy Coding! 🍌🚀**
```