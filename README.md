# CNN-Based Computer Vision Models: Art Generation, Object Detection, Face Recognition, and Image Segmentation**

## Overview

This repository contains four advanced computer vision models developed using Convolutional Neural Networks (CNNs) as part of the Deep Learning Specialization by Andrew Ng. Each model demonstrates the wide-ranging capabilities of CNNs, from creative applications like art generation to practical implementations such as real-time object detection, face recognition, and medical image segmentation.

### Project Contents:**
1. Art Generation using Neural Style Transfer
2. Autonomous Driving: Car Detection using YOLO
3. Face Recognition using CNNs
4. Image Segmentation using U-Net Architecture

These models were built using deep learning libraries, including TensorFlow and Keras, and tackle various real-world challenges in computer vision. The models show how CNNs can be applied to different domains, demonstrating expertise in deep learning, computer vision, and model deployment.

---

## 1. Art Generation using Neural Style Transfer

This model applies Neural Style Transfer** to create new images by combining the content of one image with the style of another. The approach allows for the merging of artistic style and content using a pre-trained CNN model, such as VGG-19, to capture the deep representations of both content and style.

### Key Features
- Utilizes Convolutional Neural Networks (CNNs to extract features from both content and style images.
- Minimizes the content loss** and style loss to achieve visually appealing results.
- Transforms the content image to adopt the stylistic features of the chosen style image.

### Applications
- Creative industries for generating artistic content.
- Image and media editing tools for artists and designers.

---

## 2. Autonomous Driving: Car Detection using YOLO

This project implements the **YOLO (You Only Look Once)** algorithm for real-time object detection in images, specifically focusing on detecting cars for autonomous driving applications. YOLO is a state-of-the-art object detection model that divides images into a grid and predicts bounding boxes and class probabilities for each grid cell in a single pass.

### Key Features
- YOLO algorithm allows for efficient, real-time detection of multiple objects in an image.
- Utilizes bounding boxes for object localization and non-max suppression to improve accuracy.
- Trained on labeled car datasets for effective detection of vehicles in various scenarios.

### Applications
- Autonomous driving systems where real-time car detection is critical.
- Surveillance and traffic monitoring.

---

## 3. Face Recognition using Convolutional Neural Networks**

This project involves building a CNN-based **face recognition system** that can identify and authenticate individuals from an image. The model uses a convolutional neural network to extract facial features and compare them with known faces.

### Key Features
- Implements CNNs for feature extraction from facial images.
- Uses a **triplet loss function to minimize the distance between similar faces and maximize the distance between dissimilar ones.
- Provides accurate face matching and identification in various lighting and angle conditions.

### Applications
- Security systems for authentication and access control.
- **Biometric verification** in personal devices and corporate systems.

---

## 4. Image Segmentation using U-Net Architecture

This project applies the U-Net architecture a type of CNN widely used for image segmentation particularly in medical imaging. U-Net enables pixel-wise classification, which is essential for tasks like identifying tumors or segmenting specific regions in an image.

### Key Features
- U-Net architecture is designed for image segmentation tasks, especially those requiring fine localization.
- Performs pixel-wise classification making it highly suitable for medical image analysis.
- Works well on limited datasets due to its ability to use data augmentation and skip connections to preserve spatial information.

### Applications
- Medical imaging** for tasks like tumor detection, organ segmentation, etc.
- **Autonomous vehicles** for road segmentation and scene understanding.

---

## Technologies and Libraries Used
- TensorFlow A deep learning library used for model development and training.
- Keras A high-level neural networks API for building and deploying deep learning models.
- OpenCV Used for image processing and manipulation tasks.
- NumPy & Pandas For data handling and manipulation.
- Matplotlib For visualizing results and outputs.

---

## Setup and Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/cnn-computer-vision-models.git
   ```

2. Navigate to the project directory:
   ```bash
   cd cnn-computer-vision-models
   ```

3. Install the necessary dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## How to Run

Each model is located in its own Python file. You can run the scripts individually as follows:

### 1. Art Generation using Neural Style Transfer
   ```bash
   python art_generation.py --content content_image.jpg --style style_image.jpg --output output_image.jpg
   ```

### 2. Autonomous Driving: Car Detection using YOLO
   ```bash
   python car_detection.py --input input_image.jpg --output output_image.jpg
   ```

### 3. Face Recognition using CNNs
   ```bash
   python face_recognition.py --input test_image.jpg --database face_database/
   ```

### 4. Image Segmentation using U-Net
   ```bash
   python image_segmentation_unet.py --input medical_image.jpg --output segmented_image.jpg
   ```

---

## Results and Performance

- Art Generation Stylized images generated using a combination of content and style images.
- Car Detection Accurate real-time detection of cars in various environments, with bounding boxes drawn around detected cars.
- Face Recognition Correctly identifies individuals from a given database of faces.
- Image Segmentation Produces pixel-wise classification of medical images, identifying areas of interest with high accuracy.

---

## **Future Work and Improvements:**

1. Art Generation
   - Experiment with different architectures to reduce processing time for high-resolution images.

2. Car Detection
   - Incorporate other object detection tasks, such as detecting pedestrians and traffic signs.

3. Face Recognition
   - Improve model performance for faces under extreme variations, such as occlusion or low lighting.

4. Image Segmentation
   - Extend the model to work with 3D medical imaging data, such as MRI or CT scans.

---

## **Contributors:**
- Keshav Singh 

Feel free to raise an issue or contribute to the project by submitting a pull request.

---

## **License:**
This project is licensed under the DeepLearning.ai License 
---

