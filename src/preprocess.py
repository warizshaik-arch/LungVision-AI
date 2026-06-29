"""
Project : LungVision AI

Author : Wariz Shaik

Purpose:
Prepare chest X-ray images before training
the deep learning model.

Version : 1.0
"""
import cv2
import os
# Dataset location
DATASET_PATH = r"C:\Users\Wariz\Downloads\COVID-19_Radiography_Dataset"

# Image size for AI model
IMAGE_SIZE = (224, 224)
image_path = os.path.join(
    DATASET_PATH,
    "COVID",
    "images",
    "COVID-96.png"
)

image = cv2.imread(image_path)

print("Original Shape :", image.shape)
resized = cv2.resize(image, IMAGE_SIZE)

print("Resized Shape :", resized.shape)
normalized = resized / 255.0

print(normalized)