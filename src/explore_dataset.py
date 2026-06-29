"""
Project: LungVision AI

Purpose:
Explore the Chest X-ray dataset.
"""

import os

DATASET_PATH = r"C:\Users\Wariz\Downloads\COVID-19_Radiography_Dataset"

classes = [
    "COVID",
    "Normal",
    "Lung_Opacity",
    "Viral Pneumonia"
]

print("=" * 55)
print("         LungVision AI Dataset Report")
print("=" * 55)

total = 0

for disease in classes:

    folder = os.path.join(DATASET_PATH, disease, "images")

    count = len(os.listdir(folder))

    total += count

    print(f"{disease:<20} : {count} images")

print("-" * 55)
print(f"Total Images        : {total}")
print("=" * 55)