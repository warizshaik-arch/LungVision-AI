import os
import matplotlib.pyplot as plt

DATASET_PATH = r"C:\Users\Wariz\Downloads\COVID-19_Radiography_Dataset"

classes = [
    "COVID",
    "Normal",
    "Lung_Opacity",
    "Viral Pneumonia"
]

counts = []

for disease in classes:
    folder = os.path.join(DATASET_PATH, disease, "images")
    counts.append(len(os.listdir(folder)))

plt.figure(figsize=(8,5))
plt.bar(classes, counts)

plt.title("Chest X-ray Dataset Distribution")
plt.xlabel("Disease")
plt.ylabel("Number of Images")

plt.show()