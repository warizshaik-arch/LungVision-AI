import os

# Dataset path
DATASET_PATH = r"C:\Users\Wariz\Downloads\COVID-19_Radiography_Dataset"

# Classes we want to analyze
classes = [
    "COVID",
    "Normal",
    "Lung_Opacity",
    "Viral Pneumonia"
]

print("=" * 50)
print("      LungVision AI - Dataset Summary")
print("=" * 50)

total_images = 0

for disease in classes:
    image_folder = os.path.join(DATASET_PATH, disease, "images")

    if os.path.exists(image_folder):
        image_count = len(os.listdir(image_folder))
        total_images += image_count

        print(f"{disease:<20}: {image_count} images")
    else:
        print(f"{disease:<20}: Folder not found")

print("-" * 50)
print(f"Total Images         : {total_images}")
print("=" * 50)