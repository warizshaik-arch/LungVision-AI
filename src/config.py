from pathlib import Path

# Project Root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset Location
DATASET_PATH = r"C:\Users\Wariz\Downloads\COVID-19_Clean"
# Models
BASELINE_MODEL = PROJECT_ROOT / "models" / "baseline" / "lungvision_cnn.keras"
EFFICIENTNET_MODEL = PROJECT_ROOT / "models" / "efficientnet" / "efficientnet.keras"

# Output folders
OUTPUT_DIR = PROJECT_ROOT / "outputs"
GRAPH_DIR = OUTPUT_DIR / "graphs"
HEATMAP_DIR = OUTPUT_DIR / "heatmaps"
REPORT_DIR = OUTPUT_DIR / "reports"

# Classes (must match dataset order)
CLASS_NAMES = [
    "COVID",
    "lung_opacity",
    "normal",
    "viral_pneumonia"
]