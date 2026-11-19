import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

LEGACY_DIR = os.path.join(ROOT_DIR, "crop_Production_prediction_new", "crop-production-predictor")
if os.path.exists(LEGACY_DIR) and LEGACY_DIR not in sys.path:
    sys.path.insert(1, LEGACY_DIR)

