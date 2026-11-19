import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Ensure project root is in sys.path
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Legacy fallback path compatibility
LEGACY_DIR = os.path.join(ROOT_DIR, "crop_Production_prediction_new", "crop-production-predictor")
if os.path.exists(LEGACY_DIR) and LEGACY_DIR not in sys.path:
    sys.path.insert(1, LEGACY_DIR)

# Set PYTHONPATH so debug reloader child process also retains the path
if "PYTHONPATH" in os.environ:
    if ROOT_DIR not in os.environ["PYTHONPATH"]:
        os.environ["PYTHONPATH"] = ROOT_DIR + os.pathsep + os.environ["PYTHONPATH"]
else:
    os.environ["PYTHONPATH"] = ROOT_DIR

from src.app import app

if __name__ == "__main__":
    print("Starting Machine Learning Based Crop Yield Prediction & Recommendation System...")
    print("Application URL: http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=True)
