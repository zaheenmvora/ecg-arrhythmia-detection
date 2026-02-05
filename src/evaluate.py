import os
import sys
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model

# Allow imports from src
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.preprocess import load_data, preprocess_data

DATA_PATH = "data/raw/mitbih_train.csv"
MODEL_PATH = "ecg_model.h5"

CLASS_NAMES = ["Normal", "PVC", "SVT", "Fusion", "Unknown"]

def main():

    print("Loading data...")
    X, y = load_data(DATA_PATH)

    print("Preprocessing...")
    X_train, X_test, y_train, y_test = preprocess_data(X, y)

    print("Loading trained model...")
    model = load_model(MODEL_PATH)

    print("Predicting...")
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    main()
