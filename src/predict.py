import numpy as np
from tensorflow.keras.models import load_model

MODEL_PATH = "ecg_model.h5"

CLASS_NAMES = ["Normal", "PVC", "SVT", "Fusion", "Unknown"]

# Load model once
model = load_model(MODEL_PATH)

# Prepare a single ECG signal for prediction.
def preprocess_signal(signal):
    
    signal = np.array(signal).reshape(1, 187, 1)
    return signal

# Takes a 1D ECG signal 
def predict_ecg(signal):

    signal = preprocess_signal(signal)

    probs = model.predict(signal)[0]

    results = {}

    for name, prob in zip(CLASS_NAMES, probs):
        results[name] = float(prob)

    return results
