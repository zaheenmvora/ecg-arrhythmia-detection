import os
import sys

# Allow imports from src
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.preprocess import load_data, preprocess_data, get_class_weights
from src.model import build_model
import numpy as np

DATA_PATH = "data/raw/mitbih_train.csv"
MODEL_PATH = "ecg_model.h5"

def main():

    print("Loading data...")
    X, y = load_data(DATA_PATH)

    print("Preprocessing...")
    X_train, X_test, y_train, y_test = preprocess_data(X, y)

    print("Computing class weights...")
    class_weights = get_class_weights(y)

    print("Building model...")
    model = build_model((187, 1), y_train.shape[1])

    print("Training model...")
    model.fit(
        X_train,
        y_train,
        epochs=15,
        batch_size=64,
        validation_data=(X_test, y_test),
        class_weight=class_weights
    )

    print("Saving model...")
    model.save(MODEL_PATH)

    print("Training complete!")


if __name__ == "__main__":
    main()
