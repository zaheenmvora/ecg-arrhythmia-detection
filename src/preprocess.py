import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical

def load_data(path):
    df = pd.read_csv(path, header=None)

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    return X, y

def preprocess_data(X, y, test_size=0.2):
    # Reshape for CNN (samples, timesteps, channels)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # One-hot encode labels
    y_encoded = to_categorical(y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test

def get_class_weights(y):
    classes = np.unique(y)

    weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=y
    )

    return dict(zip(classes, weights))
