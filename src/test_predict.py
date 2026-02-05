import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.predict import predict_ecg

# Load one sample from dataset
df = pd.read_csv("data/raw/mitbih_train.csv", header=None)

# Find a PVC sample
pvc_sample = df[df.iloc[:, -1] == 1].iloc[0, :-1].values

result = predict_ecg(pvc_sample)

for k, v in result.items():
    print(f"{k}: {v}")


