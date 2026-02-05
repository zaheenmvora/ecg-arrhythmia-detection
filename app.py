import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Import
sys.path.append(os.path.dirname(__file__))

from src.predict import predict_ecg

st.set_page_config(page_title="ECG Arrhythmia Detection", layout="wide")

st.title("🫀 ECG Arrhythmia Detection System")
st.markdown("Deep Learning-based ECG Beat Classification")

st.markdown("---")

uploaded_file = st.file_uploader("Upload ECG CSV file (single 187-length row)", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file, header=None)

    # If file is 187 rows × 1 column
    if df.shape[0] == 187 and df.shape[1] == 1:
        signal = df.iloc[:, 0].values

    # If file is 1 row × 187 columns
    elif df.shape[1] == 187:
        signal = df.iloc[0].values

    else:
        st.error("Uploaded file must contain exactly 187 values.")
        st.stop()


    st.subheader("📈 ECG Waveform")

    fig, ax = plt.subplots()
    ax.plot(signal)
    ax.set_title("ECG Signal")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)

    st.subheader("🔍 Prediction Probabilities")

    result = predict_ecg(signal)

    # Convert dict to dataframe for bar chart
    prob_df = pd.DataFrame({
        "Class": list(result.keys()),
        "Probability": list(result.values())
    })

    st.bar_chart(prob_df.set_index("Class"))

    predicted_class = max(result, key=result.get)
    confidence = result[predicted_class]

    st.subheader("🧠 Final Prediction")
    st.write(f"**{predicted_class}** ({confidence:.4f})")

    if predicted_class != "Normal":
        st.warning("⚠️ Abnormal rhythm detected. Clinical evaluation recommended.")
    else:
        st.success("Normal heartbeat detected.")

st.markdown("---")
st.caption("⚠️ This system is a research prototype and not intended for clinical diagnosis.")
