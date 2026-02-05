# ECG Arrhythmia Detection System 🫀

A Deep Learning-based ECG arrhythmia detection web application that classifies heartbeats using a 1D Convolutional Neural Network (CNN).  
This project combines signal preprocessing, neural network modeling, and an interactive Streamlit interface to provide real-time ECG classification and visualization.

## Features

- ECG signal preprocessing (187-length beats)
- Multi-class heartbeat classification
- 1D Convolutional Neural Network (CNN)
- Class imbalance handling using computed class weights
- Confusion matrix & classification report evaluation
- Interactive Streamlit interface with waveform visualization
- Probability confidence display for each class

## Technologies Used

- Python  
- TensorFlow / Keras (1D CNN)  
- NumPy  
- Pandas  
- Scikit-learn  
- Streamlit  
- Matplotlib  

## Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/ecg-arrhythmia-detection.git
   cd ecg-arrhythmia-detection
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the model:**
   ```bash
   python src/train.py
   ```

5. **Run the application:**
   ```bash
   streamlit run app.py
   ```

## Model Details

- Model: 1D Convolutional Neural Network  
- Architecture:
  - Conv1D (32 filters)
  - Conv1D (64 filters)
  - Conv1D (128 filters)
  - Dense (128 units)
  - Dropout (0.5)
  - Softmax Output Layer  
- Optimizer: Adam  
- Loss Function: Categorical Crossentropy  
- Class imbalance handled using computed class weights  
- Evaluated using Classification Report and Confusion Matrix  

## Dataset

- MIT-BIH Arrhythmia Dataset  
- 187-length ECG beats  
- Multi-class classification:
  - Normal
  - PVC
  - SVT
  - Fusion
  - Unknown

## Contributing

Contributions are welcome. Feel free to open issues or submit pull requests for improvements.

## Disclaimer

This system is a research prototype for educational purposes only.  
It is not intended for clinical or medical diagnosis.

## License

[MIT License](LICENSE)

## Author

Zaheen M Vora

Computer Engineering Student | Aspiring Data Science and ML Engineer
