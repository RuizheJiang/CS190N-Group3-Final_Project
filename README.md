# CS190N Group 3: Final Project

## Overview
This repository contains the code, data, and instructions to reproduce the results presented in our report. The project focuses on classifying TCP congestion control algorithms (e.g., Reno, Cubic) using machine learning techniques. Our approach integrates statistical feature extraction and an LSTM-based model to achieve high classification accuracy and robustness to class imbalance.

## Repository Structure
```
.
├── data/                # Directory for dataset and preprocessing scripts
├── model/               # Directory containing LSTM model implementation
├── results/             # Directory to save evaluation metrics, plots, and logs
├── tcp.py               # Script to preprocess data and generate flows.json
├── main.py              # Main script for training and evaluation
├── requirements.txt     # Python dependencies
└── README.md            # Project instructions (this file)
```

## Requirements
To reproduce the results, ensure the following dependencies are installed:
- Python 3.8+
- PyTorch 1.10+
- NumPy
- scikit-learn
- Matplotlib
- Scapy

Install all dependencies using:
```bash
pip install -r requirements.txt
```

## Instructions to Reproduce Results

### Step 1: Run `tcp.py` to Generate `flows.json`
The first step is to preprocess the `.pcap` dataset and generate `flows.json`. Place the `.pcap` files in the `data/` directory and run:
```bash
python tcp.py
```
This script extracts statistical features from the `.pcap` files and saves them in `flows.json` within the `data/` folder.

### Step 2: Train the Model using `main.py`
Once `flows.json` is generated, train the LSTM model using:
```bash
python main.py --mode train --epochs 10 --batch_size 8 --learning_rate 0.001 --seq_length 100
```

Key arguments:
- `--mode`: Set to `train` for training.
- `--epochs`: Number of training epochs.
- `--batch_size`: Batch size for training.
- `--learning_rate`: Learning rate for the optimizer.
- `--seq_length`: Length of input sequences.


## Results
Key findings from the experiments:
- **Accuracy**: 88.2%

Feature importance and ablation studies confirmed that `window_mean` and `window_std` are critical for distinguishing TCP congestion control algorithms.

## File Descriptions
- **`tcp.py`**: Script to preprocess `.pcap` files and generate `flows.json`.
- **`main.py`**: Main script for training and evaluation.

