#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import joblib
import argparse
import pydlt
import torch
from sklearn.metrics import classification_report

# Function to parse a DLT file and extract log messages
def parse_dlt_file(file_path):
    log_data = []
    try:
        with pydlt.DltFileReader(file_path) as msg_file:
            for message in msg_file:
                log_data.append(str(message))
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return log_data

# Preprocess logs into numerical data
def preprocess_logs(logs):
    max_length = 255  # Maximum length of log vectors
    processed_logs = []
    for log in logs:
        encoded = [ord(char) for char in log[:max_length]]  # Simple character encoding
        if len(encoded) < max_length:
            encoded += [0] * (max_length - len(encoded))  # Pad to max length
        processed_logs.append(encoded)
    return np.array(processed_logs)

# PyTorch-based model for anomaly detection
class AnomalyDetectionModel(torch.nn.Module):
    def __init__(self, input_size):
        super(AnomalyDetectionModel, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(128, input_size),
            torch.nn.ReLU()
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(input_size, 128)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Test the model with a given DLT file
def test_model(model_path, dlt_file, use_pytorch=False):
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        return

    print(f"Loading model from {model_path}...")
    if use_pytorch:
        input_size = 15  # Assuming 255 for input size, adjust as needed
        model = AnomalyDetectionModel(input_size)
        model.load_state_dict(torch.load(model_path))
        model.eval()
    else:
        model = joblib.load(model_path)

    print(f"Parsing DLT file: {dlt_file}...")
    raw_logs = parse_dlt_file(dlt_file)
    if not raw_logs:
        print("No logs found in the specified file. Exiting.")
        return

    print("Preprocessing logs...")
    log_data = preprocess_logs(raw_logs)

    if use_pytorch:
        print("Detecting anomalies with PyTorch model...")
        with torch.no_grad():
            log_data_tensor = torch.tensor(log_data, dtype=torch.float32)
            reconstructed = model(log_data_tensor)
            loss = torch.mean((log_data_tensor - reconstructed) ** 2, dim=1).numpy()
        threshold = np.percentile(loss, 95)  # Example threshold, adjust as needed
        predictions = loss > threshold
    else:
        print("Detecting anomalies with Isolation Forest model...")
        predictions = model.predict(log_data)
        predictions = predictions == -1  # Convert Isolation Forest output to boolean anomalies

    # Interpret predictions
    anomaly_count = np.sum(predictions)
    normal_count = len(predictions) - anomaly_count

    print("\nDetection Results:")
    print(f"Total logs: {len(predictions)}")
    print(f"Normal logs: {normal_count}")
    print(f"Anomalous logs: {anomaly_count}")

    print("\nSample Detection:")
    for i, log in enumerate(raw_logs[:10]):
        status = "Anomalous" if predictions[i] else "Normal"
        print(f"Log {i + 1}: {status} - {log[:100]}...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test an anomaly detection model with a DLT file.")
    parser.add_argument("--model", required=True, help="Path to the trained model file.")
    parser.add_argument("--file", required=True, help="Path to the DLT file to test.")
    parser.add_argument("--use-PyTorch", action="store_true", help="Use PyTorch-based model for testing.")
    args = parser.parse_args()

    test_model(args.model, args.file, use_pytorch=args.use_PyTorch)
