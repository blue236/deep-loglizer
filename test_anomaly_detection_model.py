#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import joblib
import argparse
import pydlt
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

# Test the model with a given DLT file
def test_model(model_path, dlt_file):
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        return

    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)

    print(f"Parsing DLT file: {dlt_file}...")
    raw_logs = parse_dlt_file(dlt_file)
    if not raw_logs:
        print("No logs found in the specified file. Exiting.")
        return

    print("Preprocessing logs...")
    log_data = preprocess_logs(raw_logs)

    print("Detecting anomalies...")
    predictions = model.predict(log_data)

    # Interpret predictions (-1 indicates anomaly in Isolation Forest)
    anomalies = predictions == -1
    anomaly_count = np.sum(anomalies)
    normal_count = len(predictions) - anomaly_count

    print("\nDetection Results:")
    print(f"Total logs: {len(predictions)}")
    print(f"Normal logs: {normal_count}")
    print(f"Anomalous logs: {anomaly_count}")

    print("\nSample Detection:")
    for i, log in enumerate(raw_logs[:10]):
        status = "Anomalous" if anomalies[i] else "Normal"
        print(f"Log {i + 1}: {status} - {log[:100]}...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test an anomaly detection model with a DLT file.")
    parser.add_argument("--model", required=True, help="Path to the trained model file.")
    parser.add_argument("--file", required=True, help="Path to the DLT file to test.")
    args = parser.parse_args()

    test_model(args.model, args.file)
