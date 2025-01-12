#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
import argparse
import pydlt

# Function to parse DLT files and extract log messages
def parse_dlt_files(folder_path):
    log_data = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".dlt") or file_name.endswith(".DLT"):
            file_path = os.path.join(folder_path, file_name)
            print(f"Start to read {file_path}.")
            try:
                with pydlt.DltFileReader(file_path) as msg_file:
                    for message in msg_file:
                        log_data.append(str(message))
            except Exception as e:
                print(f"Error reading {file_name}: {e}")
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

# Train or update a MiniBatch Isolation Forest model
# Increment n_estimators when warm_start is True to avoid warnings
def train_or_update_model(data, model_path):
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}...")
        model = joblib.load(model_path)
        print("Updating the existing model with new data...")
        model.set_params(n_estimators=model.n_estimators + 10)  # Increment trees to fit new data
        model.fit(data)
    else:
        print("Training a new MiniBatch Isolation Forest model...")
        model = IsolationForest(n_estimators=100, warm_start=True, random_state=42, max_samples='auto', contamination=0.1)
        model.fit(data)
    return model

# Save the model to a file
def save_model(model, output_path):
    print(f"Saving the model to {output_path}...")
    joblib.dump(model, output_path)
    print("Model saved successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or update a MiniBatch Isolation Forest on DLT log files.")
    parser.add_argument("--folder", required=True, help="Path to the folder containing DLT files.")
    parser.add_argument("--output_model", default="minibatch_isolation_forest_model.pkl", help="Path to save the trained model.")
    args = parser.parse_args()

    # Step 1: Read and preprocess DLT files
    print("Parsing DLT files...")
    raw_logs = parse_dlt_files(args.folder)
    if not raw_logs:
        print("No logs found in the specified folder. Exiting.")
        exit(1)

    print("Preprocessing logs...")
    log_data = preprocess_logs(raw_logs)
    print(f"Processed {len(log_data)} logs.")

    # Step 2: Train or update the MiniBatch Isolation Forest model
    model = train_or_update_model(log_data, args.output_model)

    # Step 3: Save the trained or updated model
    save_model(model, args.output_model)
