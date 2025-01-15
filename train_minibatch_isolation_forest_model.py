#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
import argparse
import pydlt

# Function to recursively find DLT files in a folder
def find_dlt_files(folder_path):
    file_list = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(".dlt"):
                file_list.append(os.path.join(root, file))
    return file_list

# Function to parse a single DLT file and extract log messages
def parse_dlt_file(file_path):
    log_data = []
    print(f"Reading {file_path}...")
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

# Train or update a MiniBatch Isolation Forest model
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
    parser = argparse.ArgumentParser(description="Train or update a MiniBatch Isolation Forest using DLT files.")
    parser.add_argument("--folder", required=True, help="Path to the folder containing DLT files.")
    parser.add_argument("--output_model", default="minibatch_isolation_forest_model.pkl", help="Path to save the trained model.")
    args = parser.parse_args()

    # Step 1: Find all DLT files
    print(f"Searching for DLT files in {args.folder}...")
    dlt_files = find_dlt_files(args.folder)
    if not dlt_files:
        print("No DLT files found. Exiting.")
        exit(1)

    # Step 2: Parse and preprocess logs
    all_logs = []
    for dlt_file in dlt_files:
        logs = parse_dlt_file(dlt_file)
        all_logs.extend(logs)

    if not all_logs:
        print("No logs extracted from DLT files. Exiting.")
        exit(1)

    print("Preprocessing logs...")
    log_data = preprocess_logs(all_logs)
    print(f"Processed {len(log_data)} logs.")

    # Step 3: Train or update the MiniBatch Isolation Forest model
    model = train_or_update_model(log_data, args.output_model)

    # Step 4: Save the trained or updated model
    save_model(model, args.output_model)
