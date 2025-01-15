#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
import argparse
import pydlt
import hashlib

# Function to recursively find DLT files in a folder
def find_dlt_files(folder_path):
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"The folder path '{folder_path}' does not exist.")
    if not os.path.isdir(folder_path):
        raise NotADirectoryError(f"The path '{folder_path}' is not a directory.")

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
        error_message = f"Error reading {file_path}: {e}"
        with open("error_log.txt", "a") as error_file:
            error_file.write(error_message + "\n")
        print(error_message)
    return log_data

# Hashing-based log preprocessing
def hash_log(log, max_length=255):
    hashed = hashlib.md5(log.encode('utf-8')).digest()[:max_length]
    return [int(byte) for byte in hashed]

# Batch-based preprocessing
def preprocess_logs(logs, batch_size=1000, use_optimization=False):
    max_length = 255  # Maximum length of log vectors
    for i in range(0, len(logs), batch_size):
        batch_logs = logs[i:i + batch_size]
        if use_optimization:
            processed_logs = [hash_log(log, max_length=max_length) for log in batch_logs]
        else:
            processed_logs = []
            for log in batch_logs:
                encoded = [int.from_bytes(char.encode('utf-8'), 'little') for char in log[:max_length]]  # UTF-8 encoding
                if len(encoded) < max_length:
                    encoded += [0] * (max_length - len(encoded))  # Pad to max length
                processed_logs.append(encoded)
        yield np.array(processed_logs)

# Train or update a MiniBatch Isolation Forest model
def train_or_update_model(data, model_path):
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}...")
        model = joblib.load(model_path)
        print("Updating the existing model with new data...")
        max_estimators = 500  # Define a reasonable upper limit for n_estimators
        if model.n_estimators + 10 > max_estimators:
            print(f"Warning: n_estimators limit reached ({max_estimators}). No additional trees will be added.")
        else:
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
    parser.add_argument("--use-optimization", action="store_true", help="Enable optimized preprocessing using hashing.")
    args = parser.parse_args()

    # Step 1: Find all DLT files
    print(f"Searching for DLT files in {args.folder}...")
    try:
        dlt_files = find_dlt_files(args.folder)
    except (FileNotFoundError, NotADirectoryError) as e:
        print(e)
        exit(1)

    if not dlt_files:
        print(f"No DLT files found in the folder '{args.folder}'. Please check if the folder contains valid '.dlt' files or if the path is correct.")
        exit(1)

    # Step 2: Process each DLT file individually
    for dlt_file in dlt_files:
        # Parse the DLT file
        logs = parse_dlt_file(dlt_file)

        if not logs:
            print(f"No logs extracted from {dlt_file}. Skipping.")
            continue

        # Preprocess logs in batches
        print(f"Preprocessing logs from {dlt_file}...")
        for log_batch in preprocess_logs(logs, use_optimization=args.use_optimization):
            # Train or update the MiniBatch Isolation Forest model
            model = train_or_update_model(log_batch, args.output_model)

            # Save the trained or updated model
            save_model(model, args.output_model)
