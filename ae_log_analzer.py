#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import joblib
import argparse

# Function to parse DLT files and extract log messages
def parse_dlt_files(folder_path):
    log_data = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".dlt"):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, "r") as file:
                for line in file:
                    log_data.append(line.strip())  # Example: append each line as a log entry
    return log_data

# Preprocess logs into numerical data
def preprocess_logs(logs):
    max_length = 100  # Maximum length of log vectors
    processed_logs = []
    for log in logs:
        encoded = [ord(char) for char in log[:max_length]]  # Simple character encoding
        if len(encoded) < max_length:
            encoded += [0] * (max_length - len(encoded))  # Pad to max length
        processed_logs.append(encoded)
    return np.array(processed_logs)

# Train an Isolation Forest model
def train_isolation_forest(data):
    print("Training Isolation Forest...")
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(data)
    return model

# Save the model to a file
def save_model(model, output_path):
    print(f"Saving the model to {output_path}...")
    joblib.dump(model, output_path)
    print("Model saved successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an Isolation Forest on DLT log files.")
    parser.add_argument("--folder", required=True, help="Path to the folder containing DLT files.")
    parser.add_argument("--output_model", default="isolation_forest_model.pkl", help="Path to save the trained model.")
    args = parser.parse_args()

    # Step 1: Read DLT files
    print("Parsing DLT files...")
    raw_logs = parse_dlt_files(args.folder)

    # Step 2: Preprocess logs
    print("Preprocessing logs...")
    log_data = preprocess_logs(raw_logs)
    print(f"Processed {len(log_data)} logs.")

    # Step 3: Split data into training and validation sets
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, val_index in sss.split(log_data, np.zeros(len(log_data))):
        x_train, x_val = log_data[train_index], log_data[val_index]

    # Step 4: Train the Isolation Forest model
    model = train_isolation_forest(x_train)

    # Step 5: Save the trained model
    save_model(model, args.output_model)
