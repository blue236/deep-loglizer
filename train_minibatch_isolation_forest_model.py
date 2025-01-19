#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
import joblib
import argparse
import pydlt
from multiprocessing import Pool, cpu_count
import torch
import torch.nn as nn
import torch.optim as optim

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

# Function to parse a single DLT file and extract log messages with filtering
def parse_single_dlt_file(file_path, filter_keywords=None):
    log_data = []
    print(f"Reading {file_path}...")
    try:
        with pydlt.DltFileReader(file_path) as msg_file:
            for message in msg_file:
                log_message = str(message.verbos_payload)
                if filter_keywords:
                    if any(keyword in log_message.lower() for keyword in filter_keywords):
                        log_data.append(log_message)
                else:
                    log_data.append(log_message)
    except Exception as e:
        error_message = f"Error reading {file_path}: {e}"
        with open("error_log.txt", "a") as error_file:
            error_file.write(error_message + "\n")
        print(error_message)
    return log_data

# Function to safely parse a file with filtering
def safe_parse(file, filter_keywords=None):
    try:
        return parse_single_dlt_file(file, filter_keywords)
    except Exception as e:
        error_message = f"Error parsing {file}: {e}"
        with open("error_log.txt", "a") as error_file:
            error_file.write(error_message + "\n")
        print(error_message)
        return []

# Multi-core parsing of DLT files with filtering
def parse_dlt_files_in_parallel(file_list, filter_keywords=None):
    num_cores = cpu_count()
    print(f"Using {num_cores} cores for DLT file parsing...")
    with Pool(num_cores) as pool:
        all_logs = pool.starmap(safe_parse, [(file, filter_keywords) for file in file_list])
    return [log for logs in all_logs for log in logs]  # Flatten the list of lists

# Batch-based preprocessing with multi-processing
def process_batch(batch_logs):
    max_length = 255
    processed_logs = []
    for log in batch_logs:
        encoded = [int.from_bytes(char.encode('utf-8'), 'little') for char in log[:max_length]]  # UTF-8 encoding
        if len(encoded) < max_length:
            encoded += [0] * (max_length - len(encoded))  # Pad to max length
        processed_logs.append(encoded)
    return processed_logs

def preprocess_logs(logs, batch_size=100000):
    num_cores = cpu_count()
    print(f"Using {num_cores} cores for preprocessing...")
    with Pool(num_cores) as pool:
        for processed_logs in pool.imap_unordered(process_batch, [logs[i:i + batch_size] for i in range(0, len(logs), batch_size)]):
            yield np.array(processed_logs)

# PyTorch-based model for optional training
class AnomalyDetectionModel(nn.Module):
    def __init__(self, input_size):
        super(AnomalyDetectionModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.ReLU(), nn.Linear(input_size, 128)
        )
        self.decoder = nn.Sequential(
            nn.ReLU(), nn.Linear(128, input_size)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def train_model(self, data_loader, criterion, optimizer, epochs=10, checkpoint_path=None):
        print("Training the PyTorch model...")
        self.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for batch in data_loader:
                inputs = batch
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, inputs)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(data_loader):.4f}")

            # Save checkpoint after every epoch
            if checkpoint_path:
                torch.save(self.state_dict(), f"{checkpoint_path}_epoch{epoch+1}.pth")
                print(f"Checkpoint saved at {checkpoint_path}_epoch{epoch+1}.pth")
        return self

    def validate_model(self, validation_data_loader, criterion):
        print("Validating the PyTorch model...")
        self.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in validation_data_loader:
                inputs = batch
                outputs = self(inputs)
                loss = criterion(outputs, inputs)
                total_loss += loss.item()
        avg_loss = total_loss / len(validation_data_loader)
        print(f"Validation Loss: {avg_loss:.4f}")
        return avg_loss

# Sklearn-based Isolation Forest model
def train_model_sklearn(model, data, validation_data=None):
    print("Configuring the Isolation Forest model...")
    max_estimators = 50000 # Define a reasonable upper limit for n_estimators
    if model.n_estimators + 10 > max_estimators:
        print(f"Warning: n_estimators limit reached ({max_estimators}). No additional trees will be added.")
    else:
        model.set_params(n_estimators=model.n_estimators + 10) # Increment trees to fit new data
    print("Training or updating the Isolation Forest model...")
    model.fit(data)

    if validation_data is not None:
        print("Validating the model...")
        predictions = model.predict(validation_data)
        anomaly_labels = predictions == -1  # Convert Isolation Forest output to boolean anomalies
        normal_count = np.sum(~anomaly_labels)
        anomaly_count = np.sum(anomaly_labels)
        print(f"Validation Results: {normal_count} normal logs, {anomaly_count} anomalies detected.")

    return model

# Save the model to a file
def save_model(model, output_path, use_pytorch=False):
    print(f"Saving the model to {output_path}...")
    if use_pytorch:
        torch.save(model.state_dict(), output_path)
    else:
        joblib.dump(model, output_path)
    print("Model saved successfully.")

# Load a model from file or initialize a new one
def load_model(output_model, use_pytorch, input_size=None):
    model = None
    if use_pytorch:
        model = AnomalyDetectionModel(input_size)
        if os.path.exists(output_model):
            print(f"Loading existing PyTorch model from {output_model}...")
            model.load_state_dict(torch.load(output_model))
        else:
            print("Initializing a new PyTorch model...")
    else:
        if os.path.exists(output_model):
            print(f"Loading existing Isolation Forest model from {output_model}...")
            model = joblib.load(output_model)
        else:
            print("Initializing a new Isolation Forest model...")
            model = IsolationForest(n_estimators=100, warm_start=True, random_state=42, max_samples='auto', contamination=0.1, n_jobs=-1, verbose=1)
    return model

# Display the model overview
def display_model_overview(model, use_pytorch):
    print("\nModel Overview:")
    if use_pytorch:
        print("PyTorch Model:")
        print(model)
    else:
        print("Isolation Forest Parameters:")
        print(f"Number of Trees (n_estimators): {model.n_estimators}")
        print(f"Max Samples: {model.max_samples}")
        print(f"Contamination: {model.contamination}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or update a MiniBatch Isolation Forest or PyTorch model using DLT files.")
    parser.add_argument("--folder", required=True, help="Path to the folder containing DLT files.")
    parser.add_argument("--output_model", default="minibatch_isolation_forest_model.pkl", help="Path to save the trained model.")
    parser.add_argument("--use-PyTorch", action="store_true", help="Use PyTorch-based model for training.")
    parser.add_argument("--checkpoint", default=None, help="Path to save intermediate model checkpoints.")
    parser.add_argument("--validation-size", type=int, default=1000, help="Number of samples for validation.")
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

    # Step 2: Parse DLT files in parallel
    print("Parsing DLT files in parallel...")
    all_logs = parse_dlt_files_in_parallel(dlt_files)

    if not all_logs:
        print("No logs extracted from DLT files. Exiting.")
        exit(1)

    # Step 3: Preprocess logs in batches with multi-processing
    print("Preprocessing logs...")
    processed_logs = []
    for batch in preprocess_logs(all_logs):
        processed_logs.extend(batch)

    processed_logs = np.array(processed_logs)
    validation_logs = processed_logs[:args.validation_size]
    training_logs = processed_logs[args.validation_size:]

    # Step 4: Load or initialize the model
    input_size = len(training_logs[0]) if training_logs else 255
    model = load_model(args.output_model, args.use_PyTorch, input_size=input_size)

    if args.use_PyTorch:
        print("Training PyTorch model...")
        data_loader = torch.utils.data.DataLoader(torch.tensor(training_logs, dtype=torch.float32), batch_size=32, shuffle=True)
        validation_loader = torch.utils.data.DataLoader(torch.tensor(validation_logs, dtype=torch.float32), batch_size=32, shuffle=False)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        model.train_model(data_loader, criterion, optimizer, checkpoint_path=args.checkpoint)
        model.validate_model(validation_loader, criterion)
    else:
        print("Training Isolation Forest model...")
        model = train_model_sklearn(model, training_logs, validation_data=validation_logs)

    # Step 5: Save the trained or updated model
    save_model(model, args.output_model, use_pytorch=args.use_PyTorch)

    # Step 6: Display the model overview
    display_model_overview(model, use_pytorch=args.use_PyTorch)
