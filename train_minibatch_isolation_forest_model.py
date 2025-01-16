#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
import argparse
import pydlt
import hashlib
from multiprocessing import Pool, cpu_count
import tensorflow as tf

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
def parse_single_dlt_file(file_path):
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

# Multi-core parsing of DLT files
def parse_dlt_files_in_parallel(file_list):
    num_cores = cpu_count()
    print(f"Using {num_cores} cores for DLT file parsing...")
    def safe_parse(file):
        try:
            return parse_single_dlt_file(file)
        except Exception as e:
            error_message = f"Error parsing {file}: {e}"
            with open("error_log.txt", "a") as error_file:
                error_file.write(error_message + "\n")
            print(error_message)
            return []

    with Pool(num_cores) as pool:
        all_logs = pool.map(safe_parse, file_list)
    return [log for logs in all_logs for log in logs]  # Flatten the list of lists

# Hashing-based log preprocessing
def hash_log(log, max_length=255):
    hashed = hashlib.md5(log.encode('utf-8')).digest()[:max_length]
    return [int(byte) for byte in hashed]

# Batch-based preprocessing with multi-processing
# Optimized to process batches directly in the pool map
def preprocess_logs(logs, batch_size=1000, use_optimization=False):
    def process_batch(batch_logs):
        max_length = 255
        if use_optimization:
            return [hash_log(log, max_length=max_length) for log in batch_logs]
        else:
            processed_logs = []
            for log in batch_logs:
                encoded = [int.from_bytes(char.encode('utf-8'), 'little') for char in log[:max_length]]  # UTF-8 encoding
                if len(encoded) < max_length:
                    encoded += [0] * (max_length - len(encoded))  # Pad to max length
                processed_logs.append(encoded)
            return processed_logs

    num_cores = cpu_count()
    print(f"Using {num_cores} cores for preprocessing...")
    with Pool(num_cores) as pool:
        for processed_logs in pool.imap_unordered(process_batch, [logs[i:i + batch_size] for i in range(0, len(logs), batch_size)]):
            yield np.array(processed_logs)

# TensorFlow-based model for optional training
def train_model_tensorflow(model, data):
    print("Training the TensorFlow model...")
    model.fit(data, data, epochs=10, batch_size=32, verbose=1)
    return model

# Sklearn-based Isolation Forest model
def train_model_sklearn(model, data):
    print("Training or updating the Isolation Forest model...")
    model.fit(data)
    return model

# Save the model to a file
def save_model(model, output_path):
    print(f"Saving the model to {output_path}...")
    if isinstance(model, tf.keras.Model):
        model.save(output_path)
    else:
        joblib.dump(model, output_path)
    print("Model saved successfully.")

# Load a model from file or initialize a new one
def load_model(output_model, use_tensorflow):
    model = None
    if use_tensorflow:
        class IsolationForestModel(tf.keras.Model):
            def __init__(self):
                super(IsolationForestModel, self).__init__()
                self.dense1 = tf.keras.layers.Dense(128, activation='relu')
                self.dense2 = tf.keras.layers.Dense(64, activation='relu')
                self.output_layer = tf.keras.layers.Dense(1, activation='linear')

            def call(self, inputs):
                x = self.dense1(inputs)
                x = self.dense2(x)
                return self.output_layer(x)

        if os.path.exists(output_model):
            print(f"Loading existing TensorFlow model from {output_model}...")
            model = tf.keras.models.load_model(output_model)
        else:
            print("Initializing a new TensorFlow model...")
            model = IsolationForestModel()
        model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mean_squared_error')
    else:
        if os.path.exists(output_model):
            print(f"Loading existing Isolation Forest model from {output_model}...")
            model = joblib.load(output_model)
        else:
            print("Initializing a new Isolation Forest model...")
            model = IsolationForest(n_estimators=100, warm_start=True, random_state=42, max_samples='auto', contamination=0.1)
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or update a MiniBatch Isolation Forest or TensorFlow model using DLT files.")
    parser.add_argument("--folder", required=True, help="Path to the folder containing DLT files.")
    parser.add_argument("--output_model", default="minibatch_isolation_forest_model.pkl", help="Path to save the trained model.")
    parser.add_argument("--use-optimization", action="store_true", help="Enable optimized preprocessing using hashing.")
    parser.add_argument("--use-TensorFlow", action="store_true", help="Use TensorFlow-based model for training.")
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

    # Load or initialize the model
    model = load_model(args.output_model, args.use_TensorFlow)

    # Step 3: Preprocess logs in batches with multi-processing
    print("Preprocessing logs...")
    total_batches = (len(all_logs) + 999) // 1000
    for batch_index, log_batch in enumerate(preprocess_logs(all_logs, use_optimization=args.use_optimization), start=1):
        print(f"Processing batch {batch_index}/{total_batches}...")
        if args.use_TensorFlow:
            model = train_model_tensorflow(model, log_batch)
        else:
            model = train_model_sklearn(model, log_batch)

    # Save the trained or updated model
    save_model(model, args.output_model)
