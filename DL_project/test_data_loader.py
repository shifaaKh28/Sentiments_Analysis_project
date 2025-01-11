from data_loader import create_data_loader
from preprocessing import load_dataset, preprocess_dataset, tokenizer

"""
Module for testing the DataLoader functionality.

This script:
- Loads and preprocesses the dataset.
- Creates a DataLoader for batching and shuffling the data.
- Iterates over the DataLoader to validate its output.
"""

# Load the dataset
df = load_dataset("data.csv")  # Replace "data.csv" with the actual dataset path
df, df_filtered = preprocess_dataset(df)  # Preprocess the dataset

# Configuration for DataLoader
MAX_LEN = 50  # Maximum sequence length for tokenization
BATCH_SIZE = 8  # Number of samples per batch

# Create DataLoader for training
train_data_loader = create_data_loader(df, tokenizer, MAX_LEN, BATCH_SIZE)

# Validate DataLoader by inspecting the first batch
for batch in train_data_loader:
    print("Batch Keys:", batch.keys())  # Output keys of the batch dictionary
    print("Input IDs Shape:", batch['input_ids'].shape)  # Shape of tokenized input IDs
    print("Attention Mask Shape:", batch['attention_mask'].shape)  # Shape of attention masks
    print("Targets Shape:", batch['targets'].shape)  # Shape of target labels
    break  # Exit after processing the first batch
