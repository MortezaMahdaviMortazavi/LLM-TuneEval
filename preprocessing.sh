#!/bin/bash

# Set the path to the input CSV file, the model ID, and the directory to save the output
INPUT_PATH="path_to_your_input.csv"
SAVE_PATH="path_to_save_processed_data"
MODEL_ID="your_model_id"  # For example, "bert-base-multilingual-cased"
TEST_SIZE=0.2  # Optional: Specify the test set proportion (e.g., 0.2 for 20% test data)

# Run the preprocessing script
python3 scripts/preprocessing.py \
    --input_path "$INPUT_PATH" \
    --save_path "$SAVE_PATH" \
    --model_id "$MODEL_ID" \
    --test_size $TEST_SIZE
