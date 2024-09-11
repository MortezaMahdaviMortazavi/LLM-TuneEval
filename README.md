# README: Training, Inference, and Evaluation Guide

This guide provides detailed instructions for preprocessing your dataset, running training (FSDP and DDP), performing model inference, and evaluating the model's performance. Follow the steps below to ensure a smooth workflow.

---

## Table of Contents
- [Preprocessing the Dataset](#preprocessing-the-dataset)
- [Fully Sharded Data Parallel (FSDP) Training Setup](#fully-sharded-data-parallel-fsdp-training-setup)
- [Distributed Data Parallel (DDP) Training Setup](#distributed-data-parallel-ddp-training-setup)
- [Model Inference](#model-inference-with-llama-and-other-causallm-models)
- [Model Evaluation](#model-evaluation-script)
---

## Preprocessing the Dataset

Use the `preprocessing.py` script to convert your dataset from CSV format into a JSON format suitable for training.

### Input Data Format:
- **CSV File**: The CSV file should contain the following columns:
  - `inputs`: Input prompts or questions from the user.
  - `targets`: Expected or generated responses.

### Saving Path:
- **save_path**: The directory path where preprocessed JSON data will be saved.

### Model Name or ID:
- **model_id**: The model identifier for the tokenizer, which is used to map data into the correct format.

### Running the Preprocessing Script:
Run the following command to preprocess your data:

```bash
python3 scripts/preprocessing.py --input_path path_to_input.csv \
                                 --save_path path_to_save_json \
                                 --model_id your_model_id \
                                 --test_size 0.2  # Optional: Split the dataset into training and testing sets

```

## Fully Sharded Data Parallel (FSDP) Training Setup

This section provides a guide to setting up and running FSDP training using torchrun and PyTorch. Ensure that your dataset has been preprocessed and saved in JSON format before starting the training process.

### Prerequisites:
    - Dataset Path: Path to the preprocessed data (JSON format).
    - Model ID: The identifier for the model to be trained (e.g., unsloth/Meta-Llama-3.1-8B-Instruct).
    - Output Directory: Directory where the checkpoints and logs will be saved.
    - GPU Setup: Make sure to define which GPUs to use by setting the CUDA_VISIBLE_DEVICES environment variable.

### Running the FSDP Training(fsdp_train.sh)
```
export TOKENIZERS_PARALLELISM=true
export CUDA_VISIBLE_DEVICES=0,1,2  # Specify the GPUs to use
export ACCELERATE_USE_FSDP=1
export FSDP_CPU_RAM_EFFICIENT_LOADING=1
torchrun --nproc_per_node=3 scripts/fsdp_train.py \
    --dataset_path "path_to_save_processed_data" \
    --model_id "your_model_name" \
    --max_seq_length 8192 \
    --output_dir "Llama3.1-FSDP-60K" \
    --report_to "mlflow" \
    --run_name "Llama3.1-FSDP-60K" \
    --num_train_epochs 4 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --logging_steps 5 \
    --save_strategy epoch \
    --evaluation_strategy epoch \
    --optim "adamw_torch_fused" \
    --learning_rate 0.0003 \
    --max_grad_norm 0.3 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --bf16 \
    --tf32 \
    --fsdp "full_shard auto_wrap offload" \
    --fsdp_backward_prefetch backward_pre \
    --fsdp_forward_prefetch "true" \
    --use_orig_params "true" \
    --fsdp_activation_checkpointing True \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --lora_rank 32 \
    --task_type "CAUSAL_LM" \
    --bias "none" \
    --seed 42 \
    --attn_implementation flash_attention_2
```

### Important Notes:
    - Ensure the dataset is preprocessed and saved in JSON format before starting.
    - Adjust the batch size, learning rate, and other training parameters based on your system configuration.


## Distributed Data Parallel (DDP) Training Setup

This section explains how to run training using PyTorch's Distributed Data Parallel (DDP) setup. The method is ideal for multi-GPU training, enabling efficient distributed training of large models.

### Steps to Run
    - Preprocess the Dataset: Ensure your dataset is preprocessed into JSON format before proceeding. Follow the instructions in the Preprocessing the Dataset section.

    - Run the DDP Training: To run the training using DDP, execute the following script (located in ddp_train.sh):

```
#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2
export TOKENIZERS_PARALLELISM=true

# Define your arguments with default or custom values
DATASET_PATH="data/TRAIN"
MODEL_ID="unsloth/Meta-Llama-3.1-8B-Instruct"
OUTPUT_DIR="models/DDP-Tebyan-Summarization-3EPOCH"
RUN_NAME="DDP"
MAX_SEQ_LENGTH=4096
NUM_TRAIN_EPOCHS=3
TRAIN_BATCH_SIZE=1
EVAL_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=4
LOGGING_STEPS=5
SAVE_STRATEGY="epoch"
EVALUATION_STRATEGY="epoch"
OPTIMIZER="adamw_torch_fused"
LEARNING_RATE=0.0002
MAX_GRAD_NORM=0.3
WARMUP_RATIO=0.03
LR_SCHEDULER_TYPE="constant"
LORA_ALPHA=32
LORA_DROPOUT=0.05
LORA_RANK=16
TASK_TYPE="CAUSAL_LM"
BIAS="none"
SEED=42
ATTN_IMPLEMENTATION="flash_attention_2"
USE_BF16="--bf16"    # Add this flag for bfloat16 precision (remove if not needed)
USE_TF32="--tf32"    # Add this flag for tf32 precision (remove if not needed)
GRADIENT_CHECKPOINTING="--gradient_checkpointing"  # Add this flag for gradient checkpointing (remove if not needed)

# Add use_ddp variable to control whether DDP is used
use_ddp=true

if [ "$use_ddp" = true ]; then
    echo "Running with Distributed Data Parallel (DDP)"
    # Set environment for DDP and run with torchrun
    export CUDA_VISIBLE_DEVICES=0,1,2
    WORLD_SIZE=3
    torchrun --nproc_per_node=3 scripts/ddp_train.py \
        --dataset_path "$DATASET_PATH" \
        --model_id "$MODEL_ID" \
        --output_dir "$OUTPUT_DIR" \
        --run_name "$RUN_NAME" \
        --max_seq_length "$MAX_SEQ_LENGTH" \
        --num_train_epochs "$NUM_TRAIN_EPOCHS" \
        --per_device_train_batch_size "$TRAIN_BATCH_SIZE" \
        --per_device_eval_batch_size "$EVAL_BATCH_SIZE" \
        --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
        --logging_steps "$LOGGING_STEPS" \
        --save_strategy "$SAVE_STRATEGY" \
        --evaluation_strategy "$EVALUATION_STRATEGY" \
        --optim "$OPTIMIZER" \
        --learning_rate "$LEARNING_RATE" \
        --max_grad_norm "$MAX_GRAD_NORM" \
        --warmup_ratio "$WARMUP_RATIO" \
        --lr_scheduler_type "$LR_SCHEDULER_TYPE" \
        --lora_alpha "$LORA_ALPHA" \
        --lora_dropout "$LORA_DROPOUT" \
        --lora_rank "$LORA_RANK" \
        --task_type "$TASK_TYPE" \
        --bias "$BIAS" \
        --seed "$SEED" \
        --attn_implementation "$ATTN_IMPLEMENTATION" \
        $USE_BF16 $USE_TF32 $GRADIENT_CHECKPOINTING
else
    echo "Running without DDP"
    # Run the script normally
    python scripts/simple_train.py \
        --dataset_path "$DATASET_PATH" \
        --model_id "$MODEL_ID" \
        --output_dir "$OUTPUT_DIR" \
        --run_name "$RUN_NAME" \
        --max_seq_length "$MAX_SEQ_LENGTH" \
        --num_train_epochs "$NUM_TRAIN_EPOCHS" \
        --per_device_train_batch_size "$TRAIN_BATCH_SIZE" \
        --per_device_eval_batch_size "$EVAL_BATCH_SIZE" \
        --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
        --logging_steps "$LOGGING_STEPS" \
        --save_strategy "$SAVE_STRATEGY" \
        --evaluation_strategy "$EVALUATION_STRATEGY" \
        --optim "$OPTIMIZER" \
        --learning_rate "$LEARNING_RATE" \
        --max_grad_norm "$MAX_GRAD_NORM" \
        --warmup_ratio "$WARMUP_RATIO" \
        --lr_scheduler_type "$LR_SCHEDULER_TYPE" \
        --lora_alpha "$LORA_ALPHA" \
        --lora_dropout "$LORA_DROPOUT" \
        --lora_rank "$LORA_RANK" \
        --task_type "$TASK_TYPE" \
        --bias "$BIAS" \
        --seed "$SEED" \
        --attn_implementation "$ATTN_IMPLEMENTATION" \
        $USE_BF16 $USE_TF32 $GRADIENT_CHECKPOINTING
fi

```

## Model Inference with LLaMA and other CausalLM Models

Follow the steps below to perform inference using a pre-trained LLaMA or other CausalLM models.

### Steps to Run the Inference
    - Model Name: Specify the model ID (e.g., from Hugging Face) or the local path to the saved model.

    - Data Path: Provide the path to the input data in CSV format.

    - Output Path: Specify where to save the results in CSV format.

    - LoRA Weights (Optional): If using LoRA weights for enhanced performance, specify the directory path where the LoRA weights are stored.

### Run the bash below(inference.sh file)
```
#!/bin/bash

# Script to run the LLaMA Model Inference Pipeline

# Activate the appropriate Python environment
# source path/to/venv/bin/activate

python scripts/inference.py \
  --model_name "models/llama-lora32-pretrained" \
  --device "cuda:0" \
  --load_in_4bit \
  --data_path "data/TEST-SUMMARIZATION-FINAL-PROMPT.csv" \
  --output_path "LlamaLora32-Summarization-FinalTest-Prediction.csv" \
  --prompt_column "prompt" \
  --max_new_tokens 4096 \
  --do_sample \
  --temperature 0.0001 \
  --top_k 50 \
  --top_p 0.95 \
  --batch_size 16
  # --lora_weights None \

```


## Model Evaluation Script

This section provides instructions for evaluating model predictions using various metrics (e.g., BERTScore, ROUGE, BLEU).

### Steps to Run the Evaluation(specify these params below)
    - Data Path: Provide the path to the original test data CSV.

    - Prediction Path: Specify the path to the CSV file containing the model's predictions.

    - Log File: Define the path where evaluation results will be saved.

    - Column Names: Specify the column names for the target labels and predictions.

### Run the bash script(evaluate.sh)
```
#!/bin/bash

MODEL_TYPE="llamaLora32-instruct"
TASK="summarization"
TOKENIZER_MODEL="unsloth/Meta-Llama-3.1-8B-Instruct"
DATA_PATH="data/TEST-SUMMARIZATION-FINAL-PROMPT.csv"
PRED_PATH="LlamaLora32-Summarization-FinalTest-Prediction.csv"
LOG_PATH="./llamaLora32-Summarization-FinalTest-BestCompletePrompt.log"
TARGET_COLUMN="targets"
PREDICT_COLUMN="assistant"



if [ "$TASK" = "summarization" ]; then
    METRICS="bert_score rouge bleu meteor"
elif [ "$TASK" = "mrc" ]; then
    METRICS="f1_score exact_match bert_score"
else
    echo "Unknown task type: $TASK. Please add the task and its corresponding metrics."
    exit 1
fi

echo "Task: $TASK"
echo "Metrics: $METRICS"

# Run the Python evaluation script with the provided parameters and chosen metrics
python scripts/evaluate.py \
  --model_type "$MODEL_TYPE" \
  --task "$TASK" \
  --metrics $METRICS \
  --tokenizer_model "$TOKENIZER_MODEL" \
  --data_path "$DATA_PATH" \
  --pred_path "$PRED_PATH" \
  --log_path "$LOG_PATH" \
  --target_column "$TARGET_COLUMN" \
  --predict_column "$PREDICT_COLUMNS"

```

### Important Notes:
    - Adjust metrics based on the task (e.g., summarization, machine reading comprehension).
    - Make sure to set the correct column names for target labels and predictions.
