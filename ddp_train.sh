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
    python scripts/ddp_train.py \
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
