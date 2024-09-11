import logging
import os
import random
import torch
import mlflow
import argparse
import time

from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig, 
    TrainingArguments, 
    set_seed,
)
from trl import SFTTrainer
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from utils import load_datasets
from callbacks import SpeedMonitoringCallback
from accelerate import PartialState , Accelerator
from args import simple_train_arguments
from liger_kernel.transformers import apply_liger_kernel_to_llama
from llama_attn_replace import replace_llama_attn

import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch.set_default_dtype(torch.bfloat16)


USE_LONGLORA = False

def training_function(args):
    ################
    # Dataset
    ################
    print("Loading datasets...")
    train_dataset, test_dataset = load_datasets(args.dataset_path)
    print("Datasets loaded successfully.")

    ################
    # Model & Tokenizer
    ################
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token    
    print("Tokenizer loaded successfully.")
    
    torch_dtype = torch.bfloat16

    if USE_LONGLORA:
        replace_llama_attn()

    print("Configuring quantization...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_quant_storage=torch_dtype,
    )
    print("Quantization configured successfully.")

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=quantization_config,
        device_map={"": PartialState().process_index},
        torch_dtype=torch_dtype,
        use_cache=True,  # No need to disable cache since we don't use FSDP
        attn_implementation=args.attn_implementation,
    )
    apply_liger_kernel_to_llama()
    print("Model loaded successfully.")
    
    ################
    # PEFT
    ################
    print("Configuring PEFT...")
    peft_config = LoraConfig(
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        r=args.lora_rank,
        bias=args.bias,
        target_modules="all-linear",
        task_type=args.task_type,
        use_dora=False,
        use_rslora=True,
    )
    print("PEFT configuration done.")

    print("Preparing model for k-bit training...")
    model = get_peft_model(
        prepare_model_for_kbit_training(model),
        peft_config,
    )
    print("Model prepared for k-bit training.")

    print("Converting model parameters to target dtype...")
    for param in model.parameters():
        param.data = param.data.to(torch_dtype)
    print("Model parameters converted successfully.")

    ################
    # Training
    ################
    print("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir=args.output_dir,  # Directory where model checkpoints and logs are saved.
        report_to="mlflow",
        num_train_epochs=args.num_train_epochs,  # Total number of training epochs.
        per_device_train_batch_size=args.per_device_train_batch_size,  # Batch size per device (GPU) during training.
        per_device_eval_batch_size=args.per_device_eval_batch_size,  # Batch size per device during evaluation.
        gradient_accumulation_steps=args.gradient_accumulation_steps,  # Number of forward passes before a backward pass.\
        gradient_checkpointing=False,
        logging_steps=args.logging_steps,  # Interval (in steps) for logging training metrics.
        save_strategy=args.save_strategy,  # Strategy for saving model checkpoints (e.g., "epoch", "steps").
        optim=args.optim,  # Optimizer to use during training (e.g., "adamw_torch").
        evaluation_strategy=args.evaluation_strategy,  # Strategy for evaluation (e.g., "epoch", "steps").
        learning_rate=args.learning_rate,  # Initial learning rate for training.
        max_grad_norm=args.max_grad_norm,  # Maximum gradient norm for gradient clipping to prevent exploding gradients.
        warmup_ratio=args.warmup_ratio,  # Ratio of total steps to use for learning rate warmup.
        lr_scheduler_type=args.lr_scheduler_type,  # Type of learning rate scheduler (e.g., "linear", "cosine").
        bf16=args.bf16,  # Use bfloat16 precision for faster training and reduced memory usage.
        tf32=args.tf32,  # Enable TensorFloat-32 (TF32) precision on supported GPUs for faster computation.
        dataloader_num_workers=6,  # Number of subprocesses to use for data loading.
        dataloader_pin_memory=True,  # Whether to pin memory during data loading for faster transfer to GPU.
        torch_compile=True,  # Enable `torch.compile()` to optimize model execution for performance.
        torch_empty_cache_steps=None,  # Option to empty CUDA cache at specific intervals (None means disabled).
        seed=args.seed,  # Random seed for reproducibility.
        ddp_find_unused_parameters=False,
        ddp_backend='nccl',
    )
    print("Training arguments set up successfully.")

    print("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        dataset_text_field="prompt",
        eval_dataset=test_dataset,
        peft_config=peft_config,
        max_seq_length=args.max_seq_length,
        tokenizer=tokenizer,
        packing=True,
        dataset_kwargs={
            "add_special_tokens": False,
            "append_concat_token": False,
        },
    )
    print("SFTTrainer initialized successfully.")
    
    print("Adding monitoring callback...")
    callbacks = [SpeedMonitoringCallback(args.logging_steps)]
    for callback in callbacks:
        trainer.add_callback(callback)
    print("Monitoring callback added successfully.")

    ##########################
    # Train model
    ##########################
    checkpoint = None
    if training_args.resume_from_checkpoint:
        checkpoint = training_args.resume_from_checkpoint
        print(f"Resuming from checkpoint: {checkpoint}")
    
    try:
        print("Starting training...")
        trainer.train(resume_from_checkpoint=checkpoint)
        print("Training completed successfully.")
    except Exception as e:
        print(f"An error occurred during training: {str(e)}")
    finally:
        trainer.save_model()
        print("Model saved successfully.")


if __name__ == "__main__":
    mlflow.set_experiment("Distributed Training Tracking")
    args = simple_train_arguments()
    set_seed(args.seed)  
    with mlflow.start_run(run_name=args.run_name):
        training_function(args)
