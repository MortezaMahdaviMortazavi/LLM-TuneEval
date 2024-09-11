import argparse

def inference_arguments():
    parser = argparse.ArgumentParser(description="LLaMA Model Inference Pipeline")
    parser.add_argument('--model_name', type=str, required=True, help="The name or path of the base model to load")
    parser.add_argument('--device', type=str, default="cuda:1", help="Device to run the model on")
    parser.add_argument('--load_in_4bit', action='store_true', help="Whether to load the model in 4-bit precision")
    parser.add_argument('--lora_weights', type=str, nargs='*', default=None, required=False, help="Path to one or more LoRA weights to load and merge")
    parser.add_argument('--data_path', type=str, required=True, help="Path to the input CSV file")
    parser.add_argument('--prompt_column', type=str, required=True, default="prompt", help="Name of the column in your CSV file that contains the prompt")
    parser.add_argument('--output_path', type=str, required=True, help="Path to save the output CSV file")
    parser.add_argument('--max_new_tokens', type=int, default=4096, help="Maximum number of tokens to generate")
    parser.add_argument('--do_sample', action='store_true', help="Whether to use sampling for generation")
    parser.add_argument('--temperature', type=float, default=0.1, help="Sampling temperature")
    parser.add_argument('--top_k', type=int, default=10, help="Top-k sampling")
    parser.add_argument('--top_p', type=float, default=0.95, help="Top-p sampling")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for generation")
    return parser.parse_args()


def evaluate_arguments():
    parser = argparse.ArgumentParser(description="Evaluate model predictions.")
    parser.add_argument('--model_type', type=str, required=True, help="The type of model (e.g., gpt4o).")
    parser.add_argument('--task',type=str,required=True,help="the task that will evaluated")
    parser.add_argument('--metrics', type=str, nargs='+', required=True, help="List of metrics to evaluate.")
    parser.add_argument('--device', type=str, default='cuda:1', help="Device to run certain metrics on (default: cuda:0).")
    parser.add_argument('--tokenizer_model', type=str, required=True, help="Model name or path for the tokenizer.")
    parser.add_argument('--data_path', type=str, required=True, help="Path to the data CSV file.")
    parser.add_argument('--pred_path', type=str, required=True, help="Path to the predictions CSV file.")
    parser.add_argument('--log_path', type=str, required=True, help="Path to save the log file.")
    parser.add_argument('--target_column',type=str,required=True,help="name of the label column in the reference dataframe")
    parser.add_argument('--predict_column',type=str,required=True,help="name of the predict column in the prediction file from inference.py")
    return parser.parse_args()


def simple_train_arguments():
    parser = argparse.ArgumentParser(description="Train a model using LoRA one gpu")
    
    parser.add_argument('--dataset_path', type=str, required=True, help="Path to the dataset")
    parser.add_argument('--model_id', type=str, required=True, help="Model ID to use for training")
    parser.add_argument('--max_seq_length', type=int, default=8192, help="Maximum sequence length")

    # Training arguments
    parser.add_argument('--output_dir', type=str, required=True, help="Output directory for the model checkpoints")
    parser.add_argument('--report_to', type=str, required=False, default="all", help="Platform to log and track with")
    parser.add_argument('--run_name', type=str, required=True, help="Name for this specific run")
    parser.add_argument('--num_train_epochs', type=int, default=3, help="Number of training epochs")
    parser.add_argument('--per_device_train_batch_size', type=int, default=1, help="Batch size per device during training")
    parser.add_argument('--per_device_eval_batch_size', type=int, default=1, help="Batch size per device during evaluation")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2, help="Number of steps before a backward/update pass")
    parser.add_argument('--logging_steps', type=int, default=5, help="Number of steps between logging")
    parser.add_argument('--save_strategy', type=str, default="epoch", help="Checkpoint save strategy")
    parser.add_argument('--evaluation_strategy', type=str, default="epoch", help="Evaluation strategy")
    parser.add_argument('--optim', type=str, default="adamw_torch", help="Optimizer type")
    parser.add_argument('--learning_rate', type=float, default=0.0002, help="Learning rate")
    parser.add_argument('--max_grad_norm', type=float, default=0.3, help="Max gradient norm")
    parser.add_argument('--warmup_ratio', type=float, default=0.03, help="Warmup ratio for learning rate scheduler")
    parser.add_argument('--lr_scheduler_type', type=str, default="constant", help="Learning rate scheduler type")
    parser.add_argument('--gradient_checkpointing', action='store_true', help="Enable gradient checkpointing")
    parser.add_argument('--bf16', action='store_true', help="Use bfloat16 precision")
    parser.add_argument('--tf32', action='store_true', help="Use tf32 precision")

    # LoRA configuration
    parser.add_argument('--lora_alpha', type=int, default=32, help="Alpha parameter for LoRA")
    parser.add_argument('--lora_dropout', type=float, default=0.05, help="Dropout rate for LoRA")
    parser.add_argument('--lora_rank', type=int, default=32, help="Rank parameter for LoRA")
    parser.add_argument('--task_type', type=str, default="CAUSAL_LM", help="Task type for LoRA")
    parser.add_argument('--bias', type=str, default="none", help="Bias parameter for LoRA")

    # Miscellaneous
    parser.add_argument('--seed', type=int, default=42, help="Random seed for initialization")
    parser.add_argument('--attn_implementation', type=str, default="flash_attention_2", help="Attention implementation to use")
    
    args = parser.parse_args()
    return args

    
def fsdp_train_arguments():
    parser = argparse.ArgumentParser(description="Train a model using LoRA with FSDP and quantization")
    
    # Dataset and model parameters
    parser.add_argument('--dataset_path', type=str, required=True, help="Path to the dataset")
    parser.add_argument('--model_id', type=str, required=True, help="Model ID to use for training")
    parser.add_argument('--max_seq_length', type=int, default=8192, help="Maximum sequence length")

    # Training arguments
    parser.add_argument('--output_dir', type=str, required=True, help="Output directory for the model checkpoints")
    parser.add_argument('--report_to', type=str, required=False, default="all", help="Platform to log and track with")
    parser.add_argument('--run_name', type=str, required=True, help="Name for this specific run")
    parser.add_argument('--num_train_epochs', type=int, default=3, help="Number of training epochs")
    parser.add_argument('--per_device_train_batch_size', type=int, default=1, help="Batch size per device during training")
    parser.add_argument('--per_device_eval_batch_size', type=int, default=1, help="Batch size per device during evaluation")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2, help="Number of steps before a backward/update pass")
    parser.add_argument('--logging_steps', type=int, default=5, help="Number of steps between logging")
    parser.add_argument('--save_strategy', type=str, default="epoch", help="Checkpoint save strategy")
    parser.add_argument('--evaluation_strategy', type=str, default="epoch", help="Evaluation strategy")
    parser.add_argument('--optim', type=str, default="adamw_torch", help="Optimizer type")
    parser.add_argument('--learning_rate', type=float, default=0.0002, help="Learning rate")
    parser.add_argument('--max_grad_norm', type=float, default=0.3, help="Max gradient norm")
    parser.add_argument('--warmup_ratio', type=float, default=0.03, help="Warmup ratio for learning rate scheduler")
    parser.add_argument('--lr_scheduler_type', type=str, default="constant", help="Learning rate scheduler type")
    parser.add_argument('--gradient_checkpointing', action='store_true', help="Enable gradient checkpointing")
    parser.add_argument('--bf16', action='store_true', help="Use bfloat16 precision")
    parser.add_argument('--tf32', action='store_true', help="Use tf32 precision")
    
    # FSDP configuration
    parser.add_argument('--fsdp', type=str, default="full_shard auto_wrap offload", help="FSDP configuration")
    parser.add_argument('--fsdp_backward_prefetch', type=str, default="backward_pre", help="FSDP backward prefetch")
    parser.add_argument('--fsdp_forward_prefetch', type=bool, default=False, help="FSDP forward prefetch")
    parser.add_argument('--use_orig_params', type=bool, default=False, help="Use original parameters in FSDP")
    parser.add_argument('--fsdp_activation_checkpointing', type=str, default="false", help="Apply gradient checkpointing")

    # LoRA configuration
    parser.add_argument('--lora_alpha', type=int, default=32, help="Alpha parameter for LoRA")
    parser.add_argument('--lora_dropout', type=float, default=0.05, help="Dropout rate for LoRA")
    parser.add_argument('--lora_rank', type=int, default=32, help="Rank parameter for LoRA")
    parser.add_argument('--task_type', type=str, default="CAUSAL_LM", help="Task type for LoRA")
    parser.add_argument('--bias', type=str, default="none", help="Bias parameter for LoRA")

    # Miscellaneous
    parser.add_argument('--seed', type=int, default=42, help="Random seed for initialization")
    parser.add_argument('--attn_implementation', type=str, default="flash_attention_2", help="Attention implementation to use")
    
    args = parser.parse_args()
    return args
