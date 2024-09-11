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
  --lora_weights None \


