MODEL_NAME="unsloth/Meta-Llama-3.1-8B-Instruct"
DEVICE="cuda:0"
LORA_WEIGHTS="models/llama-fewshot-mrc-lora32"
DATA_PATH="data/mrc_gpt4o_prepare_for_inference_fewshot.csv"
OUTPUT_PATH="loggers/cleaned-llamaLora32Fewshot-Gpt4oGenerated-Mrc.csv"
PROMPT_COLUMN="prompt"
MAX_NEW_TOKENS=4096
BATCH_SIZE=64

echo "Starting Inference..."
python src/inference.py \
  --model_name "$MODEL_NAME" \
  --device "$DEVICE" \
  --load_in_4bit \
  --lora_weights "$LORA_WEIGHTS" \
  --data_path "$DATA_PATH" \
  --output_path "$OUTPUT_PATH" \
  --prompt_column "$PROMPT_COLUMN" \
  --max_new_tokens $MAX_NEW_TOKENS \
  --do_sample \
  --temperature 0.1 \
  --top_k 10 \
  --top_p 0.95 \
  --batch_size $BATCH_SIZE

if [ $? -ne 0 ]; then
  echo "Inference step failed. Exiting."
  exit 1
fi
echo "Inference Completed Successfully."

echo "Starting Evaluation..."

MODEL_TYPE="llamaBased"
TASK="mrc"
METRICS="bleu rouge meteor bert_score f1_score"
TOKENIZER_MODEL="$MODEL_NAME"  # Using the same model as tokenizer
PRED_PATH="$OUTPUT_PATH"        # Using the output of inference as prediction path
LOG_PATH="logs/llamaFewshot-gpt4oGenerated-mrc.log"

# Run the Python evaluation script with the provided parameters
python src/eval.py \
  --model_type "$MODEL_TYPE" \
  --task "$TASK" \
  --metrics $METRICS \
  --tokenizer_model "$TOKENIZER_MODEL" \
  --data_path "$DATA_PATH" \
  --pred_path "$PRED_PATH" \
  --log_path "$LOG_PATH"

if [ $? -ne 0 ]; then
  echo "Evaluation step failed. Exiting."
  exit 1
fi
echo "Evaluation Completed Successfully."

echo "All steps completed successfully."
