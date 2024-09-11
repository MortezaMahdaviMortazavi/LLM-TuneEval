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
