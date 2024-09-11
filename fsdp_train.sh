# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:4096
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#
export TOKENIZERS_PARALLELISM=true
export CUDA_VISIBLE_DEVICES=0,1,2
export ACCELERATE_USE_FSDP=1
export FSDP_CPU_RAM_EFFICIENT_LOADING=1
torchrun --nproc_per_node=3 scripts/fsdp_train.py \
    --dataset_path "data/TRAIN" \
    --model_id "unsloth/Meta-Llama-3.1-8B-Instruct" \
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
