import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
import pandas as pd
from tqdm import tqdm
from peft import PeftModel
from trl import setup_chat_format
import argparse

class ConversationSeparator:
    @staticmethod
    def separate(text):
        parts = text.split('assistant', 1)
        user_part = parts[0].replace('user', '').strip()
        assistant_part = parts[1].strip() if len(parts) > 1 else ''
        return {
            "user": user_part,
            "assistant": assistant_part
        }

class LlamaModelInference:
    def __init__(self, model_name, device="cuda:1", load_in_4bit=True, lora_weights=None):
        self.device = device
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = self._load_model(load_in_4bit, lora_weights)
        self.pipe = self._setup_pipeline()

    def _load_model(self, load_in_4bit, lora_weights):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            quantization_config=bnb_config,
            device_map={"": torch.device(self.device)},
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )

        if lora_weights:
            model, self.tokenizer = self._load_lora_weights(model, lora_weights)
        else:
            model, self.tokenizer = setup_chat_format(model, self.tokenizer)
        
        return model

    def _load_lora_weights(self, model, lora_weights):
        model, self.tokenizer = setup_chat_format(model, self.tokenizer)
        if isinstance(lora_weights, str):
            lora_weights = [lora_weights]

        for lora_weight in lora_weights:
            model = PeftModel.from_pretrained(model, lora_weight, torch_device=self.device)
            model = model.merge_and_unload()

        return model, self.tokenizer

    def _setup_pipeline(self):
        return pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            torch_dtype=torch.float16,
            device_map=self.device,
        )

    def generate_responses(self, texts, max_new_tokens, do_sample, temperature, top_k, top_p, batch_size):
        responses = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating responses"):
            batch_texts = texts[i:i + batch_size]
            batch_prompts = [self._prepare_prompt(text) for text in batch_texts]
            outputs = self.pipe(batch_prompts, max_new_tokens=max_new_tokens, do_sample=do_sample,
                                temperature=temperature, top_k=top_k, top_p=top_p)
            responses.extend(outputs)
            torch.cuda.empty_cache()
        
        return [response[0]['generated_text'] for response in responses]

    def _prepare_prompt(self, text):
        messages = [{"role": "user", "content": text}]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

class InferencePipeline:
    def __init__(self, model_name, data_path, output_path, lora_weights=None, device="cuda:1"):
        self.data_path = data_path
        self.output_path = output_path
        self.model_inference = LlamaModelInference(model_name, device=device, lora_weights=lora_weights)

    def run(self, max_new_tokens, do_sample, temperature, top_k, top_p, batch_size):
        texts = self._load_texts()
        raw_responses = self.model_inference.generate_responses(
            texts,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            batch_size=batch_size
        )
        cleaned_responses = self._clean_responses(raw_responses)
        self._save_responses(cleaned_responses)

    def _load_texts(self):
        df = pd.read_csv(self.data_path)
        return df['context'].tolist()

    def _clean_responses(self, responses):
        return [ConversationSeparator.separate(res) for res in responses]

    def _save_responses(self, responses):
        df = pd.DataFrame(responses)
        df.to_csv(self.output_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLaMA Model Inference Pipeline")
    parser.add_argument('--model_name', type=str, required=True, help="The name or path of the base model to load")
    parser.add_argument('--device', type=str, default="cuda:1", help="Device to run the model on")
    parser.add_argument('--load_in_4bit', action='store_true', help="Whether to load the model in 4-bit precision")
    parser.add_argument('--lora_weights', type=str, nargs='*', help="Path to one or more LoRA weights to load and merge")
    parser.add_argument('--data_path', type=str, required=True, help="Path to the input CSV file")
    parser.add_argument('--output_path', type=str, required=True, help="Path to save the output CSV file")
    parser.add_argument('--max_new_tokens', type=int, default=4096, help="Maximum number of tokens to read and generate")
    parser.add_argument('--do_sample', action='store_true', help="Whether to use sampling for generation")
    parser.add_argument('--temperature', type=float, default=0.1, help="Sampling temperature")
    parser.add_argument('--top_k', type=int, default=10, help="Top-k sampling")
    parser.add_argument('--top_p', type=float, default=0.95, help="Top-p sampling")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for generation")
    
    args = parser.parse_args()

    inference_pipeline = InferencePipeline(
        model_name=args.model_name,
        data_path=args.data_path,
        output_path=args.output_path,
        lora_weights=args.lora_weights,
        device=args.device
    )
    
    inference_pipeline.run(
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        batch_size=args.batch_size
    )

