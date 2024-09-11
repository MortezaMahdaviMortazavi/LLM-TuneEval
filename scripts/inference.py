import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
import pandas as pd
from tqdm import tqdm
from peft import PeftModel
from trl import setup_chat_format
from args import inference_arguments
from liger_kernel.transformers import apply_liger_kernel_to_llama



class DataProcessor:
    @staticmethod
    def load_texts(data_path, prompt_column):
        df = pd.read_csv(data_path)
        return df[prompt_column].tolist()

    @staticmethod
    def clean_responses(responses):
        return [DataProcessor.separate(res) for res in responses]

    @staticmethod
    def save_responses(responses, output_path):
        df = pd.DataFrame(responses)
        df.to_csv(output_path, index=False)


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
        print("Applying Liger kernel to LLaMA...")
        apply_liger_kernel_to_llama()
        print("Liger kernel applied successfully.")

        if lora_weights != None:
            model, self.tokenizer = self._load_lora_weights(model, lora_weights)
        else:
            model, self.tokenizer = setup_chat_format(model, self.tokenizer)
        
        return torch.compile(model)

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
            outputs = self.pipe(batch_prompts,max_new_tokens=max_new_tokens,  # max_length=max_new_tokens,
                                do_sample=do_sample, temperature=temperature, top_k=top_k, top_p=top_p)
            responses.extend(outputs)
            torch.cuda.empty_cache()
        
        return [response[0]['generated_text'] for response in responses]

    def _prepare_prompt(self, text, system_prompt=None):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": text})
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)




if __name__ == "__main__":
    args = inference_arguments()

    # Initialize the inference pipeline
    inference_pipeline = LlamaModelInference(
        model_name=args.model_name,
        device=args.device,
        load_in_4bit=args.load_in_4bit,
        lora_weights=args.lora_weights
    )

    # Load data
    texts = DataProcessor.load_texts(args.data_path, args.prompt_column)

    # Run inference
    raw_responses = inference_pipeline.generate_responses(
        texts=texts,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        batch_size=args.batch_size
    )

    # Clean and save responses
    cleaned_responses = DataProcessor.clean_responses(raw_responses)
    DataProcessor.save_responses(cleaned_responses, args.output_path)
