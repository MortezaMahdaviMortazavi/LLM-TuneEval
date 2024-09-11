import os
import pandas as pd
import argparse
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

SYSTEM_MESSAGE = """
You are a highly skilled language model proficient in understanding and generating text in Farsi. Focus on providing accurate, context-aware, and fluent responses, ensuring that your outputs are clear, concise, and aligned with the intended purpose. Always maintain a high level of comprehension and detail in your responses, reflecting a deep understanding of the Persian language and context
""".strip()


def create_conversation(sample, system_message=SYSTEM_MESSAGE):
    """
    Adds a system message to the conversation if not already present.
    """
    if sample["messages"][0]["role"] == "system" or not system_message:
        return sample
    sample["messages"] = [{"role": "system", "content": system_message}] + sample["messages"]
    return sample

def format_chat_template(row, tokenizer, system_prompt=None):
    """
    Format a row of data into a chat template using the tokenizer and optionally include a system prompt.
    """
    row_json = []
    if system_prompt:
        row_json.append({"role": "system", "content": system_prompt})
    row_json.append({"role": "user", "content": row["inputs"]})
    row_json.append({"role": "assistant", "content": row["targets"]})
    
    row["prompt"] = tokenizer.apply_chat_template(row_json, tokenize=False)
    return {"prompt": row["prompt"]}

def preprocess_and_save(df_path: str, save_path: str, model_id: str, test_size: float = None):
    """
    Preprocess the dataset and optionally perform a train-test split, then save the data.

    Args:
        df_path (str): The path to the input CSV file.
        save_path (str): The directory where the processed datasets will be saved.
        model_id (str): The model ID for the tokenizer.
        test_size (float, optional): The proportion of the dataset to include in the test split. 
                                     If None, no splitting is performed.
    """
    # Load and preprocess the data
    df = pd.read_csv(df_path)
    dataset = Dataset.from_pandas(df)
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    dataset = dataset.map(lambda row: format_chat_template(row, tokenizer,system_prompt=SYSTEM_MESSAGE), num_proc=6)

    # Handle splitting
    if test_size is not None:
        dataset = dataset.train_test_split(test_size=test_size)
        save_dataset(dataset['train'], save_path, "train_dataset.json")
        save_dataset(dataset['test'], save_path, "test_dataset.json")
    else:
        save_dataset(dataset, save_path, "train_dataset.json")

def save_dataset(dataset, save_path, filename):
    """
    Save the dataset to the specified path as a JSON file.
    
    Args:
        dataset (Dataset): The dataset to save.
        save_path (str): The directory where the file will be saved.
        filename (str): The name of the output file.
    """
    os.makedirs(save_path, exist_ok=True)
    dataset.to_json(os.path.join(save_path, filename), orient="records", force_ascii=False)

def main():
    """
    Main function to parse command-line arguments and run the preprocessing function.
    """
    parser = argparse.ArgumentParser(description="Preprocess dataset for training.")
    parser.add_argument('--input_path', type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument('--save_path', type=str, required=True, help="Directory to save the processed datasets.")
    parser.add_argument('--model_id', type=str, required=True, help="Model ID for the tokenizer.")
    parser.add_argument('--test_size', type=float, default=None, help="Proportion of the dataset to use as test data. If not provided, no splitting is performed.")
    
    args = parser.parse_args()
    preprocess_and_save(df_path=args.input_path, save_path=args.save_path, model_id=args.model_id, test_size=args.test_size)

if __name__ == "__main__":
    main()
