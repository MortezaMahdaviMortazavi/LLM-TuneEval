import pandas as pd
import logging
import torch
import os
from tqdm import tqdm
from transformers import AutoTokenizer
from multiprocessing import Pool , cpu_count
from datasets import load_dataset


import pandas as pd
from tqdm import tqdm


class DataPreparator:

    @staticmethod
    def get_prompt_templates():
        """Returns a dictionary of all summary prompt templates."""
        return {
            "one_line_summary": (
                "Summarize the following context into one concise sentence in Persian that captures the main idea."
                "\nContext: {context}"
                "\n- The sentence should be clear, to the point, and capture the essence of the content."
                "\n- Avoid unnecessary details; focus on the main message."
            ),
            "title_generation": (
                "Generate one short, descriptive, and engaging title in Persian for the following context."
                "\nContext: {context}"
                "\n- The title should reflect the main theme or topic."
                "\n- Use clear and straightforward language."
                "\n- Make sure only one title is generated without any additional explanation."
            ),
            "detailed_summary": (
                "Provide a comprehensive and detailed summary in Persian that covers all key points of the following context."
                "\nContext: {context}"
                "\n- Include all important information and details."
                "\n- The summary should be in paragraph form, written in complete sentences."
                "\n- Avoid personal opinions; focus on factual content."
            ),
            "bullet_point_summary": (
                "Summarize the main points of the following context in Persian as bullet points."
                "\nContext: {context}"
                "\n- Start each point with a dash in a new line."
                "\n- Each point should be a brief but complete thought."
                "\n- Include only the most important and relevant information."
                "\n- Ensure clarity and conciseness in each point."
            )
        }

    @staticmethod
    def _format_prompt(template: str, context: str) -> str:
        """Formats a prompt template with the provided context."""
        return template.format(context=context)

    @staticmethod
    def summary_prompt_creation(df: pd.DataFrame) -> pd.DataFrame:
        """Generates summary prompts and appends them to the DataFrame."""
        if list(df.columns) == ['context','one_line_summary','title_generation','detailed_summary','bullet_point_summary']:
            df = DataPreparator._filter_summarization(df)
            df = DataPreparator._flat_summary_dataset(df)
            
        prompt_templates = DataPreparator.get_prompt_templates()
        contexts = [
            DataPreparator._format_prompt(prompt_templates[row['type']], row['context'])
            for idx, row in tqdm(df.iterrows(), total=len(df))
        ]
        df['context'] = contexts
        df['inputs'] = df['context']
        df['targets'] = df['summary']
        return df[['inputs', 'targets']]

    @staticmethod
    def _flat_summary_dataset(df: pd.DataFrame) -> pd.DataFrame:
        """Creates a dataset for summary tasks by expanding multiple types of summaries for each context."""
        rows = []
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            for summary_type in ['one_line_summary', 'title_generation', 'detailed_summary', 'bullet_point_summary']:
                rows.append({
                    "context": row['context'],
                    "summary": row[summary_type],
                    "type": summary_type
                })
        return pd.DataFrame(rows)

    @staticmethod
    def _filter_summarization(df:pd.DataFrame) -> pd.DataFrame:
        df_filtered = df[~df['context'].str.contains("به تازگی فعال نبوده", na=False)]
        df_filtered = df_filtered[~df_filtered['context'].str.contains("نمایش پروفایل", na=False)]
        df_filtered = df_filtered[~df_filtered['context'].str.contains("RSS", na=False)]
        df_filtered = df_filtered[~df_filtered['context'].str.contains("Votes", na=False)]
        return df_filtered

    @staticmethod
    def create_qa_prompt(row: pd.Series) -> str:
        """Creates a QA prompt for a single row."""
        return f"""
        You are given five questions and one target question. 
        Your task is to identify which question from the list is equivalent in meaning or topic to the target question. 
        Return **only** the number (1, 2, 3, 4, or 5) of the question that matches the target. 
        Do not provide any additional information or explanation—just the number.

        Questions:
        1. {row['inputs'][0]['question']}
        2. {row['inputs'][1]['question']}
        3. {row['inputs'][2]['question']}
        4. {row['inputs'][3]['question']}
        5. {row['inputs'][4]['question']}

        Target Question: {row['main_question']['formal']}
        
        Equivalent question number:
        """.strip()

    @staticmethod
    def qa_prompt_creation(df: pd.DataFrame) -> pd.DataFrame:
        """Generates QA prompts and appends them to the DataFrame."""
        qa_inputs = [
            DataPreparator.create_qa_prompt(row) for idx, row in tqdm(df.iterrows(), total=len(df))
        ]
        df['inputs'] = qa_inputs
        df['targets'] = df['target']
        return df[['inputs', 'targets']]

    @staticmethod
    def create_mrc_prompt(row: pd.Series) -> str:
        """Creates an MRC prompt for a single row."""
        return f"""
        This is a Machine Reading Comprehension (MRC) task focused on exact answer extraction. Follow these guidelines:
        1. **Exact Answer Extraction**: If the question's answer is explicitly stated in the context, extract the shortest possible answer.
        2. **No Answer Available**: If the text does not contain an answer, respond with: 'این اطلاعات در متن موجود نیست'.
        3. **Precision**: Provide answers that are accurate and free from errors. Do not add any information not present in the text.
        4. **Language**: All responses must be in FARSI.
        5. **Conciseness**: Be succinct. Respond only with the answer or the specified phrase if no answer is available.
        6. **No Inference**: Do not infer or generate answers. Respond only with information directly available from the text.

        Context: {row['context']}
        Question: {row['question']}
        """.strip()

    @staticmethod
    def _flat_mrc_dataset(df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates a DataFrame of question-answer pairs from factual and unanswerable questions.
        
        Args:
            df (pd.DataFrame): Input DataFrame with 'context', 'factual_questions', and 'unanswerable_question' columns.
        
        Returns:
            pd.DataFrame: A DataFrame with 'context', 'question', and 'answer' columns.
        """
        samples = []
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            factual_questions = row['factual_questions']
            unanswerable_question = row['unanswerable_question']
            
            # Add factual questions
            for item in factual_questions:
                samples.append({
                    "context": row['context'],
                    "question": item['question'],
                    "answer": item['answer']
                })
            
            # Add unanswerable question
            samples.append({
                "context": row['context'],
                "question": unanswerable_question['question'],
                "answer": unanswerable_question['answer']
            })
        
        return pd.DataFrame(samples)

    @staticmethod
    def mrc_prompt_creation(df: pd.DataFrame) -> pd.DataFrame:
        """Generates MRC prompts and appends them to the DataFrame."""
        if list(df.columns) == ['context', 'factual_questions', 'unanswerable_question']:
            df = DataPreparator._flat_mrc_dataset(df)
            
        df = df.dropna()
        inputs = [
            DataPreparator.create_mrc_prompt(row) for idx, row in tqdm(df.iterrows(), total=len(df))
        ]
        df['inputs'] = inputs
        df['targets'] = df['']
        return df[['inputs', 'targets']]







def load_datasets(dataset_path: str):
    train_dataset = load_dataset(
        "json",
        data_files=os.path.join(dataset_path, "train_dataset.json"),
        split="train",
    )
    test_dataset = load_dataset(
        "json",
        data_files=os.path.join(dataset_path, "test_dataset.json"),
        split="train",
    )
    return train_dataset, test_dataset




def count_total_tokens(model_id: str, texts: list) -> int:
    num_processes = cpu_count()
    chunk_size = len(texts) // num_processes + 1
    text_chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]
    
    with Pool(processes=num_processes) as pool:
        total_tokens = sum(tqdm(pool.imap(tokenize_chunk, [(model_id, chunk) for chunk in text_chunks]), 
                                total=len(text_chunks), 
                                desc="Processing chunks", 
                                unit="chunk"))
        
    return total_tokens

def count_parameters(model):
    """
    Count the number of trainable (unfrozen) and non-trainable (frozen) parameters in the model.

    Args:
    model (torch.nn.Module): The model to count parameters for.

    Returns:
    None
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Trainable parameters: {trainable_params}")
    print(f"Frozen parameters: {frozen_params}")



def read_and_concat_jsonl(directory_path):
    """
    Reads all .jsonl files from a directory and concatenates them into a single DataFrame.

    Parameters:
    - directory_path (str): Path to the directory containing the .jsonl files.

    Returns:
    - pd.DataFrame: A concatenated DataFrame containing data from all .jsonl files.
    """
    df_list = []
    for filename in os.listdir(directory_path):
        if filename.endswith('.jsonl'):
            file_path = os.path.join(directory_path, filename)
            df = pd.read_json(file_path, lines=True)
            df_list.append(df)

    combined_df = pd.concat(df_list, ignore_index=True)
    combined_df = combined_df.dropna()
    return combined_df


