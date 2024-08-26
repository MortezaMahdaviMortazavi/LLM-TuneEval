MRC_TEMPLATE = """
You are a highly accurate question-answering model. Given a passage and a natural language question, your task is to provide a precise answer based on the passage.

### Passage:
{context}

### Question:
{question}

### Response:
{output}
"""

MRC_TEMPLATE_DETAILED = """
You are a highly accurate question-answering model. Given a passage and a natural language question, your task is to provide a detailed and comprehensive answer based on the passage.

### Passage:
{context}

### Question:
{question}

### Detailed Response:
{output}
"""

SUMMARIZATION_TEMPLATE = """
You are a powerful text summarization model. Given a passage, your job is to generate a concise and coherent summary that captures the key points.

### Passage:
{context}

### Summary:
{output}
"""

SUMMARIZATION_TEMPLATE_DETAILED = """
You are a powerful text summarization model. Given a passage, your job is to generate a detailed and coherent summary that captures both key points and supporting details.

### Passage:
{context}

### Detailed Summary:
{output}
"""
