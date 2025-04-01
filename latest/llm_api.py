import os
import openai
from transformers import AutoTokenizer, AutoModel
import torch
import logging

logger = logging.getLogger(__name__)

# Load HuggingFace model for embedding (example model, can be replaced)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME)

# Set your OpenAI API key (suggested to load from environment variable)
openai.api_key = os.getenv("OPENAI_API_KEY")

def embed_text(text: str):
    """
    Generate embeddings using HuggingFace sentence transformer.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        model_output = model(**inputs)
    embeddings = mean_pooling(model_output, inputs['attention_mask'])
    return embeddings[0].tolist()

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of output contains token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_metadata_filter(question: str) -> dict:
    """
    Use OpenAI's LLM to infer the metadata filter structure from the question.
    """
    prompt = f"""
You are a helpful assistant. Extract a metadata-based filter from the following question:

QUESTION: "{question}"

Return a JSON object with the format:
{{
  "request": [list of metadata keys requested],
  "conditions": {{
    "coverage": [...],
    "Natural_disaster": {{"include": [...], "exclude": [...]}}
  }}
}}
Return only the JSON. Return null for any unknown fields.
"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You extract metadata filters from user queries."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        json_text = response['choices'][0]['message']['content']
        return json.loads(json_text)
    except Exception as e:
        logger.error(f"OpenAI metadata filter generation failed: {e}")
        raise

