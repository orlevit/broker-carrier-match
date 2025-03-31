import os

# Directories
DATA_DIR = "data"
INPUT_DIR =os.path.join(DATA_DIR, "input") 
OUTPUT_DIR =os.path.join(DATA_DIR, "output") 
DB_PATH = os.path.join(OUTPUT_DIR, "guide_db")
CHUNK_RESULTS_FILE = os.path.join(OUTPUT_DIR, "chunks.json")
STRUCTURED_METADATA_FILE = os.path.join(OUTPUT_DIR, "structured_metadata.json")
SECTION_MAP_FILE = os.path.join(OUTPUT_DIR, "section_map.json")

# Hyper-parameters
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
MAX_HEADER_DEPTH = 6

# LLM
OPENAI_MODEL = "gpt-4-turbo"
SYSTEM_PROMPT = (
    "You are an insurance domain expert who extracts structured metadata from "
    "carrier appetite guides. Extract information ONLY from the provided text. "
    "Do not infer additional information. If a requested field is not explicitly "
    "mentioned in the text, set it to null. Format in the exact JSON schema requested."
)
BASE_USER_PROMPT = """Extract metadata ONLY from this insurance carrier appetite guide.
    Do NOT include information from other sections or make assumptions about content not present in this text.
            
    Carrier Name: {carrier_name}

Text:
        {text}

Extract this information into a structured JSON with the exact following schema:
    {{
    "carrier_name": "{carrier_name}",
    "geographical_region": {{
          "include": [list of included regions ONLY from this text (or null if not mentioned)], 
  "exclude": [list of excluded regions ONLY from this text (or null if not mentioned)]
              }},
    "coverage": [list of coverage types ONLY from this text (or null if not mentioned)],
    "capacity": [list of dictionaries in the form {{"name": "price description"}} ONLY from this text (or None if not mentioned)],
    "limit": [list of dictionaries in the form {{"name": "price description"}} ONLY from this text (or None if not mentioned)],
    "Natural_disaster": {{
          "include": [list of included natural disasters ONLY from this text (or None if not mentioned)], 
  "exclude": [list of excluded natural disasters ONLY from this text (or None if not mentioned)]
              }}
              }}

              IMPORTANT:
              - capacity is the maximum risk an insurer can underwrite.
              - coverage refers to the specific risks or damages the policy protects against.
              - limit is the maximum amount the insurer will pay for a specific claim.
              - Use "None" when information is not present in this specific text
- Do NOT include information that isn't explicitly stated in this text
- Do NOT try to fill in fields by inferring from related information
    - If you're unsure, set a field to None instead
- Return only the JSON with no additional explanation.
    """
FAIL_USER_PROMPT = (
    "The previous attempt to parse valid JSON from the text failed. "
    "(attempt #{attempt}). The error was: {last_error_message} "
    "Your output was: {last_response_message}. "
    "Please respond with ONLY a valid JSON dictionary according to the exact schema. "
    "No additional text, explanation, or code fence. Do not wrap it in backticks.\n"
)
