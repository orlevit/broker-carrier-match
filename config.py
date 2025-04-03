import os

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
INPUT_DIR =os.path.join(DATA_DIR, "input") 
OUTPUT_DIR =os.path.join(DATA_DIR, "output") 
LOGS_DIR =os.path.join(OUTPUT_DIR, "logs") 
DB_PATH = os.path.join(OUTPUT_DIR, "guide_db")

# Files
INPUT_GUIDE_FILE = os.path.join(INPUT_DIR, "guide.json")
CHUNK_RESULTS_FILE = os.path.join(OUTPUT_DIR, "chunks.json")
STRUCTURED_METADATA_FILE = os.path.join(OUTPUT_DIR, "structured_metadata.json")
SECTION_MAP_FILE = os.path.join(OUTPUT_DIR, "section_map.json")
APP_LOGS = os.path.join(LOGS_DIR, "app.log")


# Hyper-parameters
COLLECTION_NAME = "docs"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
MAX_HEADER_DEPTH = 6

# LLM
TEMPERATURE = 0
DEFAULT_K_TOP_SIMILAR = 5
DEFAULT_MAX_NUM_TOKENS = 4000
OPENAI_MODEL = "gpt-4-turbo"
TOP_FILTER_NUM = 10
TOP_SIMILARITY_NUM = 5
TOTAL_TOP_SIMILAR = 20


# prompts
#Metadata

NEEDED_FIELD_NAMES = [
    "geographical_region.include",
    "geographical_region.exclude",
    "coverage",
    "capacity",
    "limit",
    "Natural_disaster.include",
    "Natural_disaster.exclude" ]

## summpary prompt
SUMMARY_SYSTEM =  "Generate a brief summary of the following insurance policy text in 1-2 sentences."

# Additional guide
IMPORTANT_MSG = """ IMPORTANT:
- capacity is the maximum risk an insurer can underwrite.
- coverage refers to the specific risks or damages the policy protects against.
- limit is the maximum amount the insurer will pay for a specific claim.
- Use "''" when information is not present in this specific text.
- Do NOT include information that isn't explicitly stated in this text.
- Do NOT try to fill in fields by inferring from related information.
- If you're unsure, set a field to '' instead.
- Return only the JSON with no additional explanation.
- Do not change the dictionary field names.
- Do not add or remove fields from the dictionary.
- Make sure that list elemetsd are one lower case word. Only a word.
"""

## Appetite guide prompts
APETITE_SYSTEM_PROMPT = (
    "You are an insurance domain expert who extracts structured metadata from "
    "carrier appetite guides. Extract information ONLY from the provided text. "
    "Do not infer additional information. If a requested field is not explicitly "
    "mentioned in the text, set it to empty string(''). Format in the exact JSON schema requested."
)
APETITE_BASE_USER_PROMPT = """Extract metadata ONLY from this insurance carrier appetite guide.
    Do NOT include information from other sections or make assumptions about content not present in this text.
            
    Carrier Name: {{carrier_name}}

Text:
        {{text}}

Extract this information into a structured JSON with the exact following schema:
    {{
    "carrier_name": "{carrier_name}",
    "geographical_region.include": [list of included regions ONLY from this text (or '' if not mentioned). Each element in the list should be one word in lower case], 
    "geographical_region.exclude": [list of excluded regions ONLY from this text (or '' if not mentioned) Each element in the list should be one word in lower case],
    "coverage": [list of coverage types ONLY from this text (or '' if not mentioned) Each element in the list should be one word in lower case],
    "capacity": [list of capacity types ONLY from this text (or '' if not mentioned) Each element in the list should be one word in lower case],
    "limit": [list of limit types ONLY from this text (or '' if not mentioned) Each element in the list should be one word in lower case],
    "Natural_disaster.include": [list of included natural disasters ONLY from this text (or '' if not mentioned). Each element in the list should be one word in lower case], 
    "Natural_disaster.exclude": [list of excluded natural disasters ONLY from this text (or '' if not mentioned). Each element in the list should be one word in lower case]
    }}

    """\
    f"{IMPORTANT_MSG}"
## User question prompt
Q_SYSTEM_PROMPT = ("You are a metadata extraction tool. Your task is to analyze a user question and extract potential metadata"
    "Extract information ONLY from the provided text. "
    "Do not infer additional information. If a requested field is not explicitly "
    "mentioned in the text, set it to empty string (''). Format in the exact JSON schema requested.")

Q_BASE_USER_PROMPT = """For the following user question:\n

{user_question}

You should return a JSON object with the following structure:
    {{
    "carrier_name": the carrier name if exists, otherwise empty string (''),
    "geographical_region.include": [list of included regions ONLY from this text (or '' if not mentioned). Each element in the list should be one word in lower case], 
    "geographical_region.exclude": [list of excluded regions ONLY from this text (or '' if not mentioned) Each element in the list should be one word in lower case],
    "coverage": [list of coverage types ONLY from this text (or '' if not mentioned) Each element in the list should be one word in lower case],
    "capacity": [list of capacity types ONLY from this text (or '' if not mentioned) Each element in the list should be one word in lower case],
    "limit": [list of limit types ONLY from this text (or '' if not mentioned) Each element in the list should be one word in lower case],
    "Natural_disaster.include": [list of included natural disasters ONLY from this text (or '' if not mentioned). Each element in the list should be one word in lower case], 
    "Natural_disaster.exclude": [list of excluded natural disasters ONLY from this text (or '' if not mentioned). Each element in the list should be one word in lower case]
    }}"""\
    f"{IMPORTANT_MSG}"

RESPONSE_SYSTEM_PROMPT = "You are a polite and well-articulated insurance expert specializing in carrier appetite guides."
RESPONSE = """
Below is a summary of the carrier appetite guide, relevant for answering the user's question:

{summary}

Full context:

{context}

Generate a coherent and accurate response to the following user question:

{user_question}
"""

FAIL_USER_PROMPT = (
    "The previous attempt to parse valid JSON from the text failed. "
    "(attempt #{attempt}). The error was: {last_error_message} "
    "Your output was: {last_response_message}. "
    f"{IMPORTANT_MSG}"
)
