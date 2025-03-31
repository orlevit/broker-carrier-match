import os
import openai
from dotenv import load_dotenv

# Load the API key from .env file
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    print("Error: OPENAI_API_KEY not found in .env file")
    exit(1)

# Set the API key
openai.api_key = api_key

# Test the API key with a simple request
try:
    response = openai.models.list()
    print("✅ API key is valid!")
except openai.error.AuthenticationError:
    print("❌ Invalid API key! Please check your .env file.")
except openai.error.OpenAIError as e:
    print(f"⚠️ OpenAI API error: {e}")

