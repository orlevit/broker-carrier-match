import os
import json
from types import prepare_class
from typing import Dict, Any
from config import OPENAI_MODEL, FAIL_USER_PROMPT, TEMPERATURE, SUMMARY_SYSTEM, NEEDED_FIELD_NAMES
from openai import OpenAI
from dotenv import load_dotenv
from logger import logger

class OpenAiAPI:
    def __init__(self, system_prompt):
        self.logger =logger
        load_dotenv()  # load .env if present
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError(
                "OpenAI API key is required. "
                "Set OPENAI_API_KEY environment variable or pass it in your .env file."
            )
        self.system_prompt= system_prompt
        self.openai_model = OPENAI_MODEL
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.logger.info("OpenAI client initialized with key from environment.")

    def prepare_chatgpt_msg(self, text, chatgpt_messages, role="user"):
        chatgpt_messages.append({"role": role, "content": text})

        return chatgpt_messages

    def chatgpt_call(self, system_text, user_text):
        messages = []
        if (system_text is not None) or (system_text != ""):
            messages = self.prepare_chatgpt_msg(system_text, messages, role="system")
            
        messages = self.prepare_chatgpt_msg(user_text, messages, role="user")

        response = self.openai_client.chat.completions.create(
            model=self.openai_model,
            messages=messages,
            temperature=TEMPERATURE,
            response_format={"type": "text"},
        )
        return response.choices[0].message.content


    def extract_metadata_with_openai(self, user_prompt: str, carrier_name: Any = None, max_attempts: int = 5) -> Dict[str, Any]:
            """
            Extract structured metadata from text using OpenAI's models.
            Will retry up to `max_attempts` times if it fails to parse valid JSON.
            """

            #base_user_prompt = self.base_user_prompt.format(carrier_name=carrier_name, text=text)

            needed_field_names = [
                "geographical_region.include",
                "geographical_region.exclude",
                "coverage",
                "capacity",
                "limit",
                "Natural_disaster.include",
                "Natural_disaster.exclude" ]

            attempt = 0
            last_error_message = ""
            last_response_message = ""
            messages = [{"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_prompt}]

            while attempt < max_attempts:
                try:
                    # ---------------------------
                    # Call OpenAI for the result
                    # ---------------------------
                    response = self.openai_client.chat.completions.create(
                        model=self.openai_model,
                        messages=messages,
                        temperature=TEMPERATURE,
                        response_format={"type": "json_object"},
                    )

                    openai_response = response.choices[0].message.content
                    last_response_message = openai_response

                    # ---------------------------
                    # Attempt to parse JSON
                    # ---------------------------
                    metadata = json.loads(openai_response)

                    # Ensure it's actually a dict
                    if not isinstance(metadata, dict):
                        raise ValueError("Parsed data is not a dictionary.")

                    # ─────────────────────────────────────────────────────────
                    # If we made it this far, we have a valid dict. Post-process it.
                    # ─────────────────────────────────────────────────────────

                    # Make sure the carrier_name is exactly as given
                    if carrier_name is not None:
                        metadata["carrier_name"] = carrier_name

                    additoinal_fields_count = len(list(set(metadata.keys()) - set(NEEDED_FIELD_NAMES))) - 1

                    if additoinal_fields_count != 0:
                        additoinal_fields= [item for item in metadata.keys() if item not in NEEDED_FIELD_NAMES]

                        raise ValueError("There are additional field in the metada dictionary that shoud not be, or the fields have improper name.\n"
                                         f"The invalid fields names: {additoinal_fields}\n"
                                         f"there should be only these names: [\"{'" , "'.join(NEEDED_FIELD_NAMES)}\"]")

                    # Process null values to ensure required keys are present
                    for key in NEEDED_FIELD_NAMES:
                        if key not in metadata:
                            metadata[key] = None

                    # If we've successfully constructed and validated the dictionary, return it!
                    return metadata

                except Exception as e:
                    # ---------------------------
                    # Retry mechanism
                    # ---------------------------
                    attempt += 1
                    last_error_message = str(e)
                    self.logger.warning(
                        "Attempt %d to parse valid JSON from OpenAI failed. Error: %s\nThe response is: %s",
                        attempt,
                        last_error_message,
                        last_response_message
                    )

                    # If we haven't reached max attempts, update the user prompt to nudge ChatGPT
                    if attempt < max_attempts:
                        # Provide some feedback
                        fail_user_prompt = FAIL_USER_PROMPT.format(attempt=attempt,last_error_message=last_error_message, last_response_message=last_response_message)

                        messages = self.prepare_chatgpt_msg(fail_user_prompt, messages)
                    

            # ---------------------------
            # Fallback: Return a default
            # ---------------------------
            self.logger.error( "Failed to parse valid metadata from OpenAI after %d attempts.", max_attempts)

            empty_dict = {}
            for field in needed_field_names:
                empty_dict[field] = None
                
            return empty_dict
