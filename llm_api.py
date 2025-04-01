import os
from config import OPENAI_MODEL, FAIL_USER_PROMPT
from openai import OpenAI
from dotenv import load_dotenv
from lgger import logger

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

        def prepare_chatgpt_msg(self, text, chatgpt_messages):
            chatgpt_messages.append({"role": "user", "content": text})

        def extract_metadata_with_openai(self, user_prompt: str, carrier_name: Any = None, max_attempts: int = 5) -> Dict[str, Any]:
                """
                Extract structured metadata from text using OpenAI's models.
                Will retry up to `max_attempts` times if it fails to parse valid JSON.
                """

                #base_user_prompt = self.base_user_prompt.format(carrier_name=carrier_name, text=text)


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
                            temperature=0,
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

                        # Process null values to ensure required keys are present
                        for key in [
                            "geographical_region",
                            "coverage",
                            "capacity",
                            "limit",
                            "Natural_disaster",
                        ]:
                            if key not in metadata or metadata[key] is None:
                                if key in ["coverage", "capacity", "limit"]:
                                    metadata[key] = None
                                elif key in ["geographical_region", "Natural_disaster"]:
                                    metadata[key] = {"include": None, "exclude": None}

                        # Ensure nested null values
                        for nested_key in ["geographical_region", "Natural_disaster"]:
                            if nested_key in metadata and metadata[nested_key] is not None:
                                for sub_key in ["include", "exclude"]:
                                    if (
                                        sub_key not in metadata[nested_key]
                                        or metadata[nested_key][sub_key] is None
                                    ):
                                        metadata[nested_key][sub_key] = None

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
                            user_prompt = FAIL_USER_PROMPT.format(attempt=attempt,last_error_message=last_error_message, last_response_message=last_response_message)

                            messages = prepare_chatgpt_msg(text, messages)
                        

                # ---------------------------
                # Fallback: Return a default
                # ---------------------------
                self.logger.error(
                    "Failed to parse valid metadata from OpenAI after %d attempts.",
                    max_attempts
                )
                return {
                    "carrier_name": carrier_name,
                    "geographical_region": {"include": None, "exclude": None},
                    "coverage": None,
                    "capacity": None,
                    "limit": None,
                    "Natural_disaster": {"include": None, "exclude": None},
                }
