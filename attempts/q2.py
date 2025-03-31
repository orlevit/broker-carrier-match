# pyright: reportAttributeAccessIssue=false

import json
import re
import os
from typing import List, Dict, Any, Tuple
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from openai import OpenAI
import markdown
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from logger import logger


class AppetiteGuideVectorDB:
    def __init__(
        self,logger_file=logger, model_name="all-MiniLM-L6-v2", max_header_depth=6, openai_api_key=None
    ):
        """
        Initialize the vector database for carrier appetite guides with local embedding model

        Args:
            model_name: Name of the HuggingFace embedding model to use
                        Default is all-MiniLM-L6-v2 which is small (80MB) but effective
            max_header_depth: Maximum number of header levels to process (default=6)
            openai_api_key: API key for OpenAI (if None, will try to load from environment)
        """
        self.logger=logger_file
        # Initialize local embedding model
        self.embeddings = ''#HuggingFaceEmbeddings(
           # model_name=model_name,
        #model_kwargs={"device": "cpu"},
       #     encode_kwargs={"normalize_embeddings": True},
       # )
        self.vector_db = None
        self.db_path = "./appetite_guide_db"
        self.model_name = model_name
        self.max_header_depth = max_header_depth
        self.section_map = {}  # Maps section names to section numbers

        # Initialize OpenAI client with proper authentication
        if not openai_api_key:
            # Try to load from environment or .env file
            load_dotenv()
            openai_api_key = os.getenv("OPENAI_API_KEY")

        if not openai_api_key:
            raise ValueError(
                "OpenAI API key is required. Either pass it directly or set OPENAI_API_KEY environment variable."
            )

        self.openai_client = OpenAI(api_key=openai_api_key)

    def load_data(self, file_path: str) -> List[Dict[str, str]]:
        """Load appetite guides from JSON file"""
        with open(file_path, "r") as f:
            data = json.load(f)
        return data

    def parse_markdown_hierarchy(self, markdown_text: str) -> List[Dict[str, Any]]:
        """
        Parse markdown text into a hierarchical structure for chunking
        Returns list of sections with their headers and content
        Handles any header depth up to max_header_depth
        """
        # Define the header patterns to split on - dynamically based on max_header_depth
        headers_to_split_on = []
        for i in range(1, self.max_header_depth + 1):
            header_marker = "#" * i
            header_name = f"Header{i}"
            headers_to_split_on.append((header_marker, header_name))

        # Initialize the markdown splitter
        splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

        # Split the document
        sections = splitter.split_text(markdown_text)
        return sections
    
    def generate_section_number_map(self, sections: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Generate a flat section name → section number mapping.
        Handles headers from Header1 to HeaderN properly including nested headers.
        """
        section_map = {}
        section_counters = {i: 0 for i in range(1, self.max_header_depth + 1)}
        current_parents = {i: None for i in range(1, self.max_header_depth + 1)}
        section_counters[1] = 1
        
        for section in sections:
            section_level = None
            section_name = None
    
            # 🔁 Get deepest header
            for j in reversed(range(1, self.max_header_depth + 1)):
                header_key = f"Header{j}"
                if header_key in section.metadata:
                    section_level = j
                    section_name = section.metadata[header_key]
                    break
    
            if section_level and section_name:
                # Reset lower-level counters and parent numbers
                for j in range(section_level + 1, self.max_header_depth + 1):
                    section_counters[j] = 0
                    current_parents[j] = None
    
                # Increment this header level
                section_counters[section_level] += 1
    
                # Build section number
                section_number = ".".join(str(section_counters[i]) for i in range(1, section_level + 1))
    
                # Set parent
                current_parents[section_level] = section_number
    
                # ✅ Add to map
                section_map[section_name] = section_number
    
                # 🔁 Also map parent titles (e.g. Header1) if not already mapped
                for j in range(1, section_level):
                    parent_title = section.metadata.get(f"Header{j}")
                    if parent_title and parent_title not in section_map:
                        parent_number = ".".join(str(section_counters[i]) for i in range(1, j + 1))
                        section_map[parent_title] = parent_number
    
        self.section_map = section_map  # store it
        return section_map
            
    def extract_bullet_points(self, text: str) -> List[str]:
        """Extract bullet points from text section"""
        # Convert Markdown to HTML for easier parsing
        html = markdown.markdown(text)
        soup = BeautifulSoup(html, "html.parser")

        # Extract lists (bullet points)
        bullet_points = []
        list_items = soup.find_all("li")

        for li in list_items:
            bullet_points.append(f"* {li.get_text()}")

        # If no bullets were found with BeautifulSoup, try regex for simple bullets
        if not bullet_points:
            bullet_regex = re.compile(r"^[\s-]*[-•*]\s+(.*?)$", re.MULTILINE)
            matches = bullet_regex.findall(text)
            bullet_points = [f"* {match}" for match in matches]

        return bullet_points

    def extract_metadata_with_openai(
        self, text: str, carrier_name: str, content_type: str = "guide"
    ) -> Dict[str, Any]:
        """
        Extract structured metadata from text using OpenAI's models

        Args:
            text: Text to extract metadata from (full guide or section)
            carrier_name: Name of the insurance carrier
            content_type: Type of content being analyzed (guide, section, or bullet)

        Returns:
            Structured metadata in the required format
        """

        # Create the prompt for OpenAI
        system_prompt = """You are an insurance domain expert who extracts structured metadata from carrier appetite guides.
Extract information ONLY from the provided text. Don't try to infer additional information.
If a requested field is not explicitly mentioned in the text, set it to null.
Format in the exact JSON schema requested."""

        user_prompt = f"""Extract metadata ONLY from this insurance carrier appetite guide.
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
- limit is the maximum amount the insurer will pay for a specific claim.:
- Use "None" when information is not present in this specific text
- Do NOT include information that isn't explicitly stated in this text
- Do NOT try to fill in fields by inferring from related information
- Set a field to None rather than guessing if you're unsure

Return only the JSON with no additional explanation.
"""

        # Call OpenAI API
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )

            # Extract the JSON from OpenAI's response
            openai_response = response.choices[0].message.content

            # Parse the JSON
            try:
                metadata = json.loads(openai_response)

                # Ensure carrier_name is exactly as provided
                metadata["carrier_name"] = carrier_name

                # Process null values to ensure consistency
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

                # Ensure nested null values are handled properly
                for nested_key in ["geographical_region", "Natural_disaster"]:
                    if nested_key in metadata and metadata[nested_key] is not None:
                        for sub_key in ["include", "exclude"]:
                            if (
                                sub_key not in metadata[nested_key]
                                or metadata[nested_key][sub_key] is None
                            ):
                                metadata[nested_key][sub_key] = None

                return metadata
            except json.JSONDecodeError as e:
                # Create a basic metadata structure with the carrier name if we can't parse
                basic_metadata = {
                    "carrier_name": carrier_name,
                    "geographical_region": {"include": None, "exclude": None},
                    "coverage": None,
                    "capacity": None,
                    "limit": None,
                    "Natural_disaster": {"include": None, "exclude": None},
                }
                print(
                    f"Error parsing JSON from OpenAI: {e}. Using basic metadata structure."
                )
                return basic_metadata

        except Exception as e:
            # Create a basic metadata structure with the carrier name if API call fails
            basic_metadata = {
                "carrier_name": carrier_name,
                "geographical_region": {"include": None, "exclude": None},
                "coverage": None,
                "capacity": None,
                "limit": None,
                "Natural_disaster": {"include": None, "exclude": None},
            }
            print(
                f"Error extracting metadata with OpenAI: {e}. Using basic metadata structure."
            )
            return basic_metadata

    def add_parent_info( self, metadata: Dict[str, Any], section_name: str, section_hierarchy: Dict[str, List[str]],) -> Dict[str, Any]:
        """
        Add parent section information to the metadata

        Args:
            metadata: The metadata to update
            section_name: The name of the current section
            section_hierarchy: Dictionary mapping section titles to their parents

        Returns:
            Updated metadata with parents field
        """
        # Add parents field
        sn = section_hierarchy.get(section_name, [])
        metadata["section_number"]= sn 

        return metadata

    def create_hierarchical_chunks( self, sections: List[Dict[str, Any]], carrier_name: str, full_guide: str) -> List[Dict[str, str]]:
        """
        Create hierarchical chunks based on the section structure and bullet points
        Each chunk will have its own metadata extracted using OpenAI

        Args:
            sections: List of parsed markdown sections
            carrier_name: Name of the insurance carrier
            full_guide: Complete text of the guide
        """
        chunks = []

        # Build section hierarchy
        section_hierarchy = self.generate_section_number_map(sections)

        # Extract metadata for the full guide
        full_guide_metadata = self.extract_metadata_with_openai( full_guide, carrier_name, content_type="guide")
        full_guide_metadata["section_number"] = '1'

        # Add the full guide as a chunk with its metadata
        chunks.append({"text": full_guide, "metadata": full_guide_metadata})

        # Process each section
        for section in sections:
            # Create the full section text with all headers
            section_text = ""
            section_name = None

            # Include all possible header levels dynamically
            for i in range(1, self.max_header_depth + 1):
                header_key = f"Header{i}"
                if header_key in section.metadata:
                    section_name = section.metadata[header_key]
                    header_level = "#" * i
                    section_text += f"{header_level} {section_name}\n\n"

            section_text += section.page_content

            # Extract metadata for this section only
            section_metadata = self.extract_metadata_with_openai(section_text, carrier_name, content_type="section")

            # Add parent information if section name is available
            if section_name:
                section_metadata = self.add_parent_info(section_metadata, section_name, section_hierarchy)

            # Add the section as a chunk with its specific metadata
            chunks.append({"text": section_text, "metadata": section_metadata})

            __import__('pdb').set_trace()
            # Extract bullet points for further chunking
            bullet_points = self.extract_bullet_points(section.page_content)

            # Process each bullet point
            for bullet in bullet_points:
                bullet_text = ""

                # Add all parent headers
                for i in range(1, self.max_header_depth + 1):
                    header_key = f"Header{i}"
                    if header_key in section.metadata:
                        header_level = "#" * i
                        bullet_text += (
                            f"{header_level} {section.metadata[header_key]}\n\n"
                        )

                # Add the bullet point
                bullet_text += bullet

                # Extract metadata for this bullet point only
                bullet_metadata = self.extract_metadata_with_openai(bullet_text, carrier_name, content_type="bullet")

                # Add parent information if section name is available
                if section_name:
                    bullet_metadata = self.add_parent_info( bullet_metadata, section_name, section_hierarchy)

                # Add the bullet point chunk with its specific metadata
                chunks.append({"text": bullet_text, "metadata": bullet_metadata})

        return chunks

    def process_guide(self, guide_data: Dict[str, str]) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
        """Process a single carrier's appetite guide"""
        carrier_name = guide_data["carrier_name"]
        guide_text = guide_data["guide"]

        self.logger.info(f"Processing guide for carrier: {carrier_name}")

        # Parse the markdown into hierarchical sections
        sections = self.parse_markdown_hierarchy(guide_text)

        # Create hierarchical chunks with block-specific metadata
        chunks = self.create_hierarchical_chunks(sections, carrier_name, guide_text)

        # Extract the full guide metadata from the first chunk
        full_guide_metadata = (
            chunks[0]["metadata"]
            if chunks
            else {
                "carrier_name": carrier_name,
                "parents": [],
                "geographical_region": {"include": None, "exclude": None},
                "coverage": None,
                "capacity": None,
                "limit": None,
                "Natural_disaster": {"include": None, "exclude": None},
            }
        )

        return chunks, full_guide_metadata

    def process_all_guides( self, guides_data: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], Dict[str, Dict]]:
        """Process all carrier appetite guides"""
        all_chunks = []
        all_metadata = {}

        for guide_data in guides_data:
            carrier_name = guide_data["carrier_name"]

            chunks, metadata = self.process_guide(guide_data)
            all_chunks.extend(chunks)
            all_metadata[carrier_name] = metadata

        return all_chunks, all_metadata

    def create_vector_db(self, chunks: List[Dict[str, str]]) -> None:
        """Create vector database from chunks"""
        texts = [chunk["text"] for chunk in chunks]

        # Serialize metadata
        metadatas = []
        for chunk in chunks:
            chunk_metadata = chunk["metadata"].copy()
            metadatas.append(json.dumps(chunk_metadata))

        # Create Chroma vector database with serialized metadata
        simple_metadatas = [{"metadata_json": m} for m in metadatas]

        self.vector_db = Chroma.from_texts(
            texts=texts,
            embedding=self.embeddings,
            metadatas=simple_metadatas,
            persist_directory=self.db_path,
        )

        # Persist the database
        self.vector_db.persist()

    def save_structured_metadata(
        self, metadata: Dict[str, Dict], output_path: str = "structured_metadata.json"
    ) -> None:
        """Save the structured metadata to a JSON file"""
        with open(output_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def save_section_map(self, output_path: str = "section_map.json") -> None:
        """Save the section map to a JSON file"""
        with open(output_path, "w") as f:
            json.dump(self.section_map, f, indent=2)

    def build(self, file_path: str) -> Dict[str, Dict]:
        """
        Build the vector database from the given file

        Returns:
            Dict mapping carrier names to their structured metadata
        """
        # Load data
        guides_data = self.load_data(file_path)

        # Process all guides
        chunks, structured_metadata = self.process_all_guides(guides_data)

        # Create vector database with serialized metadata
        self.create_vector_db(chunks)

        # Save structured metadata
        self.save_structured_metadata(structured_metadata)

        # Save section map
        self.save_section_map()

        return structured_metadata

    def search(
        self, query: str, n_results: int = 5, filter_carrier: str = None
    ) -> List[Dict[str, Any]]:
        """
        Search the vector database for relevant chunks

        Args:
            query: Search query
            n_results: Number of results to return
            filter_carrier: Optional carrier name to filter by

        Returns:
            List of search results with text and metadata
        """
        if not self.vector_db:
            if os.path.exists(self.db_path):
                # Load the database if it exists
                self.vector_db = Chroma(
                    persist_directory=self.db_path, embedding_function=self.embeddings
                )
            else:
                raise ValueError(
                    "Vector database has not been built yet. Call build() first."
                )

        # Get search results
        results = self.vector_db.similarity_search_with_relevance_scores(
            query, k=n_results * 3  # Get more results than needed to handle filtering
        )

        # Format and filter results
        formatted_results = []
        for doc, score in results:
            # Deserialize the metadata from JSON
            metadata_json = doc.metadata.get("metadata_json", "{}")
            try:
                metadata = json.loads(metadata_json)

            except json.JSONDecodeError:
                print("Warning: Could not parse metadata JSON in search result")
                continue

            # Apply carrier filter if needed
            if filter_carrier and metadata.get("carrier_name") != filter_carrier:
                continue

            result = {
                "text": doc.page_content,
                "metadata": metadata,
                "relevance_score": score,
            }

            formatted_results.append(result)

            # Stop once we have enough results
            if len(formatted_results) >= n_results:
                break

        return formatted_results

    def query_by_metadata(
        self,
        region: str = None,
        coverage_type: str = None,
        natural_disaster: str = None,
        include_exclude: str = "include",
    ) -> List[Dict[str, Any]]:
        """
        Query carriers based on structured metadata attributes

        Args:
            region: Geographical region to search for
            coverage_type: Type of coverage to search for
            natural_disaster: Natural disaster to search for
            include_exclude: Whether to search in 'include' or 'exclude' lists

        Returns:
            List of carriers matching the criteria
        """
        # Load metadata file
        try:
            with open("structured_metadata.json", "r") as f:
                all_metadata = json.load(f)
        except FileNotFoundError:
            raise ValueError(
                "Structured metadata file not found. Build the database first."
            )

        results = []

        for carrier_name, metadata in all_metadata.items():
            match = True

            # Check region
            if region and match:
                geo_data = metadata.get("geographical_region", {})
                if geo_data:
                    regions = geo_data.get(
                        "include" if include_exclude == "include" else "exclude", []
                    )
                    if regions is not None:
                        match = any(region.lower() in r.lower() for r in regions)
                    else:
                        match = False
                else:
                    match = False

            # Check coverage
            if coverage_type and match:
                coverages = metadata.get("coverage", [])
                if coverages is not None:
                    match = any(
                        coverage_type.lower() in cov.lower()
                        for cov in coverages
                        if isinstance(cov, str)
                    )
                else:
                    match = False

            # Check natural disaster
            if natural_disaster and match:
                disaster_data = metadata.get("Natural_disaster", {})
                if disaster_data:
                    disasters = disaster_data.get(
                        "include" if include_exclude == "include" else "exclude", []
                    )
                    if disasters is not None:
                        match = any(
                            natural_disaster.lower() in disaster.lower()
                            for disaster in disasters
                        )
                    else:
                        match = False
                else:
                    match = False

            if match:
                results.append({"carrier_name": carrier_name, "metadata": metadata})

        return results


# Example usage
if __name__ == "__main__":
    logger.info('start')
    # Initialize the VectorDB with local embeddings and OpenAI integration
    # It will automatically load API key from OPENAI_API_KEY environment variable
    app_db = AppetiteGuideVectorDB(logger_file=logger,model_name="all-MiniLM-L6-v2", max_header_depth=6)

    # Build the database
    structured_metadata = app_db.build("guide.json")
