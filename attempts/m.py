import json
import re
import os
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from openai import OpenAI
import markdown
from bs4 import BeautifulSoup
from dotenv import load_dotenv

class AppetiteGuideVectorDB:
    def __init__(self, model_name="all-MiniLM-L6-v2", max_header_depth=6, openai_api_key=None):
        """
        Initialize the vector database for carrier appetite guides with local embedding model
        
        Args:
            model_name: Name of the HuggingFace embedding model to use
                        Default is all-MiniLM-L6-v2 which is small (80MB) but effective
            max_header_depth: Maximum number of header levels to process (default=6)
            openai_api_key: API key for OpenAI (if None, will try to load from environment)
        """
        # Initialize local embedding model
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.vector_db = None
        self.db_path = "./appetite_guide_db"
        self.model_name = model_name
        self.max_header_depth = max_header_depth
        
        # Initialize OpenAI client with proper authentication
        if not openai_api_key:
            # Try to load from environment or .env file
            load_dotenv()
            openai_api_key = os.getenv("OPENAI_API_KEY")
            
        if not openai_api_key:
            raise ValueError("OpenAI API key is required. Either pass it directly or set OPENAI_API_KEY environment variable.")
        
        self.openai_client = OpenAI(api_key=openai_api_key)
        
    def load_data(self, file_path: str) -> List[Dict[str, str]]:
        """Load appetite guides from JSON file"""
        with open(file_path, 'r') as f:
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
    
    def extract_bullet_points(self, text: str) -> List[str]:
        """Extract bullet points from text section"""
        # Convert Markdown to HTML for easier parsing
        html = markdown.markdown(text)
        soup = BeautifulSoup(html, 'html.parser')
        
        # Extract lists (bullet points)
        bullet_points = []
        list_items = soup.find_all('li')
        
        for li in list_items:
            bullet_points.append(f"* {li.get_text()}")
            
        # If no bullets were found with BeautifulSoup, try regex for simple bullets
        if not bullet_points:
            bullet_regex = re.compile(r'^[\s-]*[-•*]\s+(.*?)$', re.MULTILINE)
            matches = bullet_regex.findall(text)
            bullet_points = [f"* {match}" for match in matches]
        
        return bullet_points
    
    def extract_metadata_with_openai(self, guide_text: str, carrier_name: str) -> Dict[str, Any]:
        """
        Extract structured metadata from the appetite guide using OpenAI's models
        
        Args:
            guide_text: Full text of the appetite guide
            carrier_name: Name of the insurance carrier
            
        Returns:
            Structured metadata in the required format
        """
        # Create the prompt for OpenAI
        system_prompt = """You are an insurance domain expert who extracts structured metadata from carrier appetite guides. Extract information accurately and format it in the exact JSON schema requested."""
        
        user_prompt = f"""Please extract structured metadata from this insurance carrier appetite guide.
        
Carrier Name: {carrier_name}

Guide Text:
{guide_text}

Extract this information into a structured JSON with the exact following schema:
{{
"carrier_name": "{carrier_name}",
"section": dictionary of section numbers to section names,
"geographical_region": {{"include": [list of included regions], "exclude": [list of excluded regions]}},
"coverage": [list of coverage types],
"capacity": [list of capacity limits as dictionaries],
"limit": [list of policy limits as dictionaries],
"Natural_disaster": {{"include": [list of included natural disasters], "exclude": [list of excluded natural disasters]}}
}}

Note: 
- Make sure to keep the carrier_name exactly as provided: "{carrier_name}"
- capacity is the maximum risk an insurer can underwrite
- coverage refers to the specific risks or damages the policy protects against
- limit is the maximum amount the insurer will pay for a specific claim
- section should map numbers (like "1", "2.1", "3.2.1") to section names

Ensure the JSON is properly formatted and contains all required fields.
Return only the JSON with no additional explanation.
"""
        
        # Call OpenAI API
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4-turbo",  # Using GPT-4 for best extraction capability
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0,
                response_format={"type": "json_object"}  # Ensure we get valid JSON
            )
            
            # Extract the JSON from OpenAI's response
            openai_response = response.choices[0].message.content
            
            # Parse the JSON
            try:
                metadata = json.loads(openai_response)
                
                # Ensure carrier_name is exactly as provided
                metadata["carrier_name"] = carrier_name
                
                return metadata
            except json.JSONDecodeError as e:
                # Create a basic metadata structure with the carrier name if we can't parse
                basic_metadata = {
                    "carrier_name": carrier_name,
                    "section": {},
                    "geographical_region": {"include": [], "exclude": []},
                    "coverage": [],
                    "capacity": [],
                    "limit": [],
                    "Natural_disaster": {"include": [], "exclude": []}
                }
                print(f"Error parsing JSON from OpenAI: {e}. Using basic metadata structure.")
                return basic_metadata
                
        except Exception as e:
            # Create a basic metadata structure with the carrier name if API call fails
            basic_metadata = {
                "carrier_name": carrier_name,
                "section": {},
                "geographical_region": {"include": [], "exclude": []},
                "coverage": [],
                "capacity": [],
                "limit": [],
                "Natural_disaster": {"include": [], "exclude": []}
            }
            print(f"Error extracting metadata with OpenAI: {e}. Using basic metadata structure.")
            return basic_metadata
    
    def create_hierarchical_chunks(self, sections: List[Dict[str, Any]], 
                                  carrier_name: str, full_guide: str,
                                  structured_metadata: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Create hierarchical chunks based on the section structure and bullet points
        Each chunk will have the same standardized metadata
        
        Args:
            sections: List of parsed markdown sections
            carrier_name: Name of the insurance carrier
            full_guide: Complete text of the guide
            structured_metadata: Structured metadata extracted from the guide
        """
        # Ensure carrier_name is in structured_metadata
        if "carrier_name" not in structured_metadata:
            structured_metadata["carrier_name"] = carrier_name
            
        chunks = []
        
        # Add the full guide as a chunk with standardized metadata
        chunks.append({
            "text": full_guide,
            "metadata": structured_metadata
        })
        
        # Add each complete section as a chunk
        for section in sections:
            # Create the full section text with all headers
            section_text = ""
            # Include all possible header levels dynamically
            for i in range(1, self.max_header_depth + 1):
                header_key = f"Header{i}"
                if header_key in section.metadata:
                    header_level = "#" * i
                    section_text += f"{header_level} {section.metadata[header_key]}\n\n"
            
            section_text += section.page_content
            
            # Add the full section as a chunk with the same standardized metadata
            chunks.append({
                "text": section_text,
                "metadata": structured_metadata
            })
            
            # Extract bullet points for further chunking
            bullet_points = self.extract_bullet_points(section.page_content)
            
            # Create individual chunks for each bullet point with its parent headers
            for bullet in bullet_points:
                bullet_text = ""
                
                # Add all parent headers
                for i in range(1, self.max_header_depth + 1):
                    header_key = f"Header{i}"
                    if header_key in section.metadata:
                        header_level = "#" * i
                        bullet_text += f"{header_level} {section.metadata[header_key]}\n\n"
                
                # Add the bullet point
                bullet_text += bullet
                
                # Add the bullet point chunk with the same standardized metadata
                chunks.append({
                    "text": bullet_text,
                    "metadata": structured_metadata
                })
        
        return chunks
    
    def process_guide(self, guide_data: Dict[str, str]) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
        """Process a single carrier's appetite guide"""
        carrier_name = guide_data["carrier_name"]
        guide_text = guide_data["guide"]
        
        # Parse the markdown into hierarchical sections
        sections = self.parse_markdown_hierarchy(guide_text)
        
        # Extract structured metadata using OpenAI
        structured_metadata = self.extract_metadata_with_openai(guide_text, carrier_name)
        
        # Double-check carrier name is correct
        structured_metadata["carrier_name"] = carrier_name
        
        # Create hierarchical chunks including the full guide with standardized metadata
        chunks = self.create_hierarchical_chunks(sections, carrier_name, guide_text, structured_metadata)
        
        return chunks, structured_metadata
    
    def process_all_guides(self, guides_data: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], Dict[str, Dict]]:
        """Process all carrier appetite guides"""
        all_chunks = []
        all_metadata = {}
        
        for guide_data in guides_data:
            carrier_name = guide_data["carrier_name"]
            print(f"Processing guide for carrier: {carrier_name}")
            
            chunks, metadata = self.process_guide(guide_data)
            all_chunks.extend(chunks)
            all_metadata[carrier_name] = metadata
            
        return all_chunks, all_metadata
    
    def create_vector_db(self, chunks: List[Dict[str, str]]) -> None:
        """Create vector database from chunks"""
        texts = [chunk["text"] for chunk in chunks]
        
        # Ensure each metadata dictionary has carrier_name before serializing
        metadatas = []
        for chunk in chunks:
            chunk_metadata = chunk["metadata"].copy()
            if "carrier_name" not in chunk_metadata:
                # This should never happen with our fixes, but just in case
                print("Warning: Missing carrier_name in chunk metadata")
            metadatas.append(json.dumps(chunk_metadata))
        
        # Create Chroma vector database with serialized metadata
        # We serialize the metadata to JSON strings to avoid Chroma's metadata type restrictions
        simple_metadatas = [{"metadata_json": m} for m in metadatas]
        
        self.vector_db = Chroma.from_texts(
            texts=texts,
            embedding=self.embeddings,
            metadatas=simple_metadatas,
            persist_directory=self.db_path
        )
        
        # Persist the database
        self.vector_db.persist()
    
    def save_structured_metadata(self, metadata: Dict[str, Dict], output_path: str = "structured_metadata.json") -> None:
        """Save the structured metadata to a JSON file"""
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
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
        __import__('pdb').set_trace()
        self.create_vector_db(chunks)
        
        # Save structured metadata
        self.save_structured_metadata(structured_metadata)
        
        # Count chunks by carrier
        carrier_counts = {}
        for chunk in chunks:
            carrier_name = chunk["metadata"]["carrier_name"]
            carrier_counts[carrier_name] = carrier_counts.get(carrier_name, 0) + 1
        
        print(f"Vector database created with {len(chunks)} total chunks using {self.model_name} embeddings")
        print("Chunk distribution by carrier:")
        for carrier, count in carrier_counts.items():
            print(f"  - {carrier}: {count} chunks")
        
        return structured_metadata
    
    def search(self, query: str, n_results: int = 5, 
               filter_carrier: str = None) -> List[Dict[str, Any]]:
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
                    persist_directory=self.db_path,
                    embedding_function=self.embeddings
                )
            else:
                raise ValueError("Vector database has not been built yet. Call build() first.")
        
        # Get search results
        results = self.vector_db.similarity_search_with_relevance_scores(
            query, k=n_results*3  # Get more results than needed to handle filtering
        )
        
        # Format and filter results
        formatted_results = []
        for doc, score in results:
            # Deserialize the metadata from JSON
            metadata_json = doc.metadata.get("metadata_json", "{}")
            try:
                metadata = json.loads(metadata_json)
                
                # Make sure carrier_name exists
                if "carrier_name" not in metadata:
                    print("Warning: Missing carrier_name in search result metadata")
                    continue
                    
            except json.JSONDecodeError:
                print("Warning: Could not parse metadata JSON in search result")
                continue
            
            # Apply carrier filter if needed
            if filter_carrier and metadata.get("carrier_name") != filter_carrier:
                continue
                
            result = {
                "text": doc.page_content,
                "metadata": metadata,
                "relevance_score": score
            }
            
            formatted_results.append(result)
            
            # Stop once we have enough results
            if len(formatted_results) >= n_results:
                break
        
        return formatted_results
    
    def query_by_metadata(self, 
                          region: str = None,
                          coverage_type: str = None, 
                          natural_disaster: str = None,
                          include_exclude: str = "include") -> List[Dict[str, Any]]:
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
            with open("structured_metadata.json", 'r') as f:
                all_metadata = json.load(f)
        except FileNotFoundError:
            raise ValueError("Structured metadata file not found. Build the database first.")
        
        results = []
        
        for carrier_name, metadata in all_metadata.items():
            # Ensure carrier_name exists in the metadata
            if "carrier_name" not in metadata:
                metadata["carrier_name"] = carrier_name
                
            match = True
            
            # Check region
            if region and match:
                geo_data = metadata.get("geographical_region", {})
                if include_exclude == "include":
                    match = region in geo_data.get("include", [])
                else:
                    match = region in geo_data.get("exclude", [])
            
            # Check coverage
            if coverage_type and match:
                coverages = metadata.get("coverage", [])
                flat_coverages = []
                for cov in coverages:
                    if isinstance(cov, dict):
                        flat_coverages.extend(cov.keys())
                    elif isinstance(cov, str):
                        flat_coverages.append(cov)
                
                match = any(coverage_type.lower() in cov.lower() for cov in flat_coverages)
            
            # Check natural disaster
            if natural_disaster and match:
                disaster_data = metadata.get("Natural_disaster", {})
                if include_exclude == "include":
                    match = any(natural_disaster.lower() in disaster.lower() for disaster in disaster_data.get("include", []))
                else:
                    match = any(natural_disaster.lower() in disaster.lower() for disaster in disaster_data.get("exclude", []))
            
            if match:
                results.append({
                    "carrier_name": carrier_name,
                    "metadata": metadata
                })
        
        return results

# Example usage
if __name__ == "__main__":
    # Initialize the VectorDB with local embeddings and OpenAI integration
    # It will automatically load API key from OPENAI_API_KEY environment variable
    app_db = AppetiteGuideVectorDB(
        model_name="all-MiniLM-L6-v2",
        max_header_depth=6
    )
    
    # Build the database
    structured_metadata = app_db.build("guide.json")
    
    # Print an example of the structured metadata
    print("\n=== Example Structured Metadata ===")
    for carrier_name, metadata in list(structured_metadata.items())[:1]:
        print(f"Carrier: {carrier_name}")
        print(json.dumps(metadata, indent=2))
    
    # Example search
    print("\n=== Search Results ===")
    results = app_db.search("What coverage is offered for wildfire risks?")
    
    for i, result in enumerate(results):
        print(f"\nResult {i+1} (Score: {result['relevance_score']:.2f}):")
        print(f"Carrier: {result['metadata']['carrier_name']}")
        
        # Print relevant disaster information
        if result['metadata']['Natural_disaster']['include']:
            print(f"Included Natural Disasters: {', '.join(result['metadata']['Natural_disaster']['include'])}")
        if result['metadata']['Natural_disaster']['exclude']:
            print(f"Excluded Natural Disasters: {', '.join(result['metadata']['Natural_disaster']['exclude'])}")
        
        # Print truncated content
        print("\nContent:")
        print(result['text'][:200] + "..." if len(result['text']) > 200 else result['text'])
