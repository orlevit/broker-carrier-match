import json
import re
import os
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
import markdown
from bs4 import BeautifulSoup

class AppetiteGuideVectorDB:
    def __init__(self, model_name="all-MiniLM-L6-v2", max_header_depth=6):
        """
        Initialize the vector database for carrier appetite guides with local embedding model
        
        Args:
            model_name: Name of the HuggingFace embedding model to use
                        Default is all-MiniLM-L6-v2 which is small (80MB) but effective
            max_header_depth: Maximum number of header levels to process (default=6)
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
    
    def extract_structured_metadata(self, sections: List[Dict[str, Any]], carrier_name: str) -> Dict[str, Any]:
        """
        Extract structured metadata from the markdown sections according to the specified schema
        
        Returns:
        Dict with the following structure:
        {
            'carrier_name': str,
            'section': Dict[str, str],  # Section numbers to section names
            'geographical_region': {'include': List[str], 'exclude': List[str]},
            'coverage': List[Dict],
            'capacity': List[Dict],
            'limit': List[Dict],
            'Natural_disaster': {'include': List[str], 'exclude': List[str]}
        }
        """
        metadata = {
            'carrier_name': carrier_name,
            'section': {},
            'geographical_region': {'include': [], 'exclude': []},
            'coverage': [],
            'capacity': [],
            'limit': [],
            'Natural_disaster': {'include': [], 'exclude': []}
        }
        
        # Initialize section counters for all possible levels
        section_counters = {i: 0 for i in range(1, self.max_header_depth + 1)}
        section_map = {}
        
        # Extract section numbers and names
        for section in sections:
            section_level = None
            section_name = None
            section_number = None
            
            # Determine section level and name
            for i in range(1, self.max_header_depth + 1):
                header_key = f"Header{i}"
                if header_key in section.metadata:
                    section_level = i
                    section_name = section.metadata[header_key]
                    break
            
            if section_level is not None and section_name is not None:
                # Reset counters for lower levels
                for j in range(section_level + 1, self.max_header_depth + 1):
                    section_counters[j] = 0
                
                # Increment counter for current level
                section_counters[section_level] += 1
                
                # Build section number based on hierarchy
                section_number_parts = []
                for j in range(1, section_level + 1):
                    section_number_parts.append(str(section_counters[j]))
                
                section_number = ".".join(section_number_parts)
                
                # Add to section map
                section_map[section_name.lower()] = section_number
                metadata['section'][section_number] = section_name
                
                # Extract specific metadata based on section name
                section_text = section.page_content.lower()
                bullet_points = self.extract_bullet_points(section.page_content)
                
                # Geographic regions
                if any(term in section_name.lower() for term in ['geographic', 'region', 'location', 'territory']):
                    # Look for inclusion terms
                    for bullet in bullet_points:
                        bullet_lower = bullet.lower()
                        if any(term in bullet_lower for term in ['available', 'offer', 'include', 'consider']):
                            # Extract locations, skip general terms like "all states"
                            locations = re.findall(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', bullet)
                            if locations:
                                metadata['geographical_region']['include'].extend(locations)
                        elif any(term in bullet_lower for term in ['not available', 'exclude', 'not consider', 'will not']):
                            locations = re.findall(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', bullet)
                            if locations:
                                metadata['geographical_region']['exclude'].extend(locations)
                
                # Special handling for 'Special Risk Areas' section
                if 'special risk' in section_name.lower():
                    for bullet in bullet_points:
                        bullet_lower = bullet.lower()
                        if 'wildfire' in bullet_lower:
                            metadata['Natural_disaster']['include'].append('Wildfire exposed risks')
                            # Extract locations
                            locations = re.findall(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', bullet)
                            if locations:
                                metadata['geographical_region']['include'].extend(locations)
                        elif 'flood' in bullet_lower:
                            if 'excluding' in bullet_lower:
                                metadata['Natural_disaster']['include'].append(bullet.strip('* '))
                            else:
                                metadata['Natural_disaster']['include'].append('Flood')
                            
                            # Extract locations to consider/not consider
                            consider_match = re.search(r'consider:\s*(.*?)(?:$|\n)', bullet_lower)
                            if consider_match:
                                locations = re.findall(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', consider_match.group(1))
                                if locations:
                                    metadata['geographical_region']['include'].extend(locations)
                            
                            not_consider_match = re.search(r'will not consider:\s*(.*?)(?:$|\n)', bullet_lower)
                            if not_consider_match:
                                locations = re.findall(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', not_consider_match.group(1))
                                if locations:
                                    metadata['geographical_region']['exclude'].extend(locations)
                
                # Coverage
                if any(term in section_name.lower() for term in ['coverage', 'covered', 'protect']):
                    coverage_items = []
                    for bullet in bullet_points:
                        # Remove exclusion language to get just the coverage type
                        coverage_item = re.sub(r'\(excluding.*?\)', '', bullet)
                        coverage_item = coverage_item.strip('* ')
                        if coverage_item:
                            coverage_items.append(coverage_item)
                    
                    if coverage_items:
                        metadata['coverage'].extend(coverage_items)
                
                # Capacity
                if any(term in section_name.lower() for term in ['capacity', 'value', 'tiv', 'overview']):
                    # Look for TIV or capacity mentions
                    tiv_match = re.search(r'(total insured value|tiv).*?(\$[\d,.]+ ?m?illion)', section_text, re.IGNORECASE)
                    if tiv_match:
                        metadata['capacity'].append({
                            'Total Insured Value': tiv_match.group(2)
                        })
                
                # Limits
                if any(term in section_name.lower() for term in ['limit', 'policy limit']):
                    limit_dict = {}
                    for bullet in bullet_points:
                        # Look for pattern: item: amount
                        limit_match = re.search(r'(.*?):\s*(\$[\d,.]+ ?[mk]?)', bullet, re.IGNORECASE)
                        if limit_match:
                            item = limit_match.group(1).strip('* ')
                            amount = limit_match.group(2)
                            limit_dict[item] = amount
                    
                    if limit_dict:
                        metadata['limit'].append(limit_dict)
                
                # Look for frame habitational limits in underwriting considerations
                if 'underwriting' in section_name.lower():
                    for bullet in bullet_points:
                        frame_match = re.search(r'(?:rate|premium|limit).*?(frame\s+habitational).*?(\$[\d,.]+ ?[mk]?)', bullet, re.IGNORECASE)
                        if frame_match:
                            metadata['limit'].append({
                                f"{frame_match.group(1)} primary": frame_match.group(2)
                            })
                
                # Natural disasters
                if 'excluded' in section_name.lower() or 'restriction' in section_name.lower():
                    for bullet in bullet_points:
                        bullet_lower = bullet.lower()
                        if any(disaster in bullet_lower for disaster in ['wind', 'flood', 'fire', 'earthquake', 'wildfire']):
                            metadata['Natural_disaster']['exclude'].append(bullet.strip('* '))
        
        # Remove duplicates
        metadata['geographical_region']['include'] = list(set(metadata['geographical_region']['include']))
        metadata['geographical_region']['exclude'] = list(set(metadata['geographical_region']['exclude']))
        metadata['Natural_disaster']['include'] = list(set(metadata['Natural_disaster']['include']))
        metadata['Natural_disaster']['exclude'] = list(set(metadata['Natural_disaster']['exclude']))
        
        return metadata
    
    def create_hierarchical_chunks(self, sections: List[Dict[str, Any]], 
                                  carrier_name: str, full_guide: str,
                                  structured_metadata: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Create hierarchical chunks based on the section structure and bullet points
        Each bullet point will be chunked with its parent headers
        Also includes the entire guide as a full document chunk
        
        Args:
            sections: List of parsed markdown sections
            carrier_name: Name of the insurance carrier
            full_guide: Complete text of the guide
            structured_metadata: Structured metadata extracted from the guide
        """
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
        
        # Extract structured metadata
        structured_metadata = self.extract_structured_metadata(sections, carrier_name)
        
        # Create hierarchical chunks including the full guide with standardized metadata
        chunks = self.create_hierarchical_chunks(sections, carrier_name, guide_text, structured_metadata)
        
        return chunks, structured_metadata
    
    def process_all_guides(self, guides_data: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], Dict[str, Dict]]:
        """Process all carrier appetite guides"""
        all_chunks = []
        all_metadata = {}
        
        for guide_data in guides_data:
            chunks, metadata = self.process_guide(guide_data)
            all_chunks.extend(chunks)
            carrier_name = guide_data["carrier_name"]
            all_metadata[carrier_name] = metadata
            
        return all_chunks, all_metadata
    
    def create_vector_db(self, chunks: List[Dict[str, str]]) -> None:
        """Create vector database from chunks"""
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [json.dumps(chunk["metadata"]) for chunk in chunks]
        
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
        __import__('pdb').set_trace() 
        # Create vector database with serialized metadata
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
        
        # Apply filter if provided
        filter_dict = None
        if filter_carrier:
            # Since we serialized the metadata, we need to filter on text content
            # This is a limitation of our approach, but the post-processing step below handles it
            # No reliable way to filter directly through Chroma in this case
            pass
        
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
            except json.JSONDecodeError:
                metadata = {}
            
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

# Demo usage
if __name__ == "__main__":
    # Initialize the VectorDB with local embeddings and support for up to 6 header levels
    app_db = AppetiteGuideVectorDB(model_name="all-MiniLM-L6-v2", max_header_depth=6)
    
    # Build the database
    structured_metadata = app_db.build("guide.json")
    
    # Print an example of the structured metadata
    print("\n=== Example Structured Metadata ===")
    for carrier_name, metadata in list(structured_metadata.items())[:1]:
        print(f"Carrier: {carrier_name}")
        print(json.dumps(metadata, indent=2))
    
    # Example search
    print("\n=== Search Results ===")
    results = app_db.search("What coverage is offered for commercial properties?")
    
    for i, result in enumerate(results):
        print(f"\nResult {i+1} (Score: {result['relevance_score']:.2f}):")
        print(f"Carrier: {result['metadata']['carrier_name']}")
        print(f"Geographical Regions:")
        print(f"  Include: {', '.join(result['metadata']['geographical_region']['include'])}")
        print(f"  Exclude: {', '.join(result['metadata']['geographical_region']['exclude'])}")
        print(f"Coverages: {result['metadata']['coverage']}")
        
        # Print truncated content
        print("\nContent:")
        print(result['text'][:200] + "..." if len(result['text']) > 200 else result['text'])
