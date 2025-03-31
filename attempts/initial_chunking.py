import json
import re
import os
from typing import List, Dict, Any, Tuple
import numpy as np
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
import markdown
from bs4 import BeautifulSoup

class AppetiteGuideVectorDB:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        Initialize the vector database for carrier appetite guides with local embedding model
        
        Args:
            model_name: Name of the HuggingFace embedding model to use
                        Default is all-MiniLM-L6-v2 which is small (80MB) but effective
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
        
    def load_data(self, file_path: str) -> List[Dict[str, str]]:
        """Load appetite guides from JSON file"""
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    
    def parse_markdown_hierarchy(self, markdown_text: str) -> List[Dict[str, Any]]:
        """
        Parse markdown text into a hierarchical structure for chunking
        Returns list of sections with their headers and content
        """
        # Define the header patterns to split on
        headers_to_split_on = [
            ("#", "Header1"),
            ("##", "Header2"),
            ("###", "Header3"),
        ]
        
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
    
    def create_hierarchical_chunks(self, sections: List[Dict[str, Any]], 
                                  carrier_name: str, full_guide: str) -> List[Dict[str, str]]:
        """
        Create hierarchical chunks based on the section structure and bullet points
        Each bullet point will be chunked with its parent headers
        Also includes the entire guide as a full document chunk
        """
        chunks = []
        
        # Add the full guide as a chunk
        chunks.append({
            "text": full_guide,
            "metadata": {
                "carrier_name": carrier_name,
                "type": "full_guide",
                "description": f"Complete appetite guide for {carrier_name}"
            }
        })
        
        # Add each complete section as a chunk
        for section in sections:
            # Create section metadata
            metadata = {
                "carrier_name": carrier_name,
                "type": "section"
            }
            
            # Add headers to metadata
            for header_key in section.metadata:
                if header_key.startswith("Header"):
                    metadata[header_key] = section.metadata[header_key]
            
            # Create the full section text with all headers
            section_text = ""
            for i in range(1, 4):  # Up to 3 levels of headers
                header_key = f"Header{i}"
                if header_key in section.metadata:
                    header_level = "#" * i
                    section_text += f"{header_level} {section.metadata[header_key]}\n\n"
            
            section_text += section.page_content
            
            # Add the full section as a chunk
            chunks.append({
                "text": section_text,
                "metadata": metadata
            })
            
            # Extract bullet points for further chunking
            bullet_points = self.extract_bullet_points(section.page_content)
            
            # Create individual chunks for each bullet point with its parent headers
            for bullet in bullet_points:
                bullet_text = ""
                
                # Add all parent headers
                for i in range(1, 4):
                    header_key = f"Header{i}"
                    if header_key in section.metadata:
                        header_level = "#" * i
                        bullet_text += f"{header_level} {section.metadata[header_key]}\n\n"
                
                # Add the bullet point
                bullet_text += bullet
                
                # Create bullet point metadata
                bullet_metadata = {
                    "carrier_name": carrier_name,
                    "type": "bullet_point"
                }
                
                # Add headers to bullet metadata
                for header_key in section.metadata:
                    if header_key.startswith("Header"):
                        bullet_metadata[header_key] = section.metadata[header_key]
                
                # Add the bullet point chunk
                chunks.append({
                    "text": bullet_text,
                    "metadata": bullet_metadata
                })
        
        return chunks
    
    def process_guide(self, guide_data: Dict[str, str]) -> List[Dict[str, str]]:
        """Process a single carrier's appetite guide"""
        carrier_name = guide_data["carrier_name"]
        guide_text = guide_data["guide"]
        
        # Parse the markdown into hierarchical sections
        sections = self.parse_markdown_hierarchy(guide_text)
        
        # Create hierarchical chunks including the full guide
        chunks = self.create_hierarchical_chunks(sections, carrier_name, guide_text)
        
        return chunks
    
    def process_all_guides(self, guides_data: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Process all carrier appetite guides"""
        all_chunks = []
        
        for guide_data in guides_data:
            chunks = self.process_guide(guide_data)
            all_chunks.extend(chunks)
            
        return all_chunks
    
    def create_vector_db(self, chunks: List[Dict[str, str]]) -> None:
        """Create vector database from chunks"""
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]
        
        # Create Chroma vector database
        self.vector_db = Chroma.from_texts(
            texts=texts,
            embedding=self.embeddings,
            metadatas=metadatas,
            persist_directory=self.db_path
        )
        
        # Persist the database
        self.vector_db.persist()
        
    def build(self, file_path: str) -> None:
        """Build the vector database from the given file"""
        # Load data
        guides_data = self.load_data(file_path)
        
        # Process all guides
        chunks = self.process_all_guides(guides_data)
        __import__('pdb').set_trace()
        # Create vector database
        self.create_vector_db(chunks)
        
        # Get counts by type
        type_counts = {}
        for chunk in chunks:
            chunk_type = chunk["metadata"]["type"]
            type_counts[chunk_type] = type_counts.get(chunk_type, 0) + 1
        
        print(f"Vector database created with {len(chunks)} total chunks using {self.model_name} embeddings")
        print("Chunk distribution:")
        for chunk_type, count in type_counts.items():
            print(f"  - {chunk_type}: {count} chunks")
        
    def search(self, query: str, n_results: int = 5, filter_metadata: Dict[str, str] = None) -> List[Dict[str, Any]]:
        """
        Search the vector database for relevant chunks
        
        Args:
            query: Search query
            n_results: Number of results to return
            filter_metadata: Optional filter to narrow search (e.g., {"carrier_name": "Moxie"})
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
        if filter_metadata:
            results = self.vector_db.similarity_search_with_relevance_scores(
                query, k=n_results, filter=filter_metadata
            )
        else:
            results = self.vector_db.similarity_search_with_relevance_scores(
                query, k=n_results
            )
        
        # Format results
        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                "text": doc.page_content,
                "metadata": doc.metadata,
                "relevance_score": score
            })
            
        return formatted_results

# Demo usage
if __name__ == "__main__":
    # Initialize the VectorDB with local embeddings (no API key needed)
    app_db = AppetiteGuideVectorDB(model_name="all-MiniLM-L6-v2")
    
    # Build the database
    app_db.build("guide.json")
    
    # Example search - general query
    print("\n=== General Search ===")
    results = app_db.search("What is Moxie's coverage for apartment buildings?")
    
    # Print results
    for i, result in enumerate(results):
        print(f"\nResult {i+1} (Score: {result['relevance_score']:.2f}):")
        print(f"Carrier: {result['metadata']['carrier_name']}")
        print(f"Type: {result['metadata']['type']}")
        
        # Print headers if present
        for key, value in result['metadata'].items():
            if key.startswith('Header'):
                print(f"{key}: {value}")
        
        # Print truncated content
        print("\nContent:")
        print(result['text'][:200] + "..." if len(result['text']) > 200 else result['text'])
    
    # Example search with filter by carrier
    print("\n=== Filtered Search (Moxie only) ===")
    filtered_results = app_db.search(
        "What are the eligibility criteria?",
        filter_metadata={"carrier_name": "Moxie"}
    )
    
    # Print filtered results
    for i, result in enumerate(filtered_results):
        print(f"\nResult {i+1} (Score: {result['relevance_score']:.2f}):")
        print(f"Type: {result['metadata']['type']}")
        
        # Print headers if present
        for key, value in result['metadata'].items():
            if key.startswith('Header'):
                print(f"{key}: {value}")
        
        # Print truncated content
        print("\nContent:")
        print(result['text'][:200] + "..." if len(result['text']) > 200 else result['text'])
