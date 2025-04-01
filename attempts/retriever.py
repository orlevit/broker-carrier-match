# pyright: reportAttributeAccessIssue=false

import json
from logger import logger
import os
from typing import Dict, List, Any
import openai
import chromadb
from chromadb.utils import embedding_functions
from config import SECTION_MAP_FILE,DB_PATH, DEFAULT_K_TOP_SIMILAR,DEFAULT_MAX_NUM_TOKENS, OPENAI_MODEL

class DocumentRetriever:
    def __init__(
        self,
        db_path: str = DB_PATH,
        section_map_path: str = SECTION_MAP_FILE,
        openai_api_key: str = None, # type: ignore
        k_top_similar: int = DEFAULT_K_TOP_SIMILAR,
        max_num_tokens: int = DEFAULT_MAX_NUM_TOKENS
    ):
        """
        Initialize the document retriever.
        
        Args:
            db_path: Path to the ChromaDB database
            section_map_path: Path to the section map JSON file
            openai_api_key: OpenAI API key for formulating metadata
            k_top_similar: Number of top similar documents to retrieve
            max_num_tokens: Maximum number of tokens to return
        """
        # Set OpenAI API key if provided
        if openai_api_key:
            openai.api_key = openai_api_key
        elif os.environ.get("OPENAI_API_KEY"):
            openai.api_key = os.environ.get("OPENAI_API_KEY")
        else:
            logger.warning("No OpenAI API key provided. GPT metadata extraction will fail.")
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=db_path)
        self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
        
        # Load all collections from ChromaDB
        self.collection = self.client.get_or_create_collection( name="document_collection", embedding_function=self.embedding_function) # type: ignore
        __import__('pdb').set_trace()
        # Load section map
        try:
            with open(section_map_path, 'r') as f:
                self.section_map = json.load(f)
            logger.info(f"Loaded section map from {section_map_path}")
        except Exception as e:
            logger.error(f"Failed to load section map: {e}")
            self.section_map = {}
        
        self.k_top_similar = k_top_similar
        self.max_num_tokens = max_num_tokens
    
    def _extract_metadata_with_gpt(self, user_question: str) -> Dict[str, Any]:
        """
        Use GPT to extract potential metadata from the user question.
        
        Args:
            user_question: The user's question
            
        Returns:
            Dict with extracted metadata for filtering
        """
        logger.info(f"Extracting metadata from question: {user_question}")
        
        # Create a system prompt for GPT
        system_prompt = """
        You are a metadata extraction tool. Your task is to analyze a user question and extract potential metadata
        for filtering insurance policy documents. You should return a JSON object with the following structure:
        
        {
          "request": [list of requested information types],
          "conditions": {
            "carrier_name": [list of carrier names mentioned],
            "geographical_region": {
              "include": [list of included regions],
              "exclude": [list of excluded regions]
            },
            "coverage": [list of coverage types mentioned],
            "capacity": [list of capacity descriptions],
            "limit": [list of limit descriptions],
            "Natural_disaster": {
              "include": [list of included natural disasters],
              "exclude": [list of excluded natural disasters]
            }
          }
        }
        
        Only include fields that are actually mentioned in the question. Use null for empty fields.
        Respond with ONLY the JSON object, nothing else.
        """
        
        # Call GPT to extract metadata
        response = openai.ChatCompletion.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_question}
            ],
            temperature=0.1
        )
        
        extracted_text = response.choices[0].message.content.strip()
        # Remove any markdown formatting if present
        if extracted_text.startswith("```json"):
            extracted_text = extracted_text.replace("```json", "").replace("```", "").strip()
        elif extracted_text.startswith("```"):
            extracted_text = extracted_text.replace("```", "").strip()
        
        metadata = json.loads(extracted_text)
        logger.info(f"Successfully extracted metadata: {metadata}")
        return metadata
    
    def _build_filter_from_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build a ChromaDB filter from the extracted metadata.
        
        Args:
            metadata: The extracted metadata
            
        Returns:
            A filter dictionary compatible with ChromaDB
        """
        filter_dict = {}
        
        # Process conditions if they exist
        if "conditions" in metadata:
            conditions = metadata["conditions"]
            
            # Process simple list fields
            for field in ["carrier_name", "coverage"]:
                if field in conditions and conditions[field]:
                    filter_dict[f"metadata.{field}"] = {"$in": conditions[field]}
            
            # Process geographical regions
            if "geographical_region" in conditions:
                geo_region = conditions["geographical_region"]
                if "include" in geo_region and geo_region["include"]:
                    filter_dict["metadata.geographical_region.include"] = {"$in": geo_region["include"]}
                if "exclude" in geo_region and geo_region["exclude"]:
                    filter_dict["metadata.geographical_region.exclude"] = {"$in": geo_region["exclude"]}
            
            # Process natural disasters
            if "Natural_disaster" in conditions:
                disasters = conditions["Natural_disaster"]
                if "include" in disasters and disasters["include"]:
                    filter_dict["metadata.Natural_disaster.include"] = {"$in": disasters["include"]}
                if "exclude" in disasters and disasters["exclude"]:
                    filter_dict["metadata.Natural_disaster.exclude"] = {"$in": disasters["exclude"]}
            
            # Process capacity and limit (more complex)
            for field in ["capacity", "limit"]:
                if field in conditions and conditions[field]:
                    # This is simplified - in a real implementation you might need
                    # more complex logic depending on how these are stored
                    filter_dict[f"metadata.{field}"] = {"$exists": True}
        
        logger.info(f"Built filter: {filter_dict}")
        return filter_dict
    
    def retrieve(self, user_question: str) -> str:
        """
        Retrieve documents based on user question, using both semantic similarity
        and metadata filtering.
        
        Args:
            user_question: The user's question
            
        Returns:
            A consolidated context string with the retrieved documents
        """
        logger.info(f"Retrieving documents for question: {user_question}")
        
        # First get the top K similar documents based on vector similarity
        vector_results = self.collection.query( query_texts=[user_question], n_results=self.k_top_similar)
        __import__('pdb').set_trace()
        vector_doc_ids = vector_results.get("ids", [[]])[0]
        vector_docs = vector_results.get("documents", [[]])[0]
        vector_metadatas = vector_results.get("metadatas", [[]])[0]
        vector_distances = vector_results.get("distances", [[]])[0]
        
        logger.info(f"Top {self.k_top_similar} similar documents by vector similarity: {vector_doc_ids}")
        
        # Extract metadata with GPT
        extracted_metadata = self._extract_metadata_with_gpt(user_question)
        metadata_filter = self._build_filter_from_metadata(extracted_metadata)
        
        # If metadata filter was successfully created
        if metadata_filter:
            # Get documents based on metadata filtering
            metadata_results = self.collection.query(
                query_texts=[user_question],
                where=metadata_filter,
                n_results=self.k_top_similar * 2  # Get more to check for non-consecutive
            )
            
            metadata_doc_ids = metadata_results.get("ids", [[]])[0]
            metadata_docs = metadata_results.get("documents", [[]])[0]
            metadata_metadatas = metadata_results.get("metadatas", [[]])[0]
            metadata_distances = metadata_results.get("distances", [[]])[0]
            
            logger.info(f"Documents retrieved by metadata filter: {metadata_doc_ids}")
            
            # Check if all metadata-filtered docs are in consecutive locations
            if metadata_doc_ids:
                section_numbers = [m.get("section_number", "") for m in metadata_metadatas]
                is_consecutive = self._check_consecutive_sections(section_numbers) # type: ignore
                if not is_consecutive:
                    logger.warning(
                        f"Non-consecutive document sections found: {section_numbers}"
                    )
            
            # Check if there are more similar docs than those filtered by metadata
            if vector_doc_ids and metadata_doc_ids:
                for i, (vec_id, vec_dist) in enumerate(zip(vector_doc_ids, vector_distances)):
                    if vec_id not in metadata_doc_ids:
                        # Find where this document would fit in metadata results by distance
                        for j, meta_dist in enumerate(metadata_distances):
                            if vec_dist < meta_dist:
                                logger.warning(
                                    f"Document {vec_id} (distance={vec_dist}) is more similar "
                                    f"than metadata-filtered document {metadata_doc_ids[j]} "
                                    f"(distance={meta_dist})"
                                )
                                break
            
            # Use metadata-filtered documents if available
            result_docs = metadata_docs
            result_metadatas = metadata_metadatas
        else:
            # Fall back to vector similarity if metadata extraction failed
            logger.warning("Using vector similarity results as fallback (no metadata filter)")
            result_docs = vector_docs
            result_metadatas = vector_metadatas
        
        # Consolidate the retrieved documents
        consolidated_text = self._consolidate_documents(result_docs, result_metadatas) # type: ignore
        
        # Truncate to max tokens (approximate)
        if len(consolidated_text.split()) > self.max_num_tokens:
            words = consolidated_text.split()
            consolidated_text = " ".join(words[:self.max_num_tokens])
            logger.info(f"Truncated context to {self.max_num_tokens} tokens")
        
        return consolidated_text
    
    def _check_consecutive_sections(self, section_numbers: List[str]) -> bool:
        """
        Check if section numbers are consecutive.
        
        Args:
            section_numbers: List of section numbers
            
        Returns:
            True if consecutive, False otherwise
        """
        # This is a simplified check - in reality, section numbering might be more complex
        sorted_sections = sorted(section_numbers, key=lambda x: [int(n) if n.isdigit() else n for n in x.split(".")])
        
        # Check if they're the same after sorting
        return sorted_sections == section_numbers
    
    def _consolidate_documents(self, documents: List[str], metadatas: List[Dict[str, Any]]) -> str:
        """
        Consolidate retrieved documents according to the section map.
        
        Args:
            documents: List of document contents
            metadatas: List of document metadatas
            
        Returns:
            Consolidated text
        """
        logger.info("Consolidating retrieved documents")
        
        # Group documents by carrier
        carrier_docs = {}
        for doc, metadata in zip(documents, metadatas):
            carrier_name = metadata.get("carrier_name", "unknown")
            section_number = metadata.get("section_number", "")
            
            if carrier_name not in carrier_docs:
                carrier_docs[carrier_name] = {}
                
            carrier_docs[carrier_name][section_number] = doc
        
        consolidated_text = ""
        
        for carrier_name, sections in carrier_docs.items():
            logger.info(f"Processing carrier: {carrier_name}")
            
            # Check if we have the full document and subsections
            carrier_map = self.section_map.get(carrier_name, {})
            full_doc_section = carrier_map.get("full_doc")
            
            if full_doc_section and full_doc_section in sections:
                # If we have the full document, use it and skip subsections
                consolidated_text += f"\n\n--- {carrier_name} Policy Document ---\n\n"
                
                # Add a summary
                summary = self._generate_summary(sections[full_doc_section])
                consolidated_text += f"Summary: {summary}\n\n"
                
                consolidated_text += sections[full_doc_section]
                logger.info(f"Used full document for {carrier_name}")
            else:
                # Otherwise use all sections we have
                consolidated_text += f"\n\n--- {carrier_name} Policy Sections ---\n\n"
                
                # Combine all sections
                all_section_text = "\n\n".join([
                    f"Section {section}: {content}" 
                    for section, content in sorted(sections.items())
                ])
                
                # Add a summary
                summary = self._generate_summary(all_section_text)
                consolidated_text += f"Summary: {summary}\n\n"
                
                consolidated_text += all_section_text
                logger.info(f"Used {len(sections)} sections for {carrier_name}")
        
        return consolidated_text
    
    def _generate_summary(self, text: str, max_length: int = 200) -> str:
        """
        Generate a short summary of the document text.
        
        Args:
            text: The document text to summarize
            max_length: Maximum length of summary in characters
            
        Returns:
            A summary string
        """
        # Use OpenAI to generate a summary
        response = openai.ChatCompletion.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "Generate a brief summary of the following insurance policy text in 1-2 sentences."},
                {"role": "user", "content": text[:4000]}  # Limit input size
            ],
            temperature=0.3,
            max_tokens=100
        )
        
        summary = response.choices[0].message.content.strip()
        if len(summary) > max_length:
            summary = summary[:max_length - 3] + "..."
        
        return summary


# Example usage
if __name__ == "__main__":
    retriever = DocumentRetriever(
        db_path=DB_PATH,
        section_map_path=SECTION_MAP_FILE, 
        k_top_similar=DEFAULT_K_TOP_SIMILAR,
        max_num_tokens=DEFAULT_MAX_NUM_TOKENS
    )
    
    user_question = "In which regions carrier supplies services, which cover houses damaged by Flood?"
    context = retriever.retrieve(user_question)
    
    print("\nRetrieved Context:")
    print(context)
