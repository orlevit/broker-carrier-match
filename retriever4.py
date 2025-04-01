import json
import logging
import os
from re import TEMPLATE
from typing import Dict, List, Any, Tuple, Optional
from config import FAIL_USER_PROMPT, Q_BASE_USER_PROMPT, Q_SYSTEM_PROMPT, TEMPERATURE
import openai
from chromadb.utils import embedding_functions
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings  # or OpenAIEmbeddings, etc.
from config import Q_SYSTEM_PROMPT, Q_BASE_USER_PROMPT
from llm_api import OpenAiAPI

# Directories and file paths
DATA_DIR = "data"
INPUT_DIR = os.path.join(DATA_DIR, "input") 
OUTPUT_DIR = os.path.join(DATA_DIR, "output") 
DB_PATH = os.path.join(OUTPUT_DIR, "guide_db")
CHUNK_RESULTS_FILE = os.path.join(OUTPUT_DIR, "chunks.json")
STRUCTURED_METADATA_FILE = os.path.join(OUTPUT_DIR, "structured_metadata.json")
SECTION_MAP_FILE = os.path.join(OUTPUT_DIR, "section_map.json")
COLLECTION_NAME = 'langchain'
# Constants for retrieval
DEFAULT_K_TOP_SIMILAR = 5
DEFAULT_MAX_NUM_TOKENS = 4000
GPT_METADATA_MODEL = "gpt-4"
GPT_SUMMARY_MODEL = "gpt-3.5-turbo"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("retriever.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("document_retriever")

class DocumentRetriever:
    def __init__(
        self,
        db_path: str = DB_PATH,
        section_map_path: str = SECTION_MAP_FILE,
        openai_api_key: str = None,
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
        
        # Initialize ChromaDB client

        # Load all collections from ChromaDB
        self.system_prompt = Q_SYSTEM_PROMPT
        self.user_prompt = Q_BASE_USER_PROMPT
        self.opeani_api =  OpenAiAPI(self.system_prompt)
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectordb = Chroma( persist_directory=DB_PATH, embedding_function=self.embeddings, collection_name=COLLECTION_NAME)
        
        __import__('pdb').set_trace()
        # Load section map
        with open(section_map_path, 'r') as f:
            self.section_map = json.load(f)

        logger.info(f"Loaded section map from {section_map_path}")
        
        self.k_top_similar = k_top_similar
        self.max_num_tokens = max_num_tokens
    
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
        
        # First get the top K similar documents based on vector similarity using LangChain
        langchain_results = self.vectordb.similarity_search_with_relevance_scores(user_question, k=self.k_top_similar)
        
        # Extract documents and their metadata from LangChain results
        vector_docs = []
        vector_metadatas = []
        vector_doc_ids = []
        vector_distances = []
        
        for doc, score in langchain_results:
            vector_docs.append(doc.page_content)
            vector_metadatas.append(doc.metadata)
            # Use the document ID if available, otherwise create an identifier
            doc_id = doc.metadata.get("id", f"doc_{len(vector_doc_ids)}")
            vector_doc_ids.append(doc_id)
            # Convert similarity score to distance (1.0 - score)
            vector_distances.append(1.0 - score)
        
        logger.info(f"Top {self.k_top_similar} similar documents by vector similarity: {vector_doc_ids}")
        
        # Extract metadata with GPT
        user_question_prompt = self.user_prompt.format(user_question=user_question)
        extracted_metadata = self.opeani_api.extract_metadata_with_openai(user_question_prompt)
        metadata_filter = self._build_filter_from_metadata(extracted_metadata)
        
        # If metadata filter was successfully created
        if metadata_filter:
            # Use the direct ChromaDB client for metadata filtering
            metadata_results = self.vectordb.query(
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
                is_consecutive = self._check_consecutive_sections(section_numbers, metadata_distances)
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
        consolidated_text = self._consolidate_documents(result_docs, result_metadatas)
        
        # Truncate to max tokens (approximate)
        if len(consolidated_text.split()) > self.max_num_tokens:
            words = consolidated_text.split()
            consolidated_text = " ".join(words[:self.max_num_tokens])
            logger.info(f"Truncated context to {self.max_num_tokens} tokens")
        
        return consolidated_text
    
    def _check_consecutive_sections(self, section_numbers: List[str], distances: List[float]) -> bool:
        """
        Check if sections are consecutive by similarity score to the user question.
        
        Args:
            section_numbers: List of section numbers
            distances: List of distance scores (lower is more similar)
            
        Returns:
            True if consecutive by similarity, False otherwise
        """
        # Sort sections by distance (similarity to query)
        paired_data = list(zip(section_numbers, distances))
        sorted_pairs = sorted(paired_data, key=lambda x: x[1])  # Sort by distance
        sorted_sections = [pair[0] for pair in sorted_pairs]
        
        # Check if any non-consecutive sections exist in the sorted order
        for i in range(1, len(sorted_sections)):
            curr_parts = sorted_sections[i].split('.')
            prev_parts = sorted_sections[i-1].split('.')
            
            # Check if they're from the same hierarchy branch
            if len(curr_parts) > 1 and len(prev_parts) > 1:
                if curr_parts[0] == prev_parts[0]:  # Same main section
                    # Check if they're consecutive
                    if len(curr_parts) == len(prev_parts):
                        # Same level - check if they're consecutive numbers
                        if int(curr_parts[-1]) != int(prev_parts[-1]) + 1:
                            return False
                    else:
                        # Different levels - check if it's a direct parent/child or consecutive
                        if not (sorted_sections[i].startswith(sorted_sections[i-1]) or 
                               sorted_sections[i-1].startswith(sorted_sections[i])):
                            return False
        
        return True
    
    def _consolidate_documents(self, documents: List[str], metadatas: List[Dict[str, Any]]) -> str:
        """
        Consolidate retrieved documents according to the section map.
        Remove subsections if a higher hierarchy section is already included.
        
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
            doc_type = metadata.get("type", "section")  # Default to section if type not specified
            
            if carrier_name not in carrier_docs:
                carrier_docs[carrier_name] = {}
                
            # Store document with its metadata
            carrier_docs[carrier_name][section_number] = {
                "content": doc,
                "type": doc_type
            }
        
        consolidated_text = ""
        
        for carrier_name, sections in carrier_docs.items():
            logger.info(f"Processing carrier: {carrier_name}")
            
            # Check if we have the full document
            carrier_map = self.section_map.get(carrier_name, {})
            full_doc_section = carrier_map.get("full_doc")
            
            if full_doc_section and full_doc_section in sections:
                # If we have the full document, use it and skip all subsections
                consolidated_text += f"\n\n--- {carrier_name} Policy Document ---\n\n"
                
                # Add a summary
                summary = self._generate_summary(sections[full_doc_section]["content"])
                consolidated_text += f"Summary: {summary}\n\n"
                
                consolidated_text += sections[full_doc_section]["content"]
                logger.info(f"Used full document for {carrier_name}")
            else:
                # Otherwise filter out subsections if a parent section is present
                filtered_sections = self._filter_subsections(sections)
                
                if filtered_sections:
                    consolidated_text += f"\n\n--- {carrier_name} Policy Sections ---\n\n"
                    
                    # Combine all remaining sections
                    all_section_text = "\n\n".join([
                        f"Section {section}: {data['content']}" 
                        for section, data in sorted(filtered_sections.items())
                    ])
                    
                    # Add a summary
                    summary = self._generate_summary(all_section_text)
                    consolidated_text += f"Summary: {summary}\n\n"
                    
                    consolidated_text += all_section_text
                    logger.info(f"Used {len(filtered_sections)} sections for {carrier_name}")
        
        return consolidated_text
        
    def _filter_subsections(self, sections: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Filter out subsections if a parent section is already present.
        
        Args:
            sections: Dictionary of section_number -> document info
            
        Returns:
            Filtered dictionary of sections
        """
        filtered_sections = {}
        section_keys = list(sections.keys())
        
        # First pass: add all sections to filtered list
        for section_num, data in sections.items():
            filtered_sections[section_num] = data
        
        # Second pass: remove items that are subsections of included sections
        to_remove = set()
        for section_num in section_keys:
            # Skip bullet points as they're always kept with their parent section
            if sections[section_num]["type"] == "bullet":
                continue
                
            for other_section in section_keys:
                # Skip comparing to self
                if section_num == other_section:
                    continue
                    
                # If this is a subsection of another section we have
                if (other_section != section_num and 
                    section_num.startswith(other_section + ".") and
                    other_section in filtered_sections):
                    to_remove.add(section_num)
                    logger.info(f"Removing subsection {section_num} (parent section {other_section} exists)")
                    break
        
        # Remove the identified subsections
        for section_num in to_remove:
            if section_num in filtered_sections:
                del filtered_sections[section_num]
        
        return filtered_sections
    
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
            model=GPT_SUMMARY_MODEL,
            messages=[
                {"role": "system", "content": "Generate a brief summary of the following insurance policy text in 1-2 sentences."},
                {"role": "user", "content": text[:4000]}  # Limit input size
            ],
            temperature=TEMPERATURE,
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
        max_num_tokens=DEFAULT_MAX_NUM_TOKENS,
        openai_api_key=openai.api_key
    )
    
    user_question = "In which regions carrier supplies services, which cover houses damaged by Flood?"
    context = retriever.retrieve(user_question)
    
    print("\nRetrieved Context:")
    print(context)
