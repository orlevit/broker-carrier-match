import json
from typing import Dict, List, Any, Tuple
from config import Q_BASE_USER_PROMPT, Q_SYSTEM_PROMPT
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings  # or OpenAIEmbeddings, etc.
from config import (
    Q_SYSTEM_PROMPT, 
    Q_BASE_USER_PROMPT, 
    DB_PATH, 
    SECTION_MAP_FILE, 
    DEFAULT_K_TOP_SIMILAR, 
    DEFAULT_MAX_NUM_TOKENS, 
    COLLECTION_NAME,
    TOP_FILTER_NUM,
    TOP_SIMILARITY_NUM,
    TOTAL_TOP_SIMILAR

)
from llm_api import OpenAiAPI
from logger import logger


class DocumentRetriever:
    def __init__(
        self,
        section_map_path: str = SECTION_MAP_FILE,
        k_top_similar: int = DEFAULT_K_TOP_SIMILAR,
        max_num_tokens: int = DEFAULT_MAX_NUM_TOKENS
    ):
        """
        Initialize the document retriever.
        
        Args:
            section_map_path: Path to the section map JSON file
            k_top_similar: Number of top similar documents to retrieve
            max_num_tokens: Maximum number of tokens to return
        """
        self.system_prompt = Q_SYSTEM_PROMPT
        self.user_prompt = Q_BASE_USER_PROMPT
        self.total_top_similar = TOTAL_TOP_SIMILAR
        self.opeani_api = OpenAiAPI(self.system_prompt)
        self.top_similarity_num = TOP_SIMILARITY_NUM
        self.top_filter_num = TOP_FILTER_NUM

        # Initialize embeddings and vectordb
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectordb = Chroma(
            persist_directory=DB_PATH,
            embedding_function=self.embeddings,
            collection_name=COLLECTION_NAME
        )

        # Load section map
        with open(section_map_path, 'r') as f:
            self.section_map = json.load(f)
        logger.info(f"Loaded section map from {section_map_path}")
        
        self.k_top_similar = k_top_similar
        self.max_num_tokens = max_num_tokens
    
    def _build_filter_from_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build a ChromaDB filter from the extracted metadata.
        If multiple conditions exist, they will be combined under "$and".
        Example of final filter for multiple conditions:
            {
                "$and": [
                    {"coverage": {"$in": ["flood"]}},
                    {"geographical_region.exclude": {"$in": ["arizona"]}}
                ]
            }
        Returns an empty dict if no filters are formed.
        """
        sub_filters = []
        
        # carrier_name
        if "carrier_name" in metadata and metadata["carrier_name"]:
            carrier_val = (
                [metadata["carrier_name"]]
                if isinstance(metadata["carrier_name"], str)
                else metadata["carrier_name"]
            )
            sub_filters.append({"carrier_name": {"$in": carrier_val}})

        # coverage
        if "coverage" in metadata and metadata["coverage"]:
            coverage_val = (
                [metadata["coverage"]]
                if isinstance(metadata["coverage"], str)
                else metadata["coverage"]
            )
            sub_filters.append({"coverage": {"$in": coverage_val}})

        # geographical_region.include
        if "geographical_region.include" in metadata and metadata["geographical_region.include"]:
            sub_filters.append({
                "geographical_region.include": {"$in": metadata["geographical_region.include"]}
            })

        # geographical_region.exclude
        if "geographical_region.exclude" in metadata and metadata["geographical_region.exclude"]:
            sub_filters.append({
                "geographical_region.exclude": {"$in": metadata["geographical_region.exclude"]}
            })

        # Natural_disaster.include
        if "Natural_disaster.include" in metadata and metadata["Natural_disaster.include"]:
            sub_filters.append({
                "Natural_disaster.include": {"$in": metadata["Natural_disaster.include"]}
            })

        # Natural_disaster.exclude
        if "Natural_disaster.exclude" in metadata and metadata["Natural_disaster.exclude"]:
            sub_filters.append({
                "Natural_disaster.exclude": {"$in": metadata["Natural_disaster.exclude"]}
            })

        # capacity
        if "capacity" in metadata and metadata["capacity"]:
            sub_filters.append({"capacity": {"$exists": True}})

        # limit
        if "limit" in metadata and metadata["limit"]:
            sub_filters.append({"limit": {"$exists": True}})

        # Combine sub-filters under "$and" if more than one
        if len(sub_filters) > 1:
            filter_dict = {"$and": sub_filters}
        elif len(sub_filters) == 1:
            filter_dict = sub_filters[0]
        else:
            filter_dict = {}

        logger.info(f"Built filter from metadata: {filter_dict}")
        return filter_dict
    
    
    def _calculate_match_score(self,user_metadata: Dict[str, Any], db_result_metadata: Dict[str, str]) -> int:
        """
        Calculate the match score between user metadata and a database result metadata.
        
        Args:
        user_metadata: Dictionary containing user extracted metadata
        db_result_metadata: Dictionary containing metadata from database result
        
        Returns:
        int: The match score (number of matching keys)
        """
        match_score = 0
        
        for key in user_metadata:
            # Skip if key doesn't exist in db_result_metadata
            if key not in db_result_metadata:
                continue
            
            user_value = user_metadata[key]
            db_value_str = db_result_metadata[key]
            
            # Match based on the type of user_value
            if isinstance(user_value, list):
                # Empty list in user metadata matches empty string or empty list in DB
                if (user_value == [] or user_value == "" ) and ( db_value_str == '""' or db_value_str == "[]"):
                    continue

                elif isinstance(user_value, list):
                    # Check list against list
                    for item in user_value:
                        if item in db_value_str:
                            match_score += 1
            
            else:
                # For strings and other types
                user_value_str = str(user_value)
                db_value_str = str(db_value_str)
                
                # Check if user value is in db value
                if user_value_str in db_value_str:
                    match_score += 1
            
        return match_score

    def _rank_results_by_match(self,
        user_extracted_metadata: Dict[str, Any], 
        top_total_similar_results: List[Tuple[Dict[str, str], float]]
    ) -> List[Tuple[Dict[str, str], float, int]]:
        """
        Rank the database results based on metadata match score.
        
        Args:
        user_extracted_metadata: Dictionary containing user extracted metadata
        top_total_similar_results: List of (metadata, similarity_score) tuples from Chroma DB
        
        Returns:
        List of (metadata, similarity_score, match_score) tuples, sorted by match_score (descending)
        """
        ranked_results = []
        
        for db_doc, similarity_score in top_total_similar_results:
            db_metadata = db_doc.metadata
            match_score = self._calculate_match_score(user_extracted_metadata, db_metadata)
            ranked_results.append((db_doc, similarity_score, match_score))
        

        return ranked_results
    
    def _get_top_filter_and_similarity(self,ranked_results, top_similarity, top_filter):
        similarity_sorted = sorted(ranked_results, key=lambda x: (x[2], x[1]), reverse=True)[:top_similarity]
        filter_sorted = sorted(ranked_results, key=lambda x: (x[1], x[2]), reverse=True)[:top_filter]
        
        combined = similarity_sorted + filter_sorted


        unique_by_id = {}
        for doc in combined:
            doc_id = doc.metadata['id']
            if doc_id not in unique_by_id:
                unique_by_id[doc_id] = doc


        unique_documents = list(unique_by_id.values())
        
        return unique_documents
    def retrieve(self, user_question: str) -> str:
        """
        Retrieve documents based on the user's question, using both semantic similarity
        and metadata filtering.
        
        Args:
            user_question: The user's question
            
        Returns:
            A consolidated context string with the retrieved documents
        """
        logger.info(f"Retrieving documents for question: {user_question}")
        
        # Step 1: Vector similarity search (top K)
        langchain_results = self.vectordb.similarity_search_with_relevance_scores( user_question, k=self.total_top_similar)

        # Step 2: Extract metadata from user question
        user_question_prompt = self.user_prompt.format(user_question=user_question)
        user_extracted_metadata = self.opeani_api.extract_metadata_with_openai(user_question_prompt)
        
        # Step 3: get top docs
        __import__('pdb').set_trace()        
        rank_metadata = self._rank_results_by_match(user_extracted_metadata, langchain_results)
        top_docs = self._get_top_filter_and_similarity(rank_metadata, self.top_similarity_num, self.top_filter_num)

        # Step 4: Consolidate the retrieved documents
        consolidated_text = self._consolidate_documents(top_docs)
        
        # Step 5: Truncate to max_num_tokens (approximate by word count)
        words = consolidated_text.split()
        if len(words) > self.max_num_tokens:
            consolidated_text = " ".join(words[:self.max_num_tokens])
            logger.info(f"Truncated context to {self.max_num_tokens} tokens.")
        
        # send to openai for response
        summary = self.opeani_api.summary(sections[full_doc_section]["content"]) sjd foidsaj f----------------------------
        consolidated_text += f"Summaried context: {summary}\n\nThe context\n\n" + consolidated_text
        
        
        return consolidated_text
    
    def _check_consecutive_sections(self, section_numbers: List[str], distances: List[float]) -> bool:
        """
        Optional check if sections are consecutive by similarity.
        """
        paired_data = list(zip(section_numbers, distances))
        sorted_pairs = sorted(paired_data, key=lambda x: x[1])
        sorted_sections = [pair[0] for pair in sorted_pairs]

        for i in range(1, len(sorted_sections)):
            curr_parts = sorted_sections[i].split('.')
            prev_parts = sorted_sections[i-1].split('.')
            
            if len(curr_parts) > 1 and len(prev_parts) > 1:
                if curr_parts[0] == prev_parts[0]:  # same main section
                    if len(curr_parts) == len(prev_parts):
                        if int(curr_parts[-1]) != int(prev_parts[-1]) + 1:
                            return False
                    else:
                        if not (sorted_sections[i].startswith(sorted_sections[i-1]) or 
                                sorted_sections[i-1].startswith(sorted_sections[i])):
                            return False
        
        return True
    
    def _consolidate_documents(self, l_results: List[Dict[Any, Any]]) -> str:
        """
        Consolidate retrieved documents according to the section map.
        Remove subsections if a higher hierarchy section is already included.
        """
        logger.info("Consolidating retrieved documents.")
        documents = [doc for doc,_ in l_results]
        metadatas = [doc.metadata for doc,_  in l_results]
        # Group by carrier
        carrier_docs = {}
        for doc, metadata in zip(documents, metadatas):
            carrier_name = metadata.get("carrier_name", "unknown")
            section_number = metadata.get("section_number", "")
            doc_type = metadata.get("type", "section")  # default
            
            if carrier_name not in carrier_docs:
                carrier_docs[carrier_name] = {}
                
            carrier_docs[carrier_name][section_number] = {
                "content": doc,
                "type": doc_type
            }
        
        consolidated_text = ""
        
        # For each carrier, check if we have a "full_doc" or we combine sections
        for carrier_name, sections in carrier_docs.items():
            logger.info(f"Processing carrier: {carrier_name}")
            
            carrier_map = self.section_map.get(carrier_name, {})
            full_doc_section = carrier_map.get("full_doc")

            if full_doc_section and full_doc_section in sections:
                # Use the full document if present
                consolidated_text += f"\n\n--- {carrier_name} Policy Document ---\n\n"

                consolidated_text += sections[full_doc_section]["content"]
                logger.info(f"Used full document for {carrier_name}")
            else:
                # Otherwise filter out subsections if parent is present
                filtered_sections = self._filtered_subsections(sections)
                
                if filtered_sections:
                    consolidated_text += f"\n\n--- {carrier_name} Policy Sections ---\n\n"
                    # Combine all remaining sections
                    all_section_text = "\n\n".join([ f"{data['content']}" for _, data in sorted(filtered_sections.items()) ])
    
                    consolidated_text += all_section_text
                    logger.info(f"Used {len(filtered_sections)} sections for {carrier_name}")
        
        return consolidated_text
        
    def _is_descendant(self, child: str, parent: str) -> bool:
        """
        Returns True if 'child' == 'parent'
        or if 'child' starts with 'parent.' (indicating deeper subsection).
        """
        return (child == parent) or child.startswith(parent + '.')

    def _filtered_subsections(self, data: dict) -> dict:
        """
        Given a dict keyed by section_number, where each value is something like:
        {"content": "...", "type": "section" or "bullet", ...}
        this function removes all records overshadowed by a kept record of type='section'.

        Rules in summary:
          - A section_number S overshadows itself (S) and any deeper ones (S.x, S.x.y, etc.).
          - If type='section' is kept at S, then any bullet/section that shares or is deeper 
        than S is removed.

        """
        # 1) Convert data dict -> list of (sec_num, record, original_index)
        items = [(sec_num, record, idx) 
             for idx, (sec_num, record) in enumerate(data.items())]

        # 2) Define sorting keys

        # Depth = number of dots

        def section_depth(sec_num: str) -> int:
            return sec_num.count('.')

        # For tie-breaking by type:
        #  - 'section' should overshadow 'bullet' at the same depth
        #  - anything else is after 'bullet'
        def type_rank(t: str) -> int:
            if t == 'section':
                return 0
            elif t == 'bullet':
                return 1
            else:
                return 2

        items_sorted = sorted(
        items,
        key=lambda x: (
            section_depth(x[0]),        # sort by depth ascending
            type_rank(x[1]["type"])     # sections before bullets before others
        )
        )

        kept = []
        kept_sections = set()  # store the section_numbers of kept type='section'

        # 3) Iterate and filter
        for sec_num, record, original_idx in items_sorted:
            rec_type = record["type"]

            # Check if overshadowed by any previously kept section
            overshadowed = any(self._is_descendant(sec_num, ks) for ks in kept_sections)
            if overshadowed:
                # This item is overshadowed -> skip it
                continue

            # Not overshadowed -> keep it
            kept.append((sec_num, record, original_idx))

            # If it's a 'section', overshadow all its descendants
            if rec_type == "section":
                kept_sections.add(sec_num)

            # If it's 'full_doc', we do NOT overshadow by default
            # (If you want a 'full_doc' to also overshadow deeper sections,
            #  handle it similarly to 'section'.)

        # 4) (Optional) Re-sort 'kept' by the original dictionary order (original_idx)
        kept_sorted = sorted(kept, key=lambda x: x[2])

        # 5) Rebuild a dictionary
        filtered_data = {}
        for sec_num, record, _ in kept_sorted:
            filtered_data[sec_num] = record

        return filtered_data


# Example usage
if __name__ == "__main__":
    retriever = DocumentRetriever(
        section_map_path=SECTION_MAP_FILE,
        k_top_similar=DEFAULT_K_TOP_SIMILAR,
        max_num_tokens=DEFAULT_MAX_NUM_TOKENS
    )
    
    user_question = "In which regions, except arizona, does the carrier supply services that cover houses damaged by Flood?"
    context = retriever.retrieve(user_question)
    
    print("\nRetrieved Context:")
    print(context)
