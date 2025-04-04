import json
import re
import os
from typing import List, Dict, Any, Tuple
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
import markdown
from bs4 import BeautifulSoup
from llm_api import OpenAiAPI
from logger import logger
from config import APETITE_SYSTEM_PROMPT, APETITE_BASE_USER_PROMPT, DB_PATH, CHUNK_RESULTS_FILE, STRUCTURED_METADATA_FILE, SECTION_MAP_FILE, EMBEDDING_MODEL_NAME, MAX_HEADER_DEPTH, COLLECTION_NAME, NEEDED_FIELD_NAMES

class AppetiteGuideVectorDB:
    def __init__(self):
        """
        Initialize the vector database for carrier appetite guides
        """

        self.logger = logger
        self.system_prompt = APETITE_SYSTEM_PROMPT
        self.base_user_prompt = APETITE_BASE_USER_PROMPT
        self.db_path = DB_PATH
        self.chunk_results_file = CHUNK_RESULTS_FILE
        self.structured_metadata_file = STRUCTURED_METADATA_FILE
        self.section_map_file = SECTION_MAP_FILE
        self.model_name = EMBEDDING_MODEL_NAME
        self.max_header_depth = MAX_HEADER_DEPTH
        self.opeani_api = OpenAiAPI(self.system_prompt)
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        except Exception as e:
            self.logger.error("Failed to load HuggingFaceEmbeddings: %s", e)
            self.embeddings = None
        self.vector_db = None

    def load_data(self, file_path: str) -> List[Dict[str, str]]:
        """Load appetite guides from JSON file"""

        self.logger.info("Loading guide data from file: %s", file_path)
        with open(file_path, "r") as f:
            data = json.load(f)
        return data

    def parse_markdown_hierarchy(self, markdown_text: str) -> List[Dict[str, Any]]:
        """
        Parse markdown text into a hierarchical structure for chunking.
        Returns list of sections with their headers and content.
        """

        headers_to_split_on = []
        for i in range(1, self.max_header_depth + 1):
            header_marker = "#" * i
            header_name = f"Header{i}"
            headers_to_split_on.append((header_marker, header_name))
        splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        sections = splitter.split_text(markdown_text)
        return sections

    def generate_section_number_map(self, sections: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Generate a flat section name → section number mapping.
        Handles headers from Header1 to HeaderN properly including nested headers.
        """
        
        section_map = {"full_doc": "1"}
        section_counters = {i: 0 for i in range(1, self.max_header_depth + 1)}
        current_parents = {i: None for i in range(1, self.max_header_depth + 1)}
        section_counters[1] = 1
        for section in sections:
            section_level = None
            section_name = None
            for j in reversed(range(1, self.max_header_depth + 1)):
                header_key = f"Header{j}"
                if header_key in section.metadata:
                    section_level = j
                    section_name = section.metadata[header_key]
                    break
            if section_level and section_name:
                for j in range(section_level + 1, self.max_header_depth + 1):
                    section_counters[j] = 0
                    current_parents[j] = None
                section_counters[section_level] += 1
                section_number = ".".join(str(section_counters[i]) for i in range(1, section_level + 1))
                current_parents[section_level] = section_number
                section_map[section_name] = section_number
                for j in range(1, section_level):
                    parent_title = section.metadata.get(f"Header{j}")
                    if parent_title and parent_title not in section_map:
                        parent_number = ".".join(str(section_counters[i]) for i in range(1, j + 1))
                        section_map[parent_title] = parent_number
        return section_map

    def extract_bullet_points(self, text: str) -> List[str]:
        """Extract bullet points from text section."""

        html = markdown.markdown(text)
        soup = BeautifulSoup(html, "html.parser")
        bullet_points = []
        list_items = soup.find_all("li")
        for li in list_items:
            bullet_points.append(f"* {li.get_text()}")
        if not bullet_points:
            bullet_regex = re.compile(r"^[\s-]*[-•*]\s+(.*?)$", re.MULTILINE)
            matches = bullet_regex.findall(text)
            bullet_points = [f"* {match}" for match in matches]
        return bullet_points

    def additional_metadata_info(self, metadata: Dict[str, Any], section_name: str, section_map: Dict[str, str], txt_type: str) -> Dict[str, Any]:
        """
        Add parent/section number info to the metadata
        """

        if section_name in section_map:
            metadata["section_number"] = section_map[section_name]
        else:
            metadata["section_number"] = None
        metadata["type"] = txt_type
        return metadata

    def create_hierarchical_chunks(
        self, sections: List[Dict[str, Any]], carrier_name: str, full_guide: str
    ) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
        """
        Create hierarchical chunks based on the section structure and bullet points.
        Returns a list of chunk dicts plus the section map.
        """

        section_map = self.generate_section_number_map(sections)
        self.logger.info("Extracting metadata for the full guide: %s", carrier_name)
        user_prompt = self.base_user_prompt.format(text=full_guide, carrier_name=carrier_name)
        full_guide_metadata_part = self.opeani_api.extract_metadata_with_openai(user_prompt, carrier_name)
        full_guide_metadata_part["section_number"] = "1"
        chunks = []
        full_guide_metadata = self.additional_metadata_info(full_guide_metadata_part, 'full_doc', section_map, 'full_doc')
        full_guide_chunk = {"text": full_guide, "metadata": full_guide_metadata}
        chunks.append(full_guide_chunk)
        self.logger.info(f'full_guide metadata: {json.dumps(full_guide_chunk)}')
        for section in sections:
            section_text = ""
            section_name = None
            for i in range(1, self.max_header_depth + 1):
                header_key = f"Header{i}"
                if header_key in section.metadata:
                    section_name = section.metadata[header_key]
                    header_level = "#" * i
                    section_text += f"{header_level} {section_name}\n\n"
            section_text += section.page_content
            section_metadata = self.opeani_api.extract_metadata_with_openai(section_text, carrier_name)
            if section_name:
                section_metadata = self.additional_metadata_info(section_metadata, section_name, section_map, 'section')
            section_chunk = {"text": section_text, "metadata": section_metadata}
            chunks.append(section_chunk)
            self.logger.info(f'section metadata: {json.dumps(section_chunk)}')
            bullet_points = self.extract_bullet_points(section.page_content)
            for bullet in bullet_points:
                bullet_text = ""
                for i in range(1, self.max_header_depth + 1):
                    header_key = f"Header{i}"
                    if header_key in section.metadata:
                        parent_name = section.metadata[header_key]
                        bullet_text += f"{('#' * i)} {parent_name}\n\n"
                bullet_text += bullet
                bullet_metadata = self.opeani_api.extract_metadata_with_openai(bullet_text, carrier_name)
                if section_name:
                    bullet_metadata = self.additional_metadata_info(bullet_metadata, section_name, section_map, "bullet")
                bullet_chunk = {"text": bullet_text, "metadata": bullet_metadata}
                chunks.append(bullet_chunk)
                self.logger.info(f'bullet metadata: {json.dumps(bullet_chunk)}')
        return chunks, section_map

    def process_guide(
        self, guide_data: Dict[str, str]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, str]]:
        """
        Process a single carrier's appetite guide, returning:
          - the list of chunks,
          - the "full guide" metadata,
          - the section map
        """

        carrier_name = guide_data["carrier_name"]
        guide_text = guide_data["guide"]
        self.logger.info("Processing guide for carrier: %s", carrier_name)
        sections = self.parse_markdown_hierarchy(guide_text)
        chunks, section_map = self.create_hierarchical_chunks(sections, carrier_name, guide_text)
        if chunks:
            full_guide_metadata = chunks[0]["metadata"]
        else:
            full_guide_metadata = {}
            for key in NEEDED_FIELD_NAMES:
                full_guide_metadata[key] = None
            full_guide_metadata["carrier_name"] = carrier_name
            full_guide_metadata["section_number"] = 1
        return chunks, full_guide_metadata, section_map

    def process_all_guides(
        self, guides_data: List[Dict[str, str]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Dict], Dict[str, Dict[str, str]]]:
        """
        Process all carrier appetite guides, returning:
          - all chunks
          - all metadata (by carrier)
          - all section maps (by carrier)
        """

        all_chunks = []
        all_metadata = {}
        all_section_maps = {}
        for guide_data in guides_data:
            carrier_name = guide_data["carrier_name"]
            chunks, metadata, section_map = self.process_guide(guide_data)
            all_chunks.extend(chunks)
            all_metadata[carrier_name] = metadata
            all_section_maps[carrier_name] = section_map
        return all_chunks, all_metadata, all_section_maps

    def save_chunks(self, chunks: List[Dict[str, Any]], output_path: Any = None) -> None:
        """Save chunk data to JSON."""

        if output_path is None:
            output_path = self.chunk_results_file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.logger.info("Saving chunk data to: %s", output_path)
        with open(output_path, "w") as f:
            json.dump(chunks, f, indent=2)

    def load_chunks(self, chunk_file: str) -> List[Dict[str, Any]]:
        """Load chunk data from JSON file."""

        self.logger.info("Loading chunk data from: %s", chunk_file)
        with open(chunk_file, "r") as f:
            chunks = json.load(f)
        return chunks

    def save_structured_metadata(
        self, metadata: Dict[str, Dict], output_path: str = None
    ) -> None:
        """Save structured metadata to JSON."""

        if output_path is None:
            output_path = self.structured_metadata_file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.logger.info("Saving structured metadata to: %s", output_path)
        with open(output_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def save_section_map(
        self, section_map: Dict[str, Dict[str, str]], output_path: str = None
    ) -> None:
        """Save the section map to a JSON file."""

        if output_path is None:
            output_path = self.section_map_file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.logger.info("Saving section map to: %s", output_path)
        with open(output_path, "w") as f:
            json.dump(section_map, f, indent=2)

    def serialized_metadata(self, metadata: Dict[List, Any]) -> Dict[str, str]:

        serialized_metatdata = {}
        for k, v in metadata.items():
            serialized_metatdata[k] = json.dumps(v)
        return serialized_metatdata

    def build_vector_db_from_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Given a list of chunks, build and persist a new Chroma vector DB.
        """

        self.logger.info("Building new Chroma vector DB from chunks...")
        if not self.embeddings:
            raise ValueError("Embeddings model not initialized.")
        if not chunks:
            self.logger.warning("No chunks to build from.")
            return
        docs = []
        for c in chunks:
            docs.append(Document(page_content=c["text"], metadata=self.serialized_metadata(c["metadata"])))
        self.vector_db = Chroma.from_documents(
            documents=docs,
            embedding=self.embeddings,
            collection_name=COLLECTION_NAME,
            persist_directory=self.db_path
        )
        self.logger.info("Persisting new vector DB to: %s", self.db_path)

    def load_vector_db(self) -> None:
        """
        Load an existing Chroma vector DB (if available).
        """

        self.logger.info("Loading existing Chroma vector DB at: %s", self.db_path)
        self.vector_db = Chroma(
            persist_directory=self.db_path,
            embedding_function=self.embeddings,
            collection_name=COLLECTION_NAME
        )

    def build(self, file_path: str) -> Dict[str, Dict]:
        """
        Main entry point to build (or load) the vector DB and structured metadata
        from the given guide JSON file.
        """

        self.logger.info("Begin build process for file: %s", file_path)
        if os.path.isdir(self.db_path) and os.path.exists(os.path.join(self.db_path, "index")):
            self.logger.info("Vector DB directory found at '%s'; loading DB instead of rebuilding.", self.db_path)
            self.load_vector_db()
        else:
            self.logger.info("No existing vector DB found at '%s'.", self.db_path)
            if os.path.exists(self.chunk_results_file):
                self.logger.info("Chunks file found. Loading chunks from '%s'.", self.chunk_results_file)
                chunks = self.load_chunks(self.chunk_results_file)
            else:
                self.logger.info("No chunks file found. Will process guides from scratch.")
                guides_data = self.load_data(file_path)
                chunks, structured_metadata, section_map = self.process_all_guides(guides_data)
                self.save_chunks(chunks, self.chunk_results_file)
                self.save_structured_metadata(structured_metadata, self.structured_metadata_file)
                self.save_section_map(section_map, self.section_map_file)
            if not self.vector_db:
                if not os.path.exists(self.chunk_results_file):
                    raise FileNotFoundError("Unable to find chunk data to build the vector DB.")
                chunks = self.load_chunks(self.chunk_results_file)
                self.build_vector_db_from_chunks(chunks)
        if os.path.exists(self.structured_metadata_file):
            with open(self.structured_metadata_file, "r") as f:
                structured_metadata = json.load(f)
            return structured_metadata
        else:
            self.logger.warning("No structured metadata file found at '%s'.", self.structured_metadata_file)
            return {}

