from logger import logger
from vectorstore import AppetiteGuideVectorDB
from retriever import DocumentRetriever
from config import INPUT_GUIDE_FILE, SECTION_MAP_FILE, DEFAULT_K_TOP_SIMILAR, DEFAULT_MAX_NUM_TOKENS

if __name__ == "__main__":
    logger.info("Starting the appetite guide vector DB build process...")
    
    app_db = AppetiteGuideVectorDB()
    structured_metadata = app_db.build(INPUT_GUIDE_FILE)

    logger.info("Build process completed. Structured metadata keys: %s", list(structured_metadata.keys()))


    user_question = input("Please enter your question: ")
    retriever = DocumentRetriever(section_map_path=SECTION_MAP_FILE, k_top_similar=DEFAULT_K_TOP_SIMILAR, max_num_tokens=DEFAULT_MAX_NUM_TOKENS)
    answer = retriever.retrieve(user_question)

    print("\nAnswer:\n")
    print(answer)
