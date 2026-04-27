# src/tools.py
from crewai.tools import tool
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from pathlib import Path

# 1. Path Management 
SRC_DIR = Path(__file__).parent
PROJECT_ROOT = SRC_DIR.parent
DB_DIR = PROJECT_ROOT / "data" / "chroma_db"

# 2. Initialize the Database Connection
print("Connecting to Vector DB for Agent Tools...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# We open the database in read-mode
vectorstore = Chroma(
    persist_directory=str(DB_DIR), 
    embedding_function=embeddings
)

# 3. The Custom Tool Decorator 
# The Docstring ("""...""") is CRITICAL. The LLM reads this string to decide IF it should use this tool.
@tool("Search Real Estate Database")
def search_real_estate_db(query: str) -> str:
    """
    Useful for searching proprietary commercial real estate reports, property valuations, 
    and internal corporate sustainability mandates. 
    Input should be a specific search query.
    """
    print(f"\n[Tool Executing] Searching DB for: '{query}'")
    
    # Perform the K-Nearest Neighbors (KNN) vector search
    docs = vectorstore.similarity_search(query, k=3)
    
    if not docs:
        return "No relevant documents found in the database."
    
    # Format the retrieved chunks for the LLM to read
    results = []
    for doc in docs:
        title = doc.metadata.get("source", "Unknown Document")
        results.append(f"--- Document: {title} ---\n{doc.page_content}")
        
    return "\n\n".join(results)