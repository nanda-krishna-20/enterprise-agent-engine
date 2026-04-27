import os
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Path Management
# This dynamically finds the root of the project so the DB always saves in the right place.
SRC_DIR = Path(__file__).parent
PROJECT_ROOT = SRC_DIR.parent
DB_DIR = PROJECT_ROOT / "data" / "chroma_db"

# 2. The Mock Enterprise Data
mock_documents = [
    {
        "id": "doc_1",
        "title": "Richardson Tech Corridor - Q3 Commercial Real Estate Report",
        "content": """The Richardson Tech Corridor continues to see high demand for Class A office space, 
        driven by expansion in the telecommunications and AI sectors. Vacancy rates have dropped to 12.4%, 
        with average asking rents stabilizing at $32.50 per square foot. Major lease renewals by telecom 
        giants indicate strong tenant retention. Cap rates for premium properties remain compressed at 
        around 5.2%. However, older Class B properties are seeing increased vacancy as companies prioritize 
        modern amenities to lure remote workers back to the office."""
    },
    {
        "id": "doc_2",
        "title": "Industrial Logistics Hub - Dallas Fort Worth Valuation",
        "content": """Industrial property valuations in the DFW metroplex have surged by 15% year-over-year. 
        The primary driver is the demand for last-mile delivery distribution centers. Net Operating Income (NOI) 
        for properties exceeding 100,000 square feet has grown significantly. A recently appraised distribution 
        center near the DFW airport was valued at $45 million, reflecting a historically low cap rate of 4.1%. 
        Supply chain bottlenecks are easing, leading to faster construction times for new industrial parks, 
        though raw material costs remain a slight headwind."""
    },
    {
        "id": "doc_3",
        "title": "Internal Memo: Corporate Sustainability Mandates 2026",
        "content": """Effective January 1, 2026, all newly acquired commercial properties must meet LEED 
        Gold certification standards. Properties currently in the portfolio that fall below LEED Silver must 
        undergo energy efficiency retrofits within the next 24 months. The budget allocation for HVAC upgrades 
        and solar panel installations has been increased by $120 million. Agents evaluating potential acquisitions 
        must heavily weight the ESG (Environmental, Social, and Governance) score of the property during the 
        due diligence phase."""
    }
]

def run_etl_pipeline():
    print("--- Starting ETL Pipeline ---")
    
    # 3. Extract
    texts = [doc["title"] + "\n\n" + doc["content"] for doc in mock_documents]
    metadatas = [{"source": doc["title"], "doc_id": doc["id"]} for doc in mock_documents]

    # 4. Transform: Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        length_function=len
    )
    
    print("Chunking documents...")
    chunks = text_splitter.create_documents(texts, metadatas=metadatas)
    print(f"Created {len(chunks)} distinct text chunks.")

    # 5. Transform: Embeddings
    print("Loading Hugging Face embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 6. Load: Insert into ChromaDB
    print(f"Initializing ChromaDB and saving to: {DB_DIR}")
    
    # Ensure the data directory exists
    DB_DIR.parent.mkdir(parents=True, exist_ok=True)
    
    vectorstore = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings,
        persist_directory=str(DB_DIR)
    )
    
    print("\n--- ETL Pipeline Complete ---")
    print("Documents are embedded and stored successfully.")

if __name__ == "__main__":
    run_etl_pipeline()