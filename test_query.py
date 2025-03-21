import os
from dotenv import load_dotenv

from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding

from llama_index.llms.openai import OpenAI
from llama_index.llms.gemini import Gemini

from utils import load_db, create_db
from config import CHROMA_PATH, CHROMA_COLLECTION, FILES, CHUNK_SIZE, CHUNK_OVERLAP

load_dotenv()

Settings.llm = OpenAI(temperature=0, model="gpt-4o-mini")
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small"
)

if not os.path.exists(CHROMA_PATH) or not os.listdir(CHROMA_PATH):
    print("⚠️ ChromaDB not found. Creating DB...")
    index = create_db()
else:
    print("✅ ChromaDB found. Loading DB...", CHROMA_PATH, CHROMA_COLLECTION)
    index = load_db()


q="What is LittleJS?"
q="How to install?"
q="Who is the author?"
# q="How to boil an egg?"
# q="What params does the EngineObject require?"
# q="What is vec2?"
query_engine = index.as_query_engine(llm=Settings.llm, similarity_top_k=5)
res = query_engine.query(q)

print(res.response)
