import os
import logging
import csv
import hashlib
from dotenv import load_dotenv

import chromadb
from llama_index.core import Settings
from llama_index.core import Document
from llama_index.core.schema import BaseNode
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from config import CHROMA_PATH, CHROMA_COLLECTION, FILES, CHUNK_SIZE, CHUNK_OVERLAP

Settings.llm = OpenAI(temperature=0, model="gpt-4o-mini")
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small"
)

load_dotenv()

def deterministic_id_func(i: int, doc: BaseNode) -> str:
    """Deterministic ID function for the text splitter.
    This will be used to generate a unique repeatable identifier for each node."""
    unique_identifier = doc.id_ + str(i)
    hasher = hashlib.sha256()
    hasher.update(unique_identifier.encode('utf-8')) 
    return hasher.hexdigest()

def create_db(return_nodes=False):
    rows = []
    # Load the file as a JSON
    for FILE in FILES:
        with open(FILE, mode="r", encoding="utf-8") as file:
            csv_reader = csv.reader(file)

            for idx, row in enumerate(csv_reader):
                if idx == 0: continue # Skip header row
                rows.append(row)

    # Convert the chunks to Document objects so the LlamaIndex framework can process them.
    documents = [Document(text=row[1], metadata={"title": row[0], "url": row[2]}) for row in rows]
    # By default, the node/chunks ids are set to random uuids. To ensure same id's per run, we manually set them.
    for idx, doc in enumerate(documents):
        doc.id_ = f"doc_{idx}"

    # Define the splitter object that split the text into segments with 512 tokens,
    # with a 128 overlap between the segments.
    text_splitter = TokenTextSplitter(
        separator=" ", chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        id_func=deterministic_id_func
    )

    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    chroma_collection = chroma_client.get_or_create_collection(CHROMA_COLLECTION)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    # Create the pipeline to apply the transformation (splitting and embedding) on each chunk,
    # and store the transformed text in the chroma vector store.
    pipeline = IngestionPipeline(
        transformations=[
            text_splitter,
            OpenAIEmbedding(model = 'text-embedding-3-small'),
        ],
        vector_store=vector_store
    )

    # Run the transformation pipeline.
    nodes = pipeline.run(documents=documents, show_progress=True)

    db = chromadb.PersistentClient(path=CHROMA_PATH)
    chroma_collection = db.get_or_create_collection(CHROMA_COLLECTION)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    index = VectorStoreIndex.from_vector_store(vector_store)
    if return_nodes:
        return nodes
    else:
        return index


def load_db():
    chroma_client = chromadb.PersistentClient(CHROMA_PATH)
    chroma_collection = chroma_client.get_collection(CHROMA_COLLECTION)

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        show_progress=True,
        use_async=True,
        embed_model=Settings.embed_model
    )

    return index

def load_asset(file):
    """Load CSS from an external file"""
    if os.path.exists(file):
        with open(file, "r", encoding="utf-8") as f:
            return f.read()

def num_tokens_from_messages(messages, model="gpt-4"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    
    tokens_per_message = 3
    tokens_per_name = 1
    
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    
    return num_tokens
