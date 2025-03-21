import os
import json
from dotenv import load_dotenv

from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding

from llama_index.llms.openai import OpenAI
from llama_index.llms.gemini import Gemini

from llama_index.core.evaluation import generate_question_context_pairs
from llama_index.core.evaluation import RetrieverEvaluator
from llama_index.core.query_engine import RetrieverQueryEngine

from utils import create_db
from config import CHROMA_PATH, CHROMA_COLLECTION

load_dotenv()

Settings.llm = OpenAI(temperature=0, model="gpt-4o-mini", max_tokens=512)
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small"
)

nodes = create_db(return_nodes=True)

# Free Tier-Gemini API key
from llama_index.core.llms.utils import LLM
from llama_index.core.schema import MetadataMode, TextNode
from tqdm import tqdm
import json
import re
import uuid
import warnings
import time
from typing import Dict, List, Tuple
from llama_index.core.evaluation import EmbeddingQAFinetuneDataset

DEFAULT_QA_GENERATE_PROMPT_TMPL = """\
Context information is below.

---------------------
{context_str}
---------------------

Given the context information and not prior knowledge.
generate only questions based on the below query.

You are a Teacher/ Professor. Your task is to setup \
{num_questions_per_chunk} questions for an upcoming \
quiz/examination. The questions should be diverse in nature \
across the document. Restrict the questions to the \
context information provided."
"""

def generate_question_context_pairs(
    nodes: List[TextNode],
    llm: LLM,
    qa_generate_prompt_tmpl: str = DEFAULT_QA_GENERATE_PROMPT_TMPL,
    num_questions_per_chunk: int = 2,
    request_delay: float = 2.0
) -> EmbeddingQAFinetuneDataset:
    """Generate examples given a set of nodes with delays between requests."""
    node_dict = {
        node.node_id: node.get_content(metadata_mode=MetadataMode.NONE)
        for node in nodes
    }

    queries = {}
    relevant_docs = {}

    for node_id, text in tqdm(node_dict.items()):
        query = qa_generate_prompt_tmpl.format(
            context_str=text, num_questions_per_chunk=num_questions_per_chunk
        )
        response = llm.complete(query)

        result = str(response).strip().split("\n")
        questions = [
            re.sub(r"^\d+[\).\s]", "", question).strip() for question in result
        ]
        questions = [question for question in questions if len(question) > 0][
            :num_questions_per_chunk
        ]

        num_questions_generated = len(questions)
        if num_questions_generated < num_questions_per_chunk:
            warnings.warn(
                f"Fewer questions generated ({num_questions_generated}) "
                f"than requested ({num_questions_per_chunk})."
            )

        for question in questions:
            question_id = str(uuid.uuid4())
            queries[question_id] = question
            relevant_docs[question_id] = [node_id]

        time.sleep(request_delay)

    return EmbeddingQAFinetuneDataset(
        queries=queries, corpus=node_dict, relevant_docs=relevant_docs
    )

#from llama_index.core.evaluation import generate_question_context_pairs
from llama_index.llms.gemini import Gemini

llm = Gemini(model="models/gemini-1.5-flash", temperature=1, max_tokens=512)

rag_eval_dataset = generate_question_context_pairs(
    nodes[:25],
    llm=llm,
    num_questions_per_chunk=1,
    request_delay=4
)

# Save the dataset as a json file for later use
rag_eval_dataset.save_json("./rag_eval_dataset.json")
