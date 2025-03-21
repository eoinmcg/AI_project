import pandas as pd
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler

from llama_index.core.evaluation import RetrieverEvaluator
from llama_index.core.evaluation import EmbeddingQAFinetuneDataset
from llama_index.core.evaluation import RelevancyEvaluator, FaithfulnessEvaluator, BatchEvalRunner
from llama_index.llms.openai import OpenAI
from utils import load_db  # Assuming utils.py is in the same directory

from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

import asyncio
import nest_asyncio

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-ada-002",
    timeout=60,  # Increase timeout to 60 seconds
    max_retries=5,  # Increase max retries
    retry_delay=2  # Wait 2 seconds between retries
)
# Create a debug handler to see more info about API calls
# debug_handler = LlamaDebugHandler(print_trace_on_end=True)
# callback_manager = CallbackManager([debug_handler])

# Configure global settings
# Settings.callback_manager = callback_manager
Settings.retry_policy = {
    "max_retries": 5,
    "retry_delay": 2.0,
    "exponential_backoff": True
}
async def run_evaluation():
    load_dotenv()
    index = load_db()
    rag_eval_dataset = EmbeddingQAFinetuneDataset.from_json("./rag_eval_dataset.json")
    # Define an LLM as a judge
    llm_gpt4o = OpenAI(temperature=0, model="gpt-4o")
    llm_gpt4o_mini = OpenAI(temperature=0, model="gpt-4o-mini")
    # Initiate the faithfulnes and relevancy evaluator objects
    faithfulness_evaluator = FaithfulnessEvaluator(llm=llm_gpt4o)
    relevancy_evaluator = RelevancyEvaluator(llm=llm_gpt4o)
    # Extract the questions from the dataset
    queries = list(rag_eval_dataset.queries.values())
    # Limit to first 20 question to save time (!!remove this line in production!!)
    batch_eval_queries = queries[:5]
    # The batch evaluator runs the evaluation in batches
    runner = BatchEvalRunner(
        {"faithfulness": faithfulness_evaluator, "relevancy": relevancy_evaluator},
        workers=8,
    )
    # Define a for-loop to try different `similarity_top_k` values
    for i in [2,4,6]:
        # Set query engine with different number of returned chunks
        query_engine = index.as_query_engine(similarity_top_k=i, llm=llm_gpt4o_mini)
        # Run the evaluation
        eval_results = await runner.aevaluate_queries(query_engine, queries=batch_eval_queries)
        # Printing the results
        faithfulness_score = sum(
            result.passing for result in eval_results["faithfulness"]
        ) / len(eval_results["faithfulness"])
        print(f"top_{i} faithfulness_score: {faithfulness_score}")
        relevancy_score = sum(result.passing for result in eval_results["relevancy"]) / len(
            eval_results["relevancy"]
        )
        print(f"top_{i} relevancy_score: {relevancy_score}")
        print("="*15)

# Run the async function
if __name__ == "__main__":
    asyncio.run(run_evaluation())
