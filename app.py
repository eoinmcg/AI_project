import logging
import os

import gradio as gr

from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.llms import MessageRole
from llama_index.core.memory import ChatSummaryMemoryBuffer
from llama_index.core.tools import RetrieverTool, ToolMetadata
from llama_index.agent.openai import OpenAIAgent
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from llama_index.core.postprocessor import LLMRerank

from utils import create_db, load_db, load_asset
from config import CHROMA_PATH, PLACEHOLDER, TITLE, PROMPT_SYSTEM_MESSAGE, TEXT_QA_TEMPLATE


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)


API_KEY=""
token_count = 0

def create_knowledge_base_if_not_exists():
    if not os.path.exists(CHROMA_PATH) or not os.listdir(CHROMA_PATH):
        print("⚠️ ChromaDB not found. Creating DB...")
        create_db()

def get_tools():
    index = load_db()
    vector_retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=15,
        embed_model=Settings.embed_model,
        use_async=True,
    )

    # Add LLMRerank for better retrieval
    reranker = LLMRerank(
        choice_batch_size=5,
        top_n=3,
    )

    def retrieve_with_rerank(query):
        retrieved_docs = vector_retriever.retrieve(query)
        reranked_docs = reranker.postprocess(retrieved_docs)
        return reranked_docs

    tools = [
        RetrieverTool(
            # retriever=vector_retriever,
            retriever=retrieve_with_rerank,
            metadata=ToolMetadata(
                name="LitleJS_related_resources",
                description="Useful for info related to the LittleJS game development library. It gathers the info from local data.",
            ),
        )
    ]
    return tools

def set_api_key(key):
    API_KEY=key
    Settings.llm = OpenAI(temperature=0, model="gpt-4o-mini", api_key=API_KEY)
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")


def generate_completion(query, history, memory, api_key):
    logging.info(f"User query: {query}")

    if not API_KEY:
        set_api_key(api_key)

    # Manage memory
    chat_list = memory.get()
    if len(chat_list) != 0:
        user_index = [i for i, msg in enumerate(chat_list) if msg.role == MessageRole.USER]
        if len(user_index) > len(history):
            user_index_to_remove = user_index[len(history)]
            chat_list = chat_list[:user_index_to_remove]
            memory.set(chat_list)
    logging.info(f"chat_history: {len(memory.get())} {memory.get()}")
    logging.info(f"gradio_history: {len(history)} {history}")

    # Create agent
    tools = get_tools()

    agent = OpenAIAgent.from_tools(
        llm=Settings.llm,
        memory=memory,
        tools=tools,
        system_prompt=PROMPT_SYSTEM_MESSAGE,
    )

    # Generate answer
    completion = agent.stream_chat(query)
    answer_str = ""
    for token in completion.response_gen:
        answer_str += token
        global token_count
        token_count += 1  # Update token count
        yield answer_str



def launch_ui():

    js=load_asset("./assets/chat.js")

    with gr.Blocks(
        title=TITLE,
        fill_height=True,
        analytics_enabled=True,
        css=load_asset("./assets/style.css"),
        js=load_asset("./assets/chat.js"),
    ) as demo:

        api_key_input = gr.Textbox(
            label="Enter your OpenAI API Key",
            type="password",
            placeholder="sk-...",
            elem_classes="api_key_input"
        )

        memory_state = gr.State(
            lambda: ChatSummaryMemoryBuffer.from_defaults(
                token_limit=120000,
            )
        )
        chatbot = gr.Chatbot(
            scale=1,
            placeholder=PLACEHOLDER,
            type='messages',
            show_label=False,
            show_copy_button=True,
            elem_classes="chatbox",
        )

        gr.ChatInterface(
            fn=generate_completion,
            chatbot=chatbot,
            type='messages',
            additional_inputs=[memory_state, api_key_input],
        )

        token_counter = gr.Button("Tokens Used: 0", elem_classes="token_counter")

        demo.queue(default_concurrency_limit=64)
        demo.launch(debug=True, favicon_path="./assets/favicon.png", share=False) # Set share=True to share the app online


if __name__ == "__main__":
    create_knowledge_base_if_not_exists()
    launch_ui()
