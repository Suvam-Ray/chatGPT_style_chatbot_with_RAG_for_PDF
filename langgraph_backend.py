import os
# HuggingFace embeddings (via torch/MKL) pull in Intel Fortran runtime which
# hijacks Ctrl+C and crashes Streamlit on shutdown. Disabling with following.
os.environ["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] = "1"

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Optional

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import sqlite3
import requests
import tempfile
import shutil

load_dotenv()

llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
embeddings = None

VECTOR_STORE_DIR = os.path.join("data", "vector_stores")
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

def _thread_key(thread_id: str) -> str:
    return str(thread_id)

def _vector_store_path(thread_id: str) -> str:
    return os.path.join(VECTOR_STORE_DIR, _thread_key(thread_id))

def _collection_name(thread_id: str) -> str:
    safe_thread_id = _thread_key(thread_id).replace("-", "_")
    return f"thread_{safe_thread_id}"

def _get_embeddings():
    global embeddings
    if embeddings is None:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-miniLM-L6-v2"
        )
    return embeddings

def _get_vector_store(thread_id: Optional[str]):
    if not thread_id:
        return None

    thread_id = _thread_key(thread_id)
    store_path = _vector_store_path(thread_id)
    if not os.path.exists(store_path):
        return None

    return Chroma(
        collection_name=_collection_name(thread_id),
        embedding_function=_get_embeddings(),
        persist_directory=store_path,
    )

def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)

        source_filename = filename or os.path.basename(temp_path)
        for chunk in chunks:
            chunk.metadata["filename"] = source_filename
            chunk.metadata["thread_id"] = _thread_key(thread_id)

        store_path = _vector_store_path(thread_id)
        os.makedirs(store_path, exist_ok=True)
        vector_store = Chroma(
            collection_name=_collection_name(thread_id),
            embedding_function=_get_embeddings(),
            persist_directory=store_path,
        )
        vector_store.add_documents(chunks)

        return {
            "filename": source_filename,
            "documents": len(docs),
            "chunks": len(chunks),
            "vector_store_path": store_path,
        }
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass

try:
    search_tool = DuckDuckGoSearchRun(region="us-en")
except ImportError:
    search_tool = None

@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """Perform a basic arithmetic operation: add, sub, mul, div."""
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}

        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}

@tool
def get_stock_price(symbol: str) -> dict:
    """Fetch latest stock price for a given symbol, e.g. AAPL or TSLA."""
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    r = requests.get(url)
    return r.json()

@tool
def rag_tool(query: str, thread_id: Optional[str] = None) -> dict:
    """Retrieve relevant information from the uploaded PDF for this chat thread."""
    vector_store = _get_vector_store(thread_id)
    if vector_store is None:
        return {"error": "No document indexed for this chat. Upload a PDF first.", "query": query}

    result = vector_store.similarity_search(query, k=10)
    return {
        "query": query,
        "context": [doc.page_content for doc in result],
        "metadata": [doc.metadata for doc in result],
    }

tools = [tool for tool in [search_tool, get_stock_price, calculator, rag_tool] if tool is not None]
llm_with_tools = llm.bind_tools(tools)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState, config=None):
    thread_id = None
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")

    system_message = SystemMessage(
        content=(
            "You are a helpful assistant. If the user asks about an uploaded PDF, "
            "uploaded document, file, attachment, or says 'this' after uploading a file, "
            "you must call `rag_tool` before answering. For summaries of uploaded files, "
            "you must call `rag_tool`. Include the thread_id "
            f"`{thread_id}` when calling `rag_tool`. You can also use web search, stock "
            "price, and calculator tools when helpful. If no document is available, ask "
            "the user to upload a PDF."
        )
    )
    messages = [system_message, *state['messages']]
    response = llm_with_tools.invoke(messages)
    if thread_id and getattr(response, "tool_calls", None):
        for tool_call in response.tool_calls:
            if tool_call.get("name") == "rag_tool":
                tool_call.setdefault("args", {})["thread_id"] = str(thread_id)
    return {"messages": [response]}

tool_node = ToolNode(tools)

DB_DIR = "data"
DB_PATH = os.path.join(DB_DIR, "chatbot.db")
os.makedirs(DB_DIR, exist_ok=True)

conn = sqlite3.connect(database=DB_PATH, check_same_thread=False)
conn.execute("""
CREATE TABLE IF NOT EXISTS thread_titles (
    thread_id TEXT PRIMARY KEY,
    title TEXT NOT NULL
)
""")
conn.commit()
# Checkpointer
checkpointer = SqliteSaver(conn=conn)
checkpointer.setup()

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)
graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)

def retrieve_all_threads():
    rows = conn.execute("""
        SELECT thread_id
        FROM checkpoints
        GROUP BY thread_id
        ORDER BY MAX(checkpoint_id)
    """).fetchall()
    return [row[0] for row in rows]

def save_thread_title(thread_id, title):
    conn.execute(
        """
        INSERT INTO thread_titles (thread_id, title)
        VALUES (?, ?)
        ON CONFLICT(thread_id) DO UPDATE SET title = excluded.title
        """,
        (str(thread_id), title)
    )
    conn.commit()

def retrieve_all_thread_titles():
    rows = conn.execute("SELECT thread_id, title FROM thread_titles").fetchall()
    return {thread_id: title for thread_id, title in rows}

def delete_thread(thread_id):
    thread_id = str(thread_id)
    conn.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))
    conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
    conn.execute("DELETE FROM thread_titles WHERE thread_id = ?", (thread_id,))
    conn.commit()
    vector_store_path = _vector_store_path(thread_id)
    if os.path.exists(vector_store_path):
        shutil.rmtree(vector_store_path)

def thread_has_document(thread_id: str) -> bool:
    return os.path.exists(_vector_store_path(thread_id))
