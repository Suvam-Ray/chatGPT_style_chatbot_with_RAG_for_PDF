from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from dotenv import load_dotenv
import sqlite3
import requests
import os

load_dotenv()

llm = ChatGroq(model="qwen/qwen3-32b")

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

tools = [tool for tool in [search_tool, get_stock_price, calculator] if tool is not None]
llm_with_tools = llm.bind_tools(tools)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    messages = state['messages']
    response = llm_with_tools.invoke(messages)
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
