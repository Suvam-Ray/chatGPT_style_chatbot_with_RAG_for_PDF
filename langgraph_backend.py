from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import sqlite3
import os

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile")

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    messages = state['messages']
    response = llm.invoke(messages)
    return {"messages": [response]}

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

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])

    return list(all_threads)

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
