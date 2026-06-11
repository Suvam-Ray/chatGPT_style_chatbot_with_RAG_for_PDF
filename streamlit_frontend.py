import streamlit as st
from langgraph_backend import chatbot, llm, retrieve_all_threads, retrieve_all_thread_titles, save_thread_title, delete_thread
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from pydantic import BaseModel, Field
import uuid

# **************************************** utility functions *************************

def generate_thread_id():
    # Create a unique internal id for every conversation.
    thread_id = uuid.uuid4()
    return thread_id

def reset_chat():
    # Do not create multiple blank chats if the current chat is already empty.
    if not st.session_state['message_history']:
        return
    # Start a fresh conversation and clear the visible chat history.
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(st.session_state['thread_id'])
    st.session_state['message_history'] = []

def add_thread(thread_id):
    # Register a new thread and show it as "New chat" until a title is generated.
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)
        st.session_state['chat_titles'][thread_id] = 'New chat'

def load_conversation(thread_id):
    # Fetch saved LangGraph messages for the selected thread.
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    # Check if messages key exists in state values, return empty list if not
    return state.values.get('messages', [])

def build_tool_summary(tool_calls):
    # Convert raw tool calls into a compact summary with counts and params.
    tool_summary = {}
    for tool_call in tool_calls:
        tool_name = tool_call.get('name', 'tool')
        tool_args = tool_call.get('args', {})
        tool_summary.setdefault(tool_name, {'name': tool_name, 'count': 0, 'args': []})
        tool_summary[tool_name]['count'] += 1
        tool_summary[tool_name]['args'].append(tool_args)
    return list(tool_summary.values())

def render_tool_summary(tools):
    # Render the retained tool usage box for current and previous responses.
    if not tools:
        return
    label = ', '.join(f"{tool['name']} x{tool['count']}" for tool in tools)
    with st.status(f"Tools used: {label}", state="complete", expanded=False):
        for tool in tools:
            st.markdown(f"- `{tool['name']}` called {tool['count']} time(s)")
            for args in tool['args']:
                st.json(args)

def convert_messages_to_history(messages):
    # Convert LangGraph messages into Streamlit history and attach tool usage to assistant replies.
    temp_messages = []
    pending_tool_calls = []

    for msg in messages:
        if isinstance(msg, HumanMessage):
            temp_messages.append({'role': 'user', 'content': msg.content})
            pending_tool_calls = []
        elif isinstance(msg, ToolMessage):
            continue
        else:
            tool_calls = getattr(msg, 'tool_calls', [])
            if tool_calls:
                pending_tool_calls.extend(tool_calls)
            if msg.content:
                temp_messages.append({
                    'role': 'assistant',
                    'content': msg.content,
                    'tools': build_tool_summary(pending_tool_calls)
                })
                pending_tool_calls = []

    return temp_messages

def get_last_tool_summary(thread_id):
    # Read the latest saved thread state and return tools used for the last assistant response.
    messages = load_conversation(thread_id)
    history = convert_messages_to_history(messages)
    for message in reversed(history):
        if message['role'] == 'assistant':
            return message.get('tools', [])
    return []

class ConversationTitle(BaseModel):
    # Pydantic schema used to force the LLM to return a clean title field.
    title: str = Field(description="Short conversation title, maximum 5 words")

def generate_conversation_title(user_message, ai_message):
    # Ask the LLM to generate a short title from the first user and AI messages.
    prompt = f"""Generate a short conversation title, maximum 5 words.

User: {user_message}
Assistant: {ai_message}
"""
    structured_llm = llm.with_structured_output(ConversationTitle)
    response = structured_llm.invoke(prompt)
    return response.title.strip()

@st.dialog("Edit conversation title")
def edit_title_dialog(thread_id):
    # Popup form for manually renaming an existing conversation.
    new_title = st.text_input(
        "Title",
        value=st.session_state['chat_titles'].get(thread_id, 'New chat')
    )
    if st.button("OK"):
        st.session_state['chat_titles'][thread_id] = new_title
        save_thread_title(thread_id, new_title)
        st.rerun()

@st.dialog("Delete conversation")
def delete_thread_dialog(thread_id):
    # Confirm before permanently deleting a conversation from the database.
    st.warning("This is a permanent action and will delete this conversation permanently. Are you sure?")
    yes_col, no_col = st.columns(2)
    if yes_col.button("Yes"):
        delete_thread(thread_id)
        st.session_state['chat_threads'].remove(thread_id)
        st.session_state['chat_titles'].pop(thread_id, None)
        if st.session_state['thread_id'] == thread_id:
            new_chat_thread = next(
                (tid for tid in st.session_state['chat_threads'] if st.session_state['chat_titles'].get(tid) == 'New chat'),
                None
            )
            st.session_state['message_history'] = []
            st.session_state['thread_id'] = new_chat_thread or generate_thread_id()
            add_thread(st.session_state['thread_id'])
        st.rerun()
    if no_col.button("No"):
        st.rerun()


# **************************************** Session Setup ******************************
# Store the visible chat messages for the currently selected thread.
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

# Store the UUID for the currently selected thread.
if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

# Store all thread UUIDs shown in the sidebar.
if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrieve_all_threads()

# Store display titles separately from internal thread UUIDs.
if 'chat_titles' not in st.session_state:
    st.session_state['chat_titles'] = retrieve_all_thread_titles()

# Give database-loaded threads a default title if no UI title exists yet.
for thread_id in st.session_state['chat_threads']:
    st.session_state['chat_titles'].setdefault(thread_id, 'New chat')

# Ensure the current thread is listed in the sidebar.
add_thread(st.session_state['thread_id'])


# **************************************** Sidebar UI *********************************

st.sidebar.title('LangGraph Chatbot')

# Create a new empty conversation when the current one has messages.
if st.sidebar.button('New Chat'):
    reset_chat()

st.sidebar.header('My Conversations')

for thread_id in st.session_state['chat_threads'][::-1]:
    # Show newest conversations first with their display title.
    title = st.session_state['chat_titles'].get(thread_id, 'New chat')
    title_col, edit_col, delete_col = st.sidebar.columns([0.70, 0.15, 0.15])

    if title_col.button(title, key=str(thread_id)):
        # Load the selected conversation into the main chat area.
        st.session_state['thread_id'] = thread_id
        messages = load_conversation(thread_id)
        st.session_state['message_history'] = convert_messages_to_history(messages)

    if title != 'New chat' and edit_col.button('✎', key=f'edit_{thread_id}'):
        edit_title_dialog(thread_id)


    if title != 'New chat' and delete_col.button('X', key=f'delete_{thread_id}'):
        delete_thread_dialog(thread_id)


# **************************************** Main UI ************************************

# Replay the current conversation history on every Streamlit rerun.
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        if message['role'] == 'assistant':
            render_tool_summary(message.get('tools', []))
        st.markdown(message['content'])

user_input = st.chat_input('Type here')

if user_input:

    # Add and display the user's new message immediately.
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)

    # Use the current thread id so LangGraph stores messages in the right conversation.
    CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {
            "thread_id": st.session_state["thread_id"]
        },
        "run_name": "chat_turn",
    }

    # Stream only assistant tokens into the assistant chat bubble.
    with st.chat_message("assistant"):
        # Keep one status box for tool execution updates during this response.
        status_holder = {"box": None}

        def ai_only_stream():
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages"
            ):
                if isinstance(message_chunk, ToolMessage):
                    # Show or update the tool status when the graph returns a tool message.
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        # Create the status box only when a tool is actually used.
                        status_holder["box"] = st.status(f"Using `{tool_name}` ...", expanded=False)
                    else:
                        # Reuse the same status box if multiple tools run.
                        status_holder["box"].update(
                            label=f"Using `{tool_name}` ...",
                            state="running",
                            expanded=False,
                        )

                if isinstance(message_chunk, AIMessage):
                    # Yield only assistant tokens, not user or system messages.
                    yield message_chunk.content

        ai_message = st.write_stream(ai_only_stream())

        tools_used = get_last_tool_summary(st.session_state['thread_id'])

        if status_holder["box"] is not None:
            # Mark the tool status complete after the assistant response finishes.
            status_holder["box"].update(
                label="Tool finished",
                state="complete",
                expanded=False,
            )
            render_tool_summary(tools_used)

    # Save the complete assistant response after streaming finishes.
    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message, 'tools': tools_used})

    # Generate the sidebar title once, after the first full assistant response.
    if st.session_state['chat_titles'].get(st.session_state['thread_id']) == 'New chat':
        title = generate_conversation_title(user_input, ai_message)
        st.session_state['chat_titles'][st.session_state['thread_id']] = title
        save_thread_title(st.session_state['thread_id'], title)
        st.rerun()
