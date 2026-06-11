import streamlit as st
from langgraph_backend import chatbot, llm
from langchain_core.messages import HumanMessage, AIMessage
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
    st.session_state['chat_threads'] = []

# Store display titles separately from internal thread UUIDs.
if 'chat_titles' not in st.session_state:
    st.session_state['chat_titles'] = {}

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
    title_col, edit_col = st.sidebar.columns([0.82, 0.18])

    if title_col.button(title, key=str(thread_id)):
        # Load the selected conversation into the main chat area.
        st.session_state['thread_id'] = thread_id
        messages = load_conversation(thread_id)

        temp_messages = []

        for msg in messages:
            # Convert LangChain message objects into Streamlit chat roles.
            if isinstance(msg, HumanMessage):
                role='user'
            else:
                role='assistant'
            temp_messages.append({'role': role, 'content': msg.content})

        st.session_state['message_history'] = temp_messages

    if title != 'New chat' and edit_col.button('✎', key=f'edit_{thread_id}'):
        edit_title_dialog(thread_id)


# **************************************** Main UI ************************************

# Replay the current conversation history on every Streamlit rerun.
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

user_input = st.chat_input('Type here')

if user_input:

    # Add and display the user's new message immediately.
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)

    # Use the current thread id so LangGraph stores messages in the right conversation.
    CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}

    # Stream only assistant tokens into the assistant chat bubble.
    with st.chat_message("assistant"):
        def ai_only_stream():
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages"
            ):
                if isinstance(message_chunk, AIMessage):
                    # Yield only assistant tokens, not user or system messages.
                    yield message_chunk.content

        ai_message = st.write_stream(ai_only_stream())

    # Save the complete assistant response after streaming finishes.
    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})

    # Generate the sidebar title once, after the first full assistant response.
    if st.session_state['chat_titles'].get(st.session_state['thread_id']) == 'New chat':
        st.session_state['chat_titles'][st.session_state['thread_id']] = generate_conversation_title(user_input, ai_message)
        st.rerun()
