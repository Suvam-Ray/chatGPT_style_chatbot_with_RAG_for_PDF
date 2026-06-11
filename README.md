# ChatGPT Style Chatbot With RAG for PDF

This project builds a ChatGPT-style chatbot step by step, starting from a basic Streamlit + LangGraph chat app and evolving toward a PDF RAG chatbot with persistent conversations and tool use.

## Run the App

From the repo root, run:

```bash
streamlit run streamlit_frontend.py
```

The first version uses Groq through `ChatGroq`, so make sure your API key is available in `.env` before running.

## Environment Setup

Create and activate a virtual environment:

```bash
python -m venv myvenv
.\myvenv\Scripts\activate
```

Install the required libraries. If a `requirements.txt` file is present, use:

```bash
pip install -r requirements.txt
```

## API Keys

A sample environment file is provided:

```text
.env_sample
```

Create a copy of it, rename the copy to:

```text
.env
```

Then fill in the required keys. For the current first version, `GROQ_API_KEY` is the important one:

```text
GROQ_API_KEY="your_key_here"
```

The sample also includes placeholders for Google, OpenRouter, and LangSmith keys because later experiments may use them.

## How to Generate Keys

### Groq

1. Go to `https://console.groq.com`.
2. Sign in, for example using Google SSO.
3. Create an API key from the Groq console.
4. Copy it into `.env` as `GROQ_API_KEY`.
5. No billing setup is needed for basic free-tier experiments.

Example Groq models:

```text
openai/gpt-oss-120b
qwen/qwen3-32b
llama-3.3-70b-versatile
openai/gpt-oss-20b
llama-3.1-8b-instant
```

### Google AI Studio

1. Go to `https://aistudio.google.com/`.
2. Sign in with Gmail.
3. Create an API key.
4. Copy it into `.env` as `GOOGLE_API_KEY`.
5. Billing is not required for basic free-tier experiments.

Example Gemini models:

```text
gemini-2.5-flash-lite
gemini-2.5-flash
gemini-2.5-pro
gemini-flash-latest
```

### OpenRouter

1. Go to OpenRouter and create an API key.
2. Copy it into `.env` as `OPENROUTER_API_KEY`.
3. Without adding a credit card, the free daily request limit may be smaller.
4. With credit added, the free-model daily request limit may be higher.

OpenRouter is useful when you want access to newer free models through an OpenAI-compatible API.

Example OpenRouter models:

```text
openrouter/free
qwen/qwen3-coder-480b-a35b-instruct:free
qwen/qwen3.6-plus:free
google/gemma-3:free
nvidia/nemotron-3-super:free
arcee-ai/trinity-large-preview:free
deepseek/deepseek-v3:free
deepseek/deepseek-r1:free
nvidia/nemotron-3-nano-30b:free
```
