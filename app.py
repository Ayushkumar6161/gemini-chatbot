"""Streamlit chatbot powered by the Google Gemini API."""
import os

import streamlit as st
from google import genai
from google.genai import types

from chat_logic import MODELS, PERSONAS, build_contents, chat_to_markdown, friendly_error

st.set_page_config(page_title="Gemini Chatbot", page_icon="💬", layout="centered")


def get_api_key():
    """Look for the key in Streamlit secrets, then the environment, then the sidebar box.
    The key is never written in the code."""
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:  # no secrets file present
        pass
    if os.getenv("GEMINI_API_KEY"):
        return os.getenv("GEMINI_API_KEY")
    return st.sidebar.text_input("Gemini API key", type="password",
                                 help="Get a free key at aistudio.google.com") or None


@st.cache_resource
def get_client(api_key):
    return genai.Client(api_key=api_key)


def stream_reply(client, model, messages, system_prompt, temperature):
    """Yield the model's answer piece by piece so it appears as it is generated."""
    stream = client.models.generate_content_stream(
        model=model,
        contents=build_contents(messages),
        config=types.GenerateContentConfig(system_instruction=system_prompt, temperature=temperature),
    )
    for chunk in stream:
        if chunk.text:
            yield chunk.text


# ---------- Sidebar ----------
st.sidebar.title("Settings")
model = st.sidebar.selectbox("Model", MODELS, help="If one model gives an error, try the other.")
persona = st.sidebar.selectbox("Persona", list(PERSONAS))
temperature = st.sidebar.slider("Creativity (temperature)", 0.0, 1.5, 0.7, 0.1)
api_key = get_api_key()

if "messages" not in st.session_state:
    st.session_state.messages = []  # this list is the chat memory

col1, col2 = st.sidebar.columns(2)
if col1.button("Clear chat", use_container_width=True):
    st.session_state.messages = []
    st.rerun()
col2.download_button("Download", chat_to_markdown(st.session_state.messages),
                     file_name="chat.md", disabled=not st.session_state.messages,
                     use_container_width=True)

# ---------- Main page ----------
st.title("💬 Gemini Chatbot")
st.caption(f"Model: {model} · Persona: {persona} · Remembers the conversation")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if not api_key:
    st.info("Enter your Gemini API key in the sidebar to start chatting.")
    st.stop()

if prompt := st.chat_input("Type your message..."):
    prompt = prompt.strip()
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                client = get_client(api_key)
                reply = st.write_stream(
                    stream_reply(client, model, st.session_state.messages, PERSONAS[persona], temperature))
                if not reply or not reply.strip():
                    raise ValueError("empty")
                st.session_state.messages.append({"role": "assistant", "content": reply})
            except ValueError:
                st.warning("The model returned no text (the answer may have been blocked). Try rephrasing.")
                st.session_state.messages.pop()  # remove the unanswered question so history stays valid
            except Exception as exc:
                st.error(friendly_error(exc, model))
                st.session_state.messages.pop()
