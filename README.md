# Gemini Chatbot (Streamlit)

A web chatbot built with **Python**, **Streamlit** and the **Google Gemini API** (default `gemini-3.5-flash-lite`, switchable in the sidebar).

## Features
- **Conversation memory:** the whole chat is sent with each request, so the bot remembers earlier messages (the latest 20 are kept to limit token use).
- **Streaming replies:** answers appear word by word instead of all at once.
- **Personas:** General assistant, Python tutor, AI/ML mentor, Interview coach (each is a different system prompt).
- **Model picker:** choose between current Gemini models, so a retired model never blocks the app.
- **Adjustable creativity:** temperature slider.
- **Secure API key handling:** read from Streamlit secrets, an environment variable, or a password box. It is never stored in the code.
- **Error handling:** clear messages for an invalid key, rate limit, server problem, network failure, or blocked/empty responses. A failed request does not corrupt the chat history.
- **Clear chat** and **download chat** (Markdown) buttons.
- **Automated tests** (pytest) for the helper logic and the app, using a fake Gemini client.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Get a free API key at https://aistudio.google.com, then either paste it in the sidebar, or copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and fill it in.

## Run the tests
```bash
pip install pytest
python -m pytest tests
```

## Project structure
```
app.py            Streamlit interface and chat flow
chat_logic.py     Helpers: history trimming, message conversion, error messages
tests/            pytest tests
```

## Deploy (Streamlit Community Cloud)
Push to GitHub, create an app at share.streamlit.io, and add `GEMINI_API_KEY` under *Secrets*.
