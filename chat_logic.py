"""Pure helper functions for the chatbot (no Streamlit code here, so they are easy to test)."""
import httpx
from google.genai import errors, types

# Model names change over time. Check https://ai.google.dev/gemini-api/docs/models for the current list.
MODELS = ["gemini-3.5-flash-lite", "gemini-3.8-flash"]
MODEL_NAME = MODELS[0]  # default: the fast, light model
MAX_HISTORY_MESSAGES = 20  # only the latest messages are sent to the model, to limit token use

PERSONAS = {
    "General assistant": "You are a helpful, friendly assistant. Answer clearly and concisely.",
    "Python tutor": (
        "You are a patient Python tutor for beginners. Explain concepts step by step, "
        "give short code examples, and end with one small practice exercise."
    ),
    "AI/ML mentor": (
        "You are an AI/ML mentor for students. Explain machine learning ideas in simple language, "
        "use small examples, and mention common mistakes to avoid."
    ),
    "Interview coach": (
        "You are a technical interview coach for internship candidates. Ask one question at a time, "
        "wait for the answer, then give short, honest feedback."
    ),
}


def trim_history(messages, limit=MAX_HISTORY_MESSAGES):
    """Keep only the most recent `limit` messages."""
    return messages[-limit:]


def build_contents(messages):
    """Convert our chat messages ({'role': 'user'|'assistant', 'content': str})
    into the Content objects the Gemini API expects (roles are 'user' and 'model')."""
    contents = []
    for m in trim_history(messages):
        role = "user" if m["role"] == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part(text=m["content"])]))
    return contents


def friendly_error(exc, model=MODEL_NAME):
    """Turn an exception into a short message a user can understand."""
    if isinstance(exc, errors.APIError):
        code = getattr(exc, "code", None)
        if code in (400, 401, 403):
            return "The API key looks invalid or has no permission. Please check your Gemini API key."
        if code == 404:
            return f"The model '{model}' was not found or is not available for your key. Pick another model in the sidebar."
        if code == 429:
            return "Rate limit or quota reached. Please wait a minute and try again."
        if code is not None and code >= 500:
            return "Google's Gemini service had a problem. Please try again in a moment."
        return f"The Gemini API returned an error (code {code})."
    if isinstance(exc, (httpx.TransportError, ConnectionError, TimeoutError, OSError)):
        return "Network problem. Please check your internet connection and try again."
    return f"Something went wrong: {type(exc).__name__}."


def chat_to_markdown(messages):
    """Format the conversation as markdown for the download button."""
    lines = ["# Chat export", ""]
    for m in messages:
        who = "You" if m["role"] == "user" else "Assistant"
        lines += [f"**{who}:** {m['content']}", ""]
    return "\n".join(lines)
