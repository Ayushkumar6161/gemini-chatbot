import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import httpx
from google.genai import errors
import streamlit as st
from streamlit.testing.v1 import AppTest

from chat_logic import MAX_HISTORY_MESSAGES, build_contents, chat_to_markdown, friendly_error, trim_history

APP = os.path.join(os.path.dirname(os.path.dirname(__file__)), "app.py")


# ---------- unit tests for chat_logic ----------
def test_build_contents_maps_roles():
    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    c = build_contents(msgs)
    assert [x.role for x in c] == ["user", "model"]
    assert c[1].parts[0].text == "hello"


def test_history_is_trimmed():
    msgs = [{"role": "user", "content": str(i)} for i in range(50)]
    assert len(trim_history(msgs)) == MAX_HISTORY_MESSAGES
    assert trim_history(msgs)[-1]["content"] == "49"


def test_friendly_error_codes():
    assert "invalid" in friendly_error(errors.ClientError(401, {"error": {}})).lower()
    assert "rate limit" in friendly_error(errors.ClientError(429, {"error": {}})).lower()
    assert "not found" in friendly_error(errors.ClientError(404, {"error": {}}), "x-model").lower()
    assert "service" in friendly_error(errors.ServerError(503, {"error": {}})).lower()
    assert "network" in friendly_error(ConnectionError()).lower()
    assert "network" in friendly_error(httpx.ConnectError("boom")).lower()
    assert "network" in friendly_error(httpx.ReadTimeout("slow")).lower()


def test_markdown_export():
    out = chat_to_markdown([{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}])
    assert "**You:** q" in out and "**Assistant:** a" in out


# ---------- app tests with a fake Gemini client ----------
def fake_client(chunks=None, exc=None):
    client = MagicMock()
    def stream(**kwargs):
        if exc:
            raise exc
        return iter(SimpleNamespace(text=t) for t in (chunks or []))
    client.models.generate_content_stream.side_effect = stream
    return client


def run_app(client, text="Hello"):
    st.cache_resource.clear()  # get_client() is cached, so reset between tests
    os.environ["GEMINI_API_KEY"] = "test-key"
    with patch("google.genai.Client", return_value=client):
        at = AppTest.from_file(APP, default_timeout=20).run()
        at.chat_input[0].set_value(text).run()
    return at


def test_app_shows_reply_and_remembers():
    at = run_app(fake_client(["Hi ", "there!"]))
    assert not at.exception
    msgs = at.session_state["messages"]
    assert msgs == [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}]


def test_app_memory_sent_on_second_turn():
    st.cache_resource.clear()
    client = fake_client(["ok"])
    os.environ["GEMINI_API_KEY"] = "test-key"
    with patch("google.genai.Client", return_value=client):
        at = AppTest.from_file(APP, default_timeout=20).run()
        at.chat_input[0].set_value("first").run()
        at.chat_input[0].set_value("second").run()
    sent = client.models.generate_content_stream.call_args.kwargs["contents"]
    assert [p.parts[0].text for p in sent] == ["first", "ok", "second"]


def test_app_api_error_is_shown_and_history_stays_valid():
    at = run_app(fake_client(exc=errors.ClientError(429, {"error": {}})))
    assert not at.exception
    assert any("rate limit" in e.value.lower() for e in at.error)
    assert at.session_state["messages"] == []


def test_app_empty_response_warns():
    at = run_app(fake_client(chunks=[]))
    assert any("no text" in w.value.lower() for w in at.warning)
    assert at.session_state["messages"] == []


def test_app_without_key_asks_for_it():
    os.environ.pop("GEMINI_API_KEY", None)
    at = AppTest.from_file(APP, default_timeout=20).run()
    assert not at.exception
    assert any("API key" in i.value for i in at.info)
    assert len(at.chat_input) == 0


def test_selected_model_is_sent_to_api():
    client = fake_client(["ok"])
    run_app(client)
    assert client.models.generate_content_stream.call_args.kwargs["model"] == "gemini-3.5-flash-lite"
