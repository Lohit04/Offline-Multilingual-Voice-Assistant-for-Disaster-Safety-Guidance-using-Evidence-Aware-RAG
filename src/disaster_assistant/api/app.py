"""Streamlit prototype UI for the disaster-assistant pipeline.

Run with: streamlit run src/disaster_assistant/api/app.py

Gives a text Q&A tab and a voice tab (st.audio_input records from the
browser's microphone -- this is also how a human can exercise the live-mic
path interactively, since that can't be driven from an automated shell).
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from disaster_assistant import config
from disaster_assistant.retrieval.hybrid import HybridRetriever
from disaster_assistant.generation.answer import answer_question
from disaster_assistant.generation.ollama_client import OllamaError
from disaster_assistant.pipeline import process_voice_question, EmptyTranscriptionError

st.set_page_config(page_title="Disaster Safety Assistant (prototype)", page_icon="⚠️")

DISCLAIMER = (
    "⚠️ **This is a research prototype providing general disaster-safety guidance from a small, "
    "fixed set of official documents. It is NOT a replacement for emergency services, official live "
    "alerts, or your local disaster management authority, and it cannot tell you about current/live "
    "conditions (shelter availability, road status, active evacuations, weather).**"
)

LANGUAGES = {
    "English": "en",
    "Hindi / हिन्दी": "hi",
    "Tamil / தமிழ்": "ta",
    "Telugu / తెలుగు": "te",
    "Kannada / ಕನ್ನಡ": "kn",
    "Malayalam / മലയാളം": "ml",
}


@st.cache_resource
def get_retriever() -> HybridRetriever:
    return HybridRetriever()


def render_answer(result, source_label: str = "Sources") -> None:
    if result.abstained:
        st.warning(result.text)
        st.caption(f"Abstained -- reason: {result.abstention_reason}")
    else:
        st.success(result.text)
        with st.expander(source_label):
            for s in result.sources:
                st.caption(f"{s.document_id}, page {s.page} (dense={s.dense_score:.3f}, bm25={s.bm25_score:.2f})")
                st.text(s.text[:400])


st.title("Disaster Safety Assistant")
st.caption("Prototype -- offline-capable multilingual voice RAG, built and tested manually (no formal evaluation yet).")
st.warning(DISCLAIMER)

tab_text, tab_voice = st.tabs(["Text", "Voice"])

try:
    retriever = get_retriever()
    index_error = None
except FileNotFoundError as e:
    retriever = None
    index_error = e

if index_error:
    st.error(f"Index files missing ({index_error}). Run `python scripts/ingest_documents.py` first.")

with tab_text:
    lang_label = st.radio("Answer language", list(LANGUAGES.keys()), horizontal=True, key="text_lang")
    question = st.text_input("Ask a disaster-safety question")
    if st.button("Ask", key="text_ask") and question and retriever:
        with st.spinner("Retrieving evidence and generating a grounded answer ..."):
            try:
                result = answer_question(retriever, question, language=LANGUAGES[lang_label])
            except OllamaError as e:
                st.error(f"Local LLM call failed ({e}). Is Ollama running?")
            else:
                render_answer(result)

with tab_voice:
    st.caption(
        "Records from your browser's microphone. Language is auto-detected from the audio "
        "(no language picker here) -- this is the live-mic path that can't be tested from an "
        "automated shell, so this is the first real interactive test of it."
    )
    audio_value = st.audio_input("Ask your question out loud")
    if audio_value is not None and retriever:
        with st.spinner("Transcribing, retrieving, answering, and synthesizing speech ..."):
            with tempfile.TemporaryDirectory() as tmp_dir:
                in_path = str(Path(tmp_dir) / "question.wav")
                out_path = str(Path(tmp_dir) / "answer.wav")
                with open(in_path, "wb") as f:
                    f.write(audio_value.getvalue())
                try:
                    result = process_voice_question(retriever, in_path, out_path)
                except EmptyTranscriptionError as e:
                    st.info(str(e))
                except OllamaError as e:
                    st.error(f"Local LLM call failed ({e}). Is Ollama running?")
                else:
                    st.caption(
                        f"Heard ({result.detected_language}, "
                        f"p={result.language_probability:.2f}): “{result.transcript}”"
                    )
                    for w in result.warnings:
                        st.caption(f"⚠️ {w}")
                    render_answer(result.answer)
                    with open(out_path, "rb") as f:
                        st.audio(f.read(), format="audio/wav")
                    t = result.timings
                    st.caption(
                        f"Latency -- ASR: {t['asr']:.1f}s, answer: {t['answer']:.1f}s, "
                        f"TTS: {t['tts']:.1f}s, total: {t['total']:.1f}s"
                    )
