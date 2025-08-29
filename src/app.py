import os
import json
import time
from datetime import datetime
from pathlib import Path

import streamlit as st
from yt_dlp import YoutubeDL
from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline

# ---------------- CONFIG ----------------
DATA_DIR = "./data"
EMBEDDINGS_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "google/flan-t5-base"  # lightweight, CPU-friendly
# ----------------------------------------

Path(DATA_DIR).mkdir(parents=True, exist_ok=True)

@st.cache_resource
def load_models():
    embed_model = SentenceTransformer(EMBEDDINGS_MODEL)
    llm = pipeline("text2text-generation", model=LLM_MODEL)
    return embed_model, llm

embed_model, llm = load_models()

# ---------------- HELPER FUNCTIONS ----------------
def download_audio(youtube_url: str, out_dir: str):
    """Download best audio and extract to MP3 using yt-dlp + ffmpeg"""
    ydl_opts = {
        "outtmpl": os.path.join(out_dir, "%(id)s.%(ext)s"),
        "format": "bestaudio/best",
        "quiet": True,
        "noplaylist": True,
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "192"}
        ],
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
        video_id = info.get("id")
        audio_mp3_path = os.path.join(out_dir, f"{video_id}.mp3")
        return audio_mp3_path, info

def transcribe_audio(audio_path: str, asr_model="small"):
    """Transcribe audio with faster-whisper"""
    model = WhisperModel(asr_model, device="cpu", compute_type="int8")
    segments_iter, info = model.transcribe(audio_path, beam_size=5)

    segments = []
    texts = []
    for seg in segments_iter:
        item = {"id": seg.id, "start": round(seg.start, 2), "end": round(seg.end, 2), "text": seg.text.strip()}
        segments.append(item)
        if item["text"]:
            texts.append(item["text"])
    return " ".join(texts), segments, info

def build_index(segments, model_name=EMBEDDINGS_MODEL):
    texts = [s["text"] for s in segments if s["text"].strip()]
    embeddings = embed_model.encode(texts, convert_to_numpy=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return index, texts

def search(index, mapping, query, top_k=5):
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    D, I = index.search(q_emb, top_k * 2)
    seen, results = set(), []
    for idx in I[0]:
        text = mapping[idx]
        if text not in seen:
            results.append(text)
            seen.add(text)
        if len(results) >= top_k:
            break
    return results

def generate_answer(query, context):
    context_text = "\n".join(context)
    prompt = f"""You are an assistant that answers questions about YouTube videos.

Context from transcript:
{context_text}

Question: {query}

Answer clearly and concisely:"""
    response = llm(prompt, max_new_tokens=200)
    return response[0]["generated_text"]

# ---------------- STREAMLIT APP ----------------
st.set_page_config(page_title="🎬 YouTube RAG Chat", page_icon="🎤", layout="wide")
st.title("🎬 YouTube RAG Chat")
st.markdown("Paste a YouTube link and chat with its transcript!")

youtube_url = st.text_input("📺 Paste YouTube URL here:")

if youtube_url:
    if st.button("🚀 Process Video"):
        with st.spinner("Downloading audio..."):
            audio_path, info = download_audio(youtube_url, DATA_DIR)

        with st.spinner("Transcribing audio..."):
            transcript, segments, asr_info = transcribe_audio(audio_path)

        with st.spinner("Building FAISS index..."):
            index, mapping = build_index(segments)

        st.success(f"✅ Processed video: {info.get('title')}")
        st.session_state["index"] = index
        st.session_state["mapping"] = mapping

# Chat UI (only if index exists)
if "index" in st.session_state:
    query = st.text_input("💬 Ask a question about the video:")
    if query:
        with st.spinner("Thinking..."):
            context = search(st.session_state["index"], st.session_state["mapping"], query)
            answer = generate_answer(query, context)

        st.subheader("🤖 Answer")
        st.write(answer)

        with st.expander("📌 Retrieved Context"):
            for c in context:
                st.markdown(f"- {c}")
