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
st.set_page_config(page_title="🎬 YouTube RAG Chat", page_icon="🎤", layout="wide")

DATA_DIR = "./data"
EMBEDDINGS_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "google/flan-t5-base"  # lightweight, CPU-friendly
ASR_MODEL_SIZE = "small"           # faster-whisper model size
# ----------------------------------------

# ✅ Configure ffmpeg to use the bundled binary from imageio-ffmpeg (no APT needed)
FFMPEG_PATH = None
try:
    import imageio_ffmpeg
    FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()
    os.environ["FFMPEG_BINARY"] = FFMPEG_PATH  # for ffmpeg-python/pydub
except Exception as e:
    # We'll also show a warning in the UI below
    FFMPEG_PATH = None

Path(DATA_DIR).mkdir(parents=True, exist_ok=True)


# ---------------- MODEL LOAD (CACHED) ----------------
@st.cache_resource
def load_models():
    """Cache embedding model and lightweight text2text LLM."""
    embed_model = SentenceTransformer(EMBEDDINGS_MODEL)
    llm = pipeline("text2text-generation", model=LLM_MODEL)
    return embed_model, llm


@st.cache_resource
def load_whisper(asr_model: str = ASR_MODEL_SIZE):
    """Cache Faster-Whisper model so we don't re-download every run."""
    return WhisperModel(asr_model, device="cpu", compute_type="int8")


embed_model, llm = load_models()
whisper_model = load_whisper()


# ---------------- HELPER FUNCTIONS ----------------
def download_audio(youtube_url: str, out_dir: str):
    """
    Download best audio and extract to MP3 using yt-dlp + ffmpeg.
    We explicitly point yt-dlp to the bundled ffmpeg binary if available.
    """
    ydl_opts = {
        "outtmpl": os.path.join(out_dir, "%(id)s.%(ext)s"),
        "format": "bestaudio/best",
        "quiet": True,
        "noplaylist": True,
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "192"}
        ],
    }
    # If we resolved a bundled ffmpeg path, pass it to yt-dlp
    if FFMPEG_PATH:
        ydl_opts["ffmpeg_location"] = os.path.dirname(FFMPEG_PATH)

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
        video_id = info.get("id")
        audio_mp3_path = os.path.join(out_dir, f"{video_id}.mp3")
        return audio_mp3_path, info


def transcribe_audio(audio_path: str):
    """Transcribe audio with cached faster-whisper model."""
    segments_iter, info = whisper_model.transcribe(audio_path, beam_size=5)

    segments = []
    texts = []
    for seg in segments_iter:
        item = {
            "id": seg.id,
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "text": seg.text.strip(),
        }
        segments.append(item)
        if item["text"]:
            texts.append(item["text"])
    return " ".join(texts), segments, info


def build_index(segments):
    """Create FAISS index from transcript segments."""
    texts = [s["text"] for s in segments if s["text"].strip()]
    if not texts:
        return None, []

    embeddings = embed_model.encode(texts, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, texts


def search(index, mapping, query, top_k=5):
    """Retrieve top_k unique snippets for a query."""
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    D, I = index.search(q_emb, top_k * 2)
    seen, results = set(), []
    for idx in I[0]:
        if idx < 0 or idx >= len(mapping):
            continue
        text = mapping[idx]
        if text not in seen:
            results.append(text)
            seen.add(text)
        if len(results) >= top_k:
            break
    return results


def generate_answer(query, context):
    """Generate answer from LLM using retrieved context."""
    context_text = "\n".join(context)
    prompt = f"""You are an assistant that answers questions about YouTube videos.

Context from transcript:
{context_text}

Question: {query}

Answer clearly and concisely:"""
    response = llm(prompt, max_new_tokens=200)
    # handle huggingface pipeline dict formats
    if isinstance(response, list) and len(response) and "generated_text" in response[0]:
        return response[0]["generated_text"]
    if isinstance(response, list) and len(response) and isinstance(response[0], dict):
        # fallback to any text-like field
        return next(iter(response[0].values()))
    return str(response)


# ---------------- STREAMLIT APP ----------------
st.title("🎬 YouTube RAG Chat")
st.markdown("Paste a YouTube link and chat with its transcript!")

with st.expander("ℹ️ Environment checks", expanded=False):
    st.write("Python working directory:", os.getcwd())
    st.write("Data dir exists:", os.path.isdir(DATA_DIR))
    st.write("FFmpeg path resolved:", FFMPEG_PATH if FFMPEG_PATH else "❌ not set (will still try)")
    st.write("Embedding model:", EMBEDDINGS_MODEL)
    st.write("LLM model:", LLM_MODEL)
    st.write("ASR model:", ASR_MODEL_SIZE)

youtube_url = st.text_input("📺 Paste YouTube URL here:")

if youtube_url:
    if st.button("🚀 Process Video", use_container_width=True):
        try:
            with st.spinner("Downloading audio..."):
                audio_path, info = download_audio(youtube_url, DATA_DIR)

            with st.spinner("Transcribing audio..."):
                transcript, segments, asr_info = transcribe_audio(audio_path)

            with st.spinner("Building FAISS index..."):
                index, mapping = build_index(segments)

            if index is None or not mapping:
                st.error("Could not build an index from the transcript (empty text).")
            else:
                st.success(f"✅ Processed video: {info.get('title')}")
                st.session_state["index"] = index
                st.session_state["mapping"] = mapping

                with st.expander("🧾 Transcript summary"):
                    st.write(f"Segments: {len(mapping)}")
                    st.write(f"Language: {getattr(asr_info, 'language', 'unknown')}")
        except Exception as e:
            st.error(f"❌ Processing failed: {e}")

# Chat UI (only if index exists)
if "index" in st.session_state and "mapping" in st.session_state:
    query = st.text_input("💬 Ask a question about the video:")
    if query:
        with st.spinner("Thinking..."):
            context = search(st.session_state["index"], st.session_state["mapping"], query)
            if not context:
                st.warning("No relevant context found. Try another question.")
            else:
                answer = generate_answer(query, context)
                st.subheader("🤖 Answer")
                st.write(answer)

                with st.expander("📌 Retrieved Context"):
                    for c in context:
                        st.markdown(f"- {c}")
