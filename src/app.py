# src/app.py
import os
import re
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional

import streamlit as st
import numpy as np
from yt_dlp import YoutubeDL
from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline

# ---------------- CONFIG ----------------
st.set_page_config(page_title="🎬 YouTube RAG Chat", page_icon="🎤", layout="wide")

DATA_DIR = Path("./data")
EMBEDDINGS_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "google/flan-t5-base"   # lightweight, CPU-friendly
ASR_MODEL_SIZE = "small"            # faster-whisper size: tiny/base/small/medium/large
TOP_K = 5                            # retrieved chunks
CPU_THREADS = 1                      # keep low for shared runners
# ----------------------------------------


# ✅ Configure ffmpeg to use the bundled binary from imageio-ffmpeg (no APT needed)
FFMPEG_PATH = None
try:
    import imageio_ffmpeg
    FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()
    # This env var is respected by ffmpeg-python, pydub, and generally by tools that search for ffmpeg
    os.environ["IMAGEIO_FFMPEG_EXE"] = FFMPEG_PATH
except Exception:
    FFMPEG_PATH = None

DATA_DIR.mkdir(parents=True, exist_ok=True)


# ---------------- CACHED MODELS ----------------
@st.cache_resource(show_spinner=False)
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDINGS_MODEL)

@st.cache_resource(show_spinner=False)
def get_llm():
    # Text2Text pipeline (T5-like). Keep defaults small for CPU.
    return pipeline("text2text-generation", model=LLM_MODEL)

@st.cache_resource(show_spinner=False)
def get_whisper(asr_model: str = ASR_MODEL_SIZE) -> WhisperModel:
    # CPU-friendly, int8, single thread
    return WhisperModel(asr_model, device="cpu", compute_type="int8", cpu_threads=CPU_THREADS)


embed_model = get_embedder()
llm = get_llm()
whisper_model = get_whisper()


# ---------------- UTILITIES ----------------
YTDLP_ID_PAT = re.compile(r"(?:v=|/shorts/|/live/|/v/|/embed/)([A-Za-z0-9_-]{6,})")

def extract_video_id(url: str, info: Optional[dict] = None) -> str:
    """Best effort to resolve a YouTube video id."""
    if info and "id" in info and info["id"]:
        return info["id"]
    m = YTDLP_ID_PAT.search(url)
    if m:
        return m.group(1)
    # fallback to a hash if we truly cannot parse (rare)
    return str(abs(hash(url)))

def video_artifact_dir(video_id: str) -> Path:
    d = DATA_DIR / video_id
    d.mkdir(parents=True, exist_ok=True)
    return d

def transcript_json_path(video_id: str) -> Path:
    return video_artifact_dir(video_id) / "transcript.json"

def transcript_txt_path(video_id: str) -> Path:
    return video_artifact_dir(video_id) / "transcript.txt"

def index_path(video_id: str) -> Path:
    return video_artifact_dir(video_id) / "index.faiss"

def mapping_path(video_id: str) -> Path:
    return video_artifact_dir(video_id) / "mapping.json"

def emb_path(video_id: str) -> Path:
    return video_artifact_dir(video_id) / "embeddings.npy"

def safe_write_json(p: Path, obj: dict):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def safe_read_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


# ---------------- CORE OPS ----------------
def download_audio(youtube_url: str, out_dir: Path) -> Tuple[Path, dict, str]:
    """
    Download best audio and extract to MP3 using yt-dlp + ffmpeg.
    We explicitly point yt-dlp to the bundled ffmpeg binary if available.
    """
    ydl_opts = {
        "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
        "format": "bestaudio/best",
        "quiet": True,
        "noplaylist": True,
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "192"}
        ],
    }
    if FFMPEG_PATH:
        # yt-dlp uses the directory for ffmpeg_location
        ydl_opts["ffmpeg_location"] = os.path.dirname(FFMPEG_PATH)

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
        video_id = extract_video_id(youtube_url, info)
        audio_mp3_path = out_dir / f"{video_id}.mp3"
        return audio_mp3_path, info, video_id


def transcribe_audio(audio_path: Path) -> Tuple[str, List[dict], object]:
    """Transcribe audio with cached faster-whisper model -> (full_text, segments[], asr_info)."""
    segments_iter, info = whisper_model.transcribe(str(audio_path), beam_size=5)

    segments: List[dict] = []
    texts: List[str] = []
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


def build_or_load_index(video_id: str, segments: Optional[List[dict]] = None):
    """
    Build FAISS index for a video's transcript, or load if it already exists.
    Returns (faiss_index, mapping_texts).
    """
    ip = index_path(video_id)
    mp = mapping_path(video_id)
    ep = emb_path(video_id)

    # Prefer loading existing index + mapping if present
    if ip.exists() and mp.exists() and ep.exists():
        try:
            idx = faiss.read_index(str(ip))
            mapping = safe_read_json(mp)
            # NOTE: We don't need to load embeddings to search with FAISS,
            # but we keep them saved to allow potential future dimensionality checks.
            return idx, mapping
        except Exception:
            pass  # fall back to rebuild

    if segments is None:
        # No text to embed
        return None, []

    mapping = [s["text"] for s in segments if s["text"].strip()]
    if not mapping:
        return None, []

    emb = embed_model.encode(mapping, convert_to_numpy=True, batch_size=64, show_progress_bar=False)
    dim = emb.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(emb.astype(np.float32))

    # persist
    faiss.write_index(index, str(ip))
    safe_write_json(mp, mapping)
    np.save(str(ep), emb.astype(np.float32))
    return index, mapping


def load_index_if_exists(video_id: str):
    ip = index_path(video_id)
    mp = mapping_path(video_id)
    if ip.exists() and mp.exists():
        try:
            idx = faiss.read_index(str(ip))
            mapping = safe_read_json(mp)
            return idx, mapping
        except Exception:
            return None, []
    return None, []


def save_transcript(video_id: str, full_text: str, segments: List[dict], info: dict, asr_info: object):
    tjson = {
        "video_id": video_id,
        "title": info.get("title"),
        "uploader": info.get("uploader"),
        "duration": info.get("duration"),
        "webpage_url": info.get("webpage_url"),
        "fetched_at": datetime.utcnow().isoformat() + "Z",
        "asr": {
            "language": getattr(asr_info, "language", None),
            "duration": getattr(asr_info, "duration", None),
        },
        "full_text": full_text,
        "segments": segments,
    }
    safe_write_json(transcript_json_path(video_id), tjson)
    transcript_txt_path(video_id).write_text(full_text, encoding="utf-8")


def load_transcript_if_exists(video_id: str):
    tj = transcript_json_path(video_id)
    if tj.exists():
        try:
            meta = safe_read_json(tj)
            full_text = meta.get("full_text", "")
            segments = meta.get("segments", [])
            return full_text, segments, meta
        except Exception:
            return "", [], {}
    return "", [], {}


def retrieve(index, mapping: List[str], query: str, top_k: int = TOP_K) -> List[str]:
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    D, I = index.search(q_emb.astype(np.float32), top_k * 2)
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


def answer_with_context(query: str, context: List[str]) -> str:
    context_text = "\n".join(context)
    prompt = (
        "You are an assistant that answers questions about a YouTube video's content.\n"
        "Use ONLY the context. If the answer isn't in the context, say you don't know.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {query}\nAnswer:"
    )
    out = llm(prompt, max_new_tokens=256)
    # huggingface pipeline returns a list of dicts; flan-t5 uses 'generated_text'
    if isinstance(out, list) and len(out):
        item = out[0]
        if isinstance(item, dict):
            if "generated_text" in item:
                return item["generated_text"]
            # fallback to any text-like field
            for v in item.values():
                if isinstance(v, str) and v.strip():
                    return v
    return str(out)


# ---------------- STREAMLIT UI ----------------
st.title("🎬 YouTube RAG Chat")
st.markdown("Paste a YouTube link, process the transcript, then ask questions about the video.")

with st.expander("ℹ️ Environment checks", expanded=False):
    st.write("Working directory:", os.getcwd())
    st.write("Data dir:", str(DATA_DIR.resolve()))
    st.write("FFmpeg resolved:", FFMPEG_PATH if FFMPEG_PATH else "❌ not set (will still try)")
    st.write("Embedding model:", EMBEDDINGS_MODEL)
    st.write("LLM model:", LLM_MODEL)
    st.write("ASR model:", ASR_MODEL_SIZE)
    st.write("CPU threads for ASR:", CPU_THREADS)

youtube_url = st.text_input("📺 Paste YouTube URL here:", placeholder="https://www.youtube.com/watch?v=...")

col_a, col_b = st.columns([1, 1])
process_clicked = col_a.button("🚀 Process Video", use_container_width=True)
clear_clicked = col_b.button("🧹 Clear State", use_container_width=True)

if clear_clicked:
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.rerun()

if youtube_url and process_clicked:
    try:
        with st.spinner("Downloading audio..."):
            audio_path, info, video_id = download_audio(youtube_url, DATA_DIR)

        # Attempt to reuse prior transcript if exists
        full_text, segments, meta = load_transcript_if_exists(video_id)
        if not segments:
            with st.spinner("Transcribing audio (Whisper)…"):
                full_text, segments, asr_info = transcribe_audio(audio_path)
                save_transcript(video_id, full_text, segments, info, asr_info)
        else:
            asr_info = meta.get("asr", {})

        # Try to load FAISS; otherwise build it
        idx, mapping = load_index_if_exists(video_id)
        if idx is None or not mapping:
            with st.spinner("Building FAISS index…"):
                idx, mapping = build_or_load_index(video_id, segments)

        if idx is None or not mapping:
            st.error("Could not build an index from the transcript (no usable text).")
        else:
            st.success(f"✅ Processed: {info.get('title', video_id)}")
            st.session_state["video_id"] = video_id
            st.session_state["index"] = idx
            st.session_state["mapping"] = mapping
            st.session_state["title"] = info.get("title", "")
            st.session_state["url"] = info.get("webpage_url", youtube_url)

            with st.expander("🧾 Transcript summary"):
                st.write(f"Segments: {len(mapping)}")
                lang = asr_info.get("language") if isinstance(asr_info, dict) else getattr(asr_info, "language", "unknown")
                st.write(f"Language: {lang}")
                st.download_button(
                    "Download transcript (.txt)",
                    data=full_text,
                    file_name=f"{video_id}.txt",
                    mime="text/plain"
                )

    except Exception as e:
        st.error(f"❌ Processing failed: {e}")

# ---------------- CHAT ----------------
if "index" in st.session_state and "mapping" in st.session_state:
    st.divider()
    st.subheader("💬 Ask about the video")
    query = st.text_input("Your question:", placeholder="What is the main argument of the presenter?")
    if query:
        with st.spinner("Thinking…"):
            ctx = retrieve(st.session_state["index"], st.session_state["mapping"], query, top_k=TOP_K)
            if not ctx:
                st.warning("No relevant context found. Try another question.")
            else:
                answer = answer_with_context(query, ctx)
                st.markdown("### 🤖 Answer")
                st.write(answer)

                with st.expander("📌 Retrieved Context"):
                    for c in ctx:
                        st.markdown(f"- {c}")
