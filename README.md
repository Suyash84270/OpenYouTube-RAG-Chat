# StreamTube-RAG

🎬 **Chat with any YouTube video** — paste a link and this app will:
1) download audio (yt-dlp)  
2) transcribe with Whisper (faster-whisper)  
3) embed transcript with `sentence-transformers` + **FAISS**  
4) answer questions using **Flan-T5** (CPU-friendly, free)

Built with **Streamlit** so you can deploy and share easily.

---

## ✨ Features
- Paste a YouTube URL → full pipeline runs inside the app
- Whisper (via `faster-whisper`) for accurate, fast ASR
- `all-MiniLM-L6-v2` embeddings + FAISS vector search
- Hugging Face `google/flan-t5-base` for concise answers
- 100% local + free (no API keys needed)

---

## 🧱 Tech Stack
- Streamlit, yt-dlp, ffmpeg
- faster-whisper (Whisper), pydub
- sentence-transformers, FAISS
- transformers (Flan-T5), torch
- Python 3.11

---

## 🖥️ Run locally

```bash
# 1) Create & activate venv (Windows PowerShell)
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

# 2) Install Python deps
pip install -r requirements.txt

# 3) Run the app
streamlit run src/app.py
