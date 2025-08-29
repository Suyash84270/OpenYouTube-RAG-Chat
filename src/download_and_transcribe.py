import os
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from yt_dlp import YoutubeDL
from faster_whisper import WhisperModel


def load_config():
    load_dotenv()
    cfg = {
        "DATA_DIR": os.getenv("DATA_DIR", "./data"),
        "ASR_MODEL": os.getenv("ASR_MODEL", "small"),
        "CHUNK_SIZE": int(os.getenv("CHUNK_SIZE", "1200")),
        "CHUNK_OVERLAP": int(os.getenv("CHUNK_OVERLAP", "200")),
    }
    Path(cfg["DATA_DIR"]).mkdir(parents=True, exist_ok=True)
    return cfg


def download_audio(youtube_url: str, out_dir: str):
    """
    Downloads best audio and extracts to MP3 using ffmpeg via yt-dlp.
    Returns (audio_mp3_path, info_dict).
    """
    ydl_opts = {
        "outtmpl": os.path.join(out_dir, "%(id)s.%(ext)s"),
        "format": "bestaudio/best",
        "noplaylist": True,
        "quiet": False,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
    }

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
        video_id = info.get("id")
        audio_mp3_path = os.path.join(out_dir, f"{video_id}.mp3")
        if not os.path.exists(audio_mp3_path):
            # fallback: try to find a file with same id
            candidates = list(Path(out_dir).glob(f"{video_id}.*"))
            if candidates:
                audio_mp3_path = str(candidates[0])
        return audio_mp3_path, info


def transcribe_audio(audio_path: str, asr_model: str):
    """
    Transcribe using faster-whisper on CPU with int8 compute.
    Returns (full_text, segments_list, info_dict).
    """
    model = WhisperModel(asr_model, device="cpu", compute_type="int8")
    segments_iter, info = model.transcribe(
        audio_path,
        beam_size=5,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        condition_on_previous_text=True,
    )

    segments = []
    full_text_chunks = []
    for seg in segments_iter:
        item = {
            "id": seg.id,
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "text": seg.text.strip(),
            "avg_logprob": seg.avg_logprob,
            "compression_ratio": seg.compression_ratio,
            "no_speech_prob": seg.no_speech_prob,
        }
        segments.append(item)
        if item["text"]:
            full_text_chunks.append(item["text"])

    full_text = " ".join(full_text_chunks).strip()
    meta = {
        "language": info.language,
        "duration": info.duration,
        "asr_model": asr_model,
        "transcribed_at": datetime.utcnow().isoformat() + "Z",
    }
    return full_text, segments, meta


def save_outputs(base_path_no_ext: str, transcript: str, segments: list, video_info: dict, asr_meta: dict):
    txt_path = base_path_no_ext + "_transcript.txt"
    seg_path = base_path_no_ext + "_segments.json"
    meta_path = base_path_no_ext + "_meta.json"

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(transcript)

    with open(seg_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)

    kept = {
        "id": video_info.get("id"),
        "title": video_info.get("title"),
        "webpage_url": video_info.get("webpage_url"),
        "uploader": video_info.get("uploader"),
        "upload_date": video_info.get("upload_date"),
        "duration": video_info.get("duration"),
        "channel_id": video_info.get("channel_id"),
        "channel": video_info.get("channel"),
        "thumbnails": video_info.get("thumbnails"),
    }
    meta = {
        "video": kept,
        "asr": asr_meta,
        "saved_at": datetime.utcnow().isoformat() + "Z",
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return txt_path, seg_path, meta_path


def main():
    if len(sys.argv) < 2:
        print("Usage: python src/download_and_transcribe.py <YOUTUBE_URL>")
        sys.exit(1)

    youtube_url = sys.argv[1].strip()
    cfg = load_config()

    print(f"[1/3] Downloading audio from: {youtube_url}")
    try:
        audio_path, info = download_audio(youtube_url, cfg["DATA_DIR"])
    except Exception as e:
        print("❌ Error during download. Ensure ffmpeg is installed (winget install Gyan.FFmpeg).")
        raise

    if not os.path.exists(audio_path):
        print(f"❌ Audio file not found after download: {audio_path}")
        sys.exit(2)

    video_id = info.get("id", f"vid_{int(time.time())}")
    base_no_ext = os.path.join(cfg["DATA_DIR"], video_id)

    print(f"[2/3] Transcribing with faster-whisper model: {cfg['ASR_MODEL']}")
    transcript, segments, asr_meta = transcribe_audio(audio_path, cfg["ASR_MODEL"])

    print(f"[3/3] Saving outputs to data/ …")
    txt_path, seg_path, meta_path = save_outputs(base_no_ext, transcript, segments, info, asr_meta)

    print("\n✅ Done!")
    print(f"Audio:      {audio_path}")
    print(f"Transcript: {txt_path}")
    print(f"Segments:   {seg_path}")
    print(f"Metadata:   {meta_path}")


if __name__ == "__main__":
    main()
