import os
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

DATA_DIR = "./data"
EMBEDDINGS_MODEL = "all-MiniLM-L6-v2"  # free HuggingFace model

def load_segments(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_index(segments, model_name=EMBEDDINGS_MODEL, out_path="index.faiss"):
    model = SentenceTransformer(model_name)
    texts = [s["text"] for s in segments if s["text"].strip()]

    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, out_path)

    # Save mapping
    with open("index_mapping.json", "w", encoding="utf-8") as f:
        json.dump({"texts": texts}, f, ensure_ascii=False, indent=2)

    print(f"✅ Index built with {len(texts)} chunks")

if __name__ == "__main__":
    seg_file = os.path.join(DATA_DIR, "dQw4w9WgXcQ_segments.json")  # adjust video id if needed
    segments = load_segments(seg_file)
    build_index(segments, EMBEDDINGS_MODEL, "index.faiss")
