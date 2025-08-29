import faiss
import json
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# ---------------- CONFIG ----------------
INDEX_PATH = "index.faiss"
MAPPING_PATH = "index_mapping.json"
EMBEDDINGS_MODEL = "all-MiniLM-L6-v2"

# Try heavy model, else fallback to lightweight
try:
    LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"  # gated, may fail
    print("⏳ Trying to load Mistral LLM...")
    llm = pipeline("text2text-generation", model=LLM_MODEL, device_map="auto")
except Exception as e:
    print("⚠️ Mistral not accessible. Falling back to FLAN-T5.")
    LLM_MODEL = "google/flan-t5-base"
    llm = pipeline("text2text-generation", model=LLM_MODEL, device_map="auto")
# ----------------------------------------

# Load FAISS index + mapping
index = faiss.read_index(INDEX_PATH)
with open(MAPPING_PATH, "r", encoding="utf-8") as f:
    mapping = json.load(f)["texts"]

# Load embedding model
embed_model = SentenceTransformer(EMBEDDINGS_MODEL)

def search(query, top_k=5):
    """Retrieve top-k most relevant transcript chunks from FAISS."""
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    D, I = index.search(q_emb, top_k * 2)
    seen = set()
    results = []
    for idx in I[0]:
        text = mapping[idx]
        if text not in seen:
            results.append(text)
            seen.add(text)
        if len(results) >= top_k:
            break
    return results

def generate_answer(query, context):
    """Feed retrieved context + question into LLM to get a natural answer."""
    context_text = "\n".join(context)
    prompt = f"""You are an assistant that answers questions about YouTube videos.

Context from transcript:
{context_text}

Question: {query}

Answer clearly and concisely:"""

    response = llm(prompt, max_new_tokens=200)
    return response[0]["generated_text"]

if __name__ == "__main__":
    print("🎙️ YouTube RAG Chat – Ask me about the video! (type 'exit' to quit)")
    while True:
        q = input("\nYou: ")
        if q.lower() in ["exit", "quit"]:
            break

        results = search(q)
        print("\n📌 Retrieved context:")
        for r in results:
            print(" -", r)

        print("\n🤖 Answer:")
        answer = generate_answer(q, results)
        print(answer)
