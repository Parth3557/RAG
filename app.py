import os
import re
import io
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

import fitz  # PyMuPDF
import numpy as np
from PIL import Image
import pytesseract

from sentence_transformers import SentenceTransformer
import faiss
from huggingface_hub import InferenceClient

# Load environment variables
load_dotenv()

app = Flask(__name__)

HF_TOKEN = os.environ.get("HF_TOKEN", "")
if not HF_TOKEN:
    print("WARNING: HF_TOKEN not found in .env.")

HF_MODEL_ID = os.environ.get("HF_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")

# ---------------------------------------------------------
# Variables config
# ---------------------------------------------------------
import glob
PDF_PATHS = glob.glob("data/*.pdf")
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 180
TOP_K = 5
MIN_RELEVANCE_SCORE = 0.15
MAX_NEW_TOKENS = 350
TEMPERATURE = 0.6
TOP_P = 0.1

OUT_OF_SCOPE_REPLY = "I don't know. I am a ML Study Assistant."

client = None
if HF_TOKEN:
    client = InferenceClient(token=HF_TOKEN)

embedder = None
index = None
final_chunks = []

# ---------------------------------------------------------
# PDF Extraction + Chunks logic
# ---------------------------------------------------------
def normalize_whitespace(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text

def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    return normalize_whitespace(text)

def ocr_image(pil_img: Image.Image) -> str:
    try:
        return clean_text(pytesseract.image_to_string(pil_img))
    except Exception:
        return ""

def pixmap_to_pil(pix: fitz.Pixmap) -> Image.Image:
    if pix.alpha:
        pix = fitz.Pixmap(pix, 0)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

@dataclass
class Chunk:
    text: str
    metadata: Dict[str, Any]

def extract_pdf_chunks(pdf_path: str) -> List[Chunk]:
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening PDF {pdf_path}: {e}")
        return []

    chunks: List[Chunk] = []
    pdf_name = os.path.basename(pdf_path)

    for page_index in range(len(doc)):
        page = doc[page_index]
        page_num = page_index + 1

        page_text = clean_text(page.get_text("text") or "")

        ocr_full_page = ""
        if len(page_text) < 80:
            try:
                mat = fitz.Matrix(2, 2)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                ocr_full_page = ocr_image(pixmap_to_pil(pix))
            except Exception:
                pass

        image_texts = []
        try:
            image_list = page.get_images(full=True)
            for img in image_list:
                xref = img[0]
                base = doc.extract_image(xref)
                img_bytes = base["image"]
                pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                txt = ocr_image(pil_img)
                if txt:
                    image_texts.append(txt)
        except Exception:
            pass

        combined = "\n".join(
            [t for t in [page_text, ocr_full_page, "\n".join(image_texts)] if t]
        ).strip()

        if combined:
            chunks.append(
                Chunk(
                    text=combined,
                    metadata={
                        "source_file": pdf_name,
                        "page": page_num,
                        "type": "page_text_ocr"
                    },
                )
            )

    return chunks

def is_good_chunk(text: str) -> bool:
    t = text.lower()
    bad_patterns = [
        "o'reilly",
        "early release",
        "unedited",
        "hands-on machine learning",
        "copyright",
        "isbn"
    ]
    if any(p in t for p in bad_patterns):
        return False
    if len(t) < 100:
        return False
    return True

def split_into_chunks(chunks: List[Chunk], chunk_size: int, chunk_overlap: int) -> List[Chunk]:
    splitted_chunks = []
    for c in chunks:
        text = c.text
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_slice = text[start:end]
            splitted_chunks.append(Chunk(text=chunk_slice, metadata=c.metadata.copy()))
            if end == len(text):
                break
            start += chunk_size - chunk_overlap
    return splitted_chunks

# ---------------------------------------------------------
# Embeddings logic
# ---------------------------------------------------------
def build_embeddings(texts: List[str], model_name: str) -> Tuple[SentenceTransformer, np.ndarray]:
    model = SentenceTransformer(model_name)
    emb = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return model, emb.astype("float32")

def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index

def init_vector_store():
    global embedder, index, final_chunks
    all_page_chunks = []

    for pdf in PDF_PATHS:
        print(f"Loading PDF: {pdf}")
        page_chunks = extract_pdf_chunks(pdf)
        all_page_chunks.extend(page_chunks)

    if not all_page_chunks:
        print("No chunks extracted. Please check the PDF.")
        return

    filtered = [c for c in all_page_chunks if is_good_chunk(c.text)]
    if not filtered:
        print("No good chunks after filtering.")
        return

    final_chunks = split_into_chunks(filtered, CHUNK_SIZE, CHUNK_OVERLAP)

    chunk_texts = [c.text for c in final_chunks]
    embedder, chunk_embeddings = build_embeddings(chunk_texts, EMBED_MODEL_NAME)
    index = build_faiss_index(chunk_embeddings)
    print(f"Vector index ready with {len(final_chunks)} chunks.")

def retrieve(question: str, top_k: int = TOP_K):
    if not embedder or not index:
        return []
    q_emb = embedder.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    scores, ids = index.search(q_emb, top_k)

    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue
        results.append(
            {
                "score": float(score),
                "text": final_chunks[idx].text,
                "metadata": final_chunks[idx].metadata,
                "chunk_id": int(idx),
            }
        )
    return results

def is_relevant(results: List[Dict[str, Any]]) -> bool:
    return bool(results) and results[0]["score"] >= MIN_RELEVANCE_SCORE

def format_context(results: List[Dict[str, Any]]) -> str:
    blocks = []
    for i, r in enumerate(results, start=1):
        md = r["metadata"]
        header = (
            f"[{i}] Source: {md.get('source_file', 'unknown')} | "
            f"Page: {md.get('page', '?')} | Score: {r['score']:.3f}"
        )
        blocks.append(header + "\n" + r["text"])
    return "\n\n".join(blocks)

# ---------------------------------------------------------
# Generation logic
# ---------------------------------------------------------
SYSTEM_PROMPT = """
You are a highly knowledgeable ML Study Assistant.

Rules:
1) You will be provided with some context from a textbook. If the context contains the answer, use it and cite the page numbers.
2) If the context does NOT contain the answer, or if the context is empty, DO NOT refuse to answer. Instead, use your own pre-trained knowledge about Machine Learning, AI, Data Science, or Programming to answer the question comprehensively.
3) If the user asks something completely unrelated to computer science, technology, or machine learning (e.g., historical wars), politely decline and say exactly:
   I don't know. I am a ML Study Assistant.
4) Keep answers clear, accurate, and structured with bullet points if helpful.
""".strip()

def generate_answer(question: str, context: str) -> str:
    if not client:
        return "Error: HF InferenceClient is not initialized. Please set HF_TOKEN."

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Question:\n{question}\n\nContext:\n{context}\n\nReturn only the answer."
        }
    ]

    response = client.chat_completion(
        model=HF_MODEL_ID,
        messages=messages,
        max_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )

    try:
        answer = response.choices[0].message.content.strip()
    except Exception:
        answer = str(response).strip()

    if not answer or len(answer) < 3:
        return OUT_OF_SCOPE_REPLY

    return answer

def answer_question(question: str) -> tuple:
    q = (question or "").strip()
    if not q:
        return "Please type a question.", []
        
    if not index:
        init_vector_store()
        if not index:
            return "Vector Store is empty or initializing. Please put a valid PDF file inside the data/ folder.", []

    results = retrieve(q, top_k=TOP_K)
    if not is_relevant(results):
        results = []

    context = format_context(results)
    
    chunks_meta = []
    for r in results[:3]:
        chunks_meta.append({
            "source": r["metadata"].get("source_file", "unknown"),
            "page": r["metadata"].get("page", "?"),
            "score": round(r["score"], 3),
            "text": r["text"]
        })

    try:
        answer = generate_answer(q, context)
    except Exception as e:
        print(f"Error calling Huggingface: {e}")
        return OUT_OF_SCOPE_REPLY, []

    # Clean up the exact refusal phrase if the model hallucinated it at the end of a valid answer
    cleaned_answer = answer.replace(OUT_OF_SCOPE_REPLY, "").strip()
    
    if not cleaned_answer or cleaned_answer.lower() == "i don't know.":
        return OUT_OF_SCOPE_REPLY, []

    return cleaned_answer, chunks_meta

# ---------------------------------------------------------
# Flask Routes
# ---------------------------------------------------------
@app.route("/")
def index_route():
    return render_template("index.html")

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json
    question = data.get("message", "")
    if not question:
        return jsonify({"error": "Empty message"}), 400

    reply, chunks = answer_question(question)
    return jsonify({"reply": reply, "chunks": chunks})

if __name__ == "__main__":
    init_vector_store()
    app.run(debug=True, host="0.0.0.0", port=5000)
