# 🧠 ML Study Assistant — RAG-Powered Chatbot

A **Retrieval-Augmented Generation (RAG)** chatbot built with Flask that lets you upload Machine Learning textbooks (PDFs) and ask questions about them. The system extracts text and images from PDFs using OCR, builds a semantic vector index with FAISS, retrieves the most relevant passages, and generates accurate answers using Hugging Face's hosted LLMs.

> **Ask anything about Machine Learning** — the assistant answers using your own study material as context, with page-number citations.

---

## ✨ Features

- **📄 PDF Ingestion & OCR** — Extracts text from PDFs using PyMuPDF, with Tesseract OCR fallback for scanned/image-heavy pages.
- **🔍 Semantic Search** — Embeds document chunks using `all-MiniLM-L6-v2` sentence-transformers and indexes them in a FAISS vector store for fast similarity search.
- **🤖 LLM-Powered Answers** — Generates responses via Hugging Face Inference API (`Qwen/Qwen2.5-7B-Instruct` by default), grounded in retrieved context.
- **📑 Source Citations** — Shows the top retrieved chunks with source file, page number, and relevance score for transparency.
- **🌙 Modern Dark UI** — Sleek, dark-themed chat interface built with Tailwind CSS and Material Symbols, featuring typing indicators and hover animations.
- **⚡ Lazy Initialization** — Vector store is built on first query, so the server starts instantly.

---

## 🏗️ Architecture

```
User Question
     │
     ▼
┌──────────────┐     ┌─────────────────────┐     ┌──────────────────┐
│  Flask API   │────▶│  FAISS Vector Store  │────▶│  Top-K Retrieval │
│  /api/chat   │     │  (all-MiniLM-L6-v2) │     │  (cosine sim)    │
└──────────────┘     └─────────────────────┘     └────────┬─────────┘
                                                          │
                                                          ▼
                                                 ┌────────────────────┐
                                                 │  HF Inference API  │
                                                 │  (Qwen2.5-7B)     │
                                                 └────────┬───────────┘
                                                          │
                                                          ▼
                                                   Final Answer
                                                 + Source Chunks
```

---

## 🛠️ Tech Stack

| Layer          | Technology                                      |
| -------------- | ----------------------------------------------- |
| **Backend**    | Python, Flask                                   |
| **PDF Parsing**| PyMuPDF (fitz), Tesseract OCR, Pillow           |
| **Embeddings** | Sentence-Transformers (`all-MiniLM-L6-v2`)      |
| **Vector DB**  | FAISS (Facebook AI Similarity Search)            |
| **LLM**        | Hugging Face Inference API (Qwen2.5-7B-Instruct)|
| **Frontend**   | HTML, Tailwind CSS, JavaScript                   |
| **Fonts/Icons**| Google Fonts (Inter), Material Symbols           |

---

## 📦 Project Structure

```
Project/
├── app.py                  # Flask backend — RAG pipeline + API routes
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (HF_TOKEN, HF_MODEL_ID)
├── data/
│   └── *.pdf               # Place your ML textbook PDFs here
├── templates/
│   └── index.html          # Chat UI frontend
├── context.txt             # Additional context notes
└── README.md               # This file
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+**
- **Tesseract OCR** — [Install Guide](https://github.com/tesseract-ocr/tesseract#installing-tesseract)
- **Hugging Face API Token** — [Get one here](https://huggingface.co/settings/tokens)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/ml-study-assistant.git
   cd ml-study-assistant
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv

   # Windows
   .\venv\Scripts\activate

   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**

   Create a `.env` file in the project root:
   ```env
   HF_TOKEN=your_huggingface_token_here
   HF_MODEL_ID=Qwen/Qwen2.5-7B-Instruct
   ```

5. **Add your PDFs**

   Place any ML/AI textbook PDFs into the `data/` folder.

6. **Run the application**
   ```bash
   python app.py
   ```

7. **Open in browser**

   Navigate to [http://localhost:5000](http://localhost:5000)

---

## ⚙️ Configuration

These parameters can be tuned in `app.py`:

| Parameter            | Default | Description                                  |
| -------------------- | ------- | -------------------------------------------- |
| `EMBED_MODEL_NAME`   | `all-MiniLM-L6-v2` | Sentence-Transformer model for embeddings |
| `CHUNK_SIZE`         | `1200`  | Max characters per text chunk                |
| `CHUNK_OVERLAP`      | `180`   | Overlap between consecutive chunks           |
| `TOP_K`              | `5`     | Number of chunks to retrieve per query       |
| `MIN_RELEVANCE_SCORE`| `0.15`  | Minimum cosine similarity threshold          |
| `MAX_NEW_TOKENS`     | `350`   | Max tokens in the generated response         |
| `TEMPERATURE`        | `0.6`   | LLM sampling temperature                     |

---

## 📡 API Endpoints

| Method | Endpoint     | Description                 | Body                          |
| ------ | ------------ | --------------------------- | ----------------------------- |
| `GET`  | `/`          | Serves the chat UI          | —                             |
| `POST` | `/api/chat`  | Send a question, get answer | `{ "message": "your query" }` |

**Response format:**
```json
{
  "reply": "The answer based on retrieved context...",
  "chunks": [
    {
      "source": "textbook.pdf",
      "page": 42,
      "score": 0.847,
      "text": "Retrieved passage text..."
    }
  ]
}
```

---

## 📝 License

This project is for **educational purposes only**. The PDFs in the `data/` folder are not included in this repository.

---

## Acknowledgements

- [Hugging Face](https://huggingface.co/) — Inference API & Sentence-Transformers
- [FAISS](https://github.com/facebookresearch/faiss) — Vector similarity search
- [PyMuPDF](https://pymupdf.readthedocs.io/) — PDF text extraction
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) — Optical character recognition
