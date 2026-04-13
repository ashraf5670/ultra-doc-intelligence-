## Setup

1. Clone the repo
2. Create a `.env` file:
3. Install dependencies: AIzaSyCMZh2gz1_W0d265RM-9yE7Y-RQcBTyhn0
4. Run: python -m uvicorn app.main:app --port 8000



## Architecture

The system follows a simple Retrieval-Augmented Generation (RAG) pipeline designed for clarity and control:

1. The uploaded document is parsed into raw text
2. The text is split into smaller chunks for better retrieval
3. Each chunk is converted into vector embeddings
4. Embeddings are stored in a FAISS index (in-memory)
5. When a user asks a question:

   * The query is embedded
   * Relevant chunks are retrieved using similarity search
6. Retrieved chunks are passed to the LLM (Gemini) to generate a grounded answer

The backend is built using FastAPI and the frontend is a lightweight HTML/JS interface.

---

## Chunking Strategy

The document is split into overlapping word-based chunks:

* Chunk size: ~300 words
* Overlap: ~50–60 words

This helps in:

* Preserving context across chunk boundaries
* Improving retrieval quality for partial matches

The approach is intentionally simple and deterministic to keep the system predictable and easy to debug.

---

## Retrieval Method

* Embeddings are generated using `sentence-transformers (all-MiniLM-L6-v2)`
* All embeddings are normalized and stored in a FAISS index
* Similarity search is performed using inner product (cosine similarity equivalent after normalization)

For each query:

* Top-K relevant chunks are retrieved (default: 3–5)
* These chunks form the context for answer generation

---

## Guardrails Approach

To reduce hallucinations and ensure grounded responses, the following guardrails are implemented:

1. **Confidence Threshold Check**

   * If retrieval confidence is below a threshold → system returns:
     `"Not found in document"`

2. **LLM Instruction Constraint**

   * The model is explicitly instructed to answer only from provided context

3. **Post-response Validation**

   * If the model still returns a “not found” type response → system enforces fallback

This ensures the system avoids making unsupported claims.

---

## Confidence Scoring Method

Confidence is computed based on retrieval similarity scores:

* Uses top similarity score and average similarity of retrieved chunks

* Formula (simplified heuristic):

  `confidence = weighted combination of top score and mean score`

* Final score is normalized between 0 and 1

This score is used to:

* Trigger guardrails
* Provide transparency to the user via UI

---

## Failure Cases

Some known limitations and edge cases:

* **Scanned PDFs (no OCR):** Text extraction may fail
* **Poorly formatted documents:** Tables or structured layouts may not parse cleanly
* **Ambiguous queries:** Retrieval may return partially relevant chunks
* **Very large documents:** In-memory storage may not scale
* **LLM variability:** Extraction output may occasionally be inconsistent

---

## Improvement Ideas

If extended further, the system can be improved by:

* Adding OCR support for scanned documents
* Using semantic or adaptive chunking instead of fixed-size chunks
* Implementing hybrid retrieval (keyword + vector search)
* Persisting embeddings in a database (e.g., Pinecone, Weaviate)
* Improving confidence scoring using cross-chunk agreement
* Adding multi-document support
* Introducing caching for embeddings and queries
* Enhancing extraction using schema validation

---
