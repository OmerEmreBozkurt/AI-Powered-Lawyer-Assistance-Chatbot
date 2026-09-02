# AI-Powered Lawyer Assistance Chatbot

An intelligent **Retrieval-Augmented Generation (RAG)** chatbot that provides Turkish legal assistance by combining statutory law analysis with real court decisions. The system parses legislation documents (`.docx`) and Supreme Court rulings (`.pdf`), generates semantic embeddings via Google Gemini, and delivers grounded, citation-backed legal responses through a conversational interface.

---

## Key Features

| Feature | Description |
|---|---|
| **RAG Pipeline** | Retrieves the most relevant legal articles and court rulings before generating each response, ensuring answers are grounded in source material. |
| **Dual-Source Knowledge Base** | Ingests both statutory law (Consumer Protection Law No. 6502) and 10 real Yargıtay (Supreme Court of Appeals) rulings for comprehensive legal coverage. |
| **Semantic Search** | Uses Google Gemini `embedding-001` model to encode all legal texts into 768-dimensional vectors and performs cosine-similarity retrieval at query time. |
| **Smart Court Decision Parsing** | Extracts structured metadata (case number, decision number, date, case type, related articles) from unstructured PDF court rulings using regex-based NLP. |
| **Hybrid Retrieval Strategy** | Combines metadata-based exact matching (case numbers, dates, vehicle brands) with embedding-based semantic search for optimal recall. |
| **Conversation Memory with Chunking** | Implements a sliding-window memory system that summarizes older conversation chunks to maintain long-context coherence without exceeding token limits. |
| **Pinned Context Tracking** | Automatically detects and pins key legal references (article numbers, case numbers, court names) mentioned in the conversation for contextual continuity. |
| **Response Completeness Validation** | Multi-pass generation pipeline that validates response structure (source citations, completeness) and triggers re-generation if incomplete. |
| **Embedding Caching** | Persists computed embeddings to `kanun_embeddings.json` for fast subsequent startups, only regenerating embeddings for new or modified documents. |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     User Query                          │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
         ┌─────────────────────────┐
         │   Query Analysis        │
         │  • Regex pattern match  │
         │  • Case number detect   │
         │  • Question type class. │
         └───────────┬─────────────┘
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
 ┌─────────────────┐  ┌─────────────────┐
 │ Metadata Match  │  │ Semantic Search │
 │ (Exact: case #, │  │ (Embedding dot  │
 │  date, brand)   │  │  product, top-k)│
 └────────┬────────┘  └────────┬────────┘
          │                     │
          └──────────┬──────────┘
                     ▼
         ┌─────────────────────────┐
         │   Prompt Construction   │
         │  • Pinned context       │
         │  • Conversation summary │
         │  • Retrieved sources    │
         │  • Special instructions │
         └───────────┬─────────────┘
                     │
                     ▼
         ┌─────────────────────────┐
         │  Gemini 2.0 Flash LLM   │
         │  (Multi-pass generation │
         │   with completeness     │
         │   validation)           │
         └───────────┬─────────────┘
                     │
                     ▼
         ┌─────────────────────────┐
         │  Grounded Legal Response │
         │  with Source Citations   │
         └─────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **LLM** | Google Gemini 2.0 Flash |
| **Embeddings** | Google Gemini `embedding-001` (768-dim) |
| **Document Parsing** | `python-docx` (legislation), `pdfplumber` (court rulings) |
| **Vector Operations** | NumPy (dot-product similarity) |
| **Information Extraction** | Regex-based NLP patterns |
| **Language** | Python 3.10+ |

---

## Project Structure

```
AI-Powered-Lawyer-Assistance-Chatbot/
├── chatbot.py               # Main application — RAG chatbot engine
├── requirements.txt          # Python dependencies
├── docs/
│   ├── kanun/
│   │   └── Yasa1.docx        # Consumer Protection Law No. 6502
│   ├── yargitay/
│   │   ├── 3-hukuk-dairesi-e-2024-*.pdf   # 3rd Civil Chamber rulings
│   │   ├── 5-hukuk-dairesi-e-2024-*.pdf   # 5th Civil Chamber rulings
│   │   ├── hukuk-genel-kurulu-*.pdf       # General Assembly ruling
│   │   └── e-2024-*.pdf                   # Additional rulings
│   └── Yasa1.docx            # Backup copy
├── kanun_embeddings.json     # Cached embedding vectors (auto-generated)
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- A [Google AI Studio](https://aistudio.google.com/) API key with Gemini access

### Installation

```bash
# Clone the repository
git clone https://github.com/OmerEmreBozkurt/AI-Powered-Lawyer-Assistance-Chatbot.git
cd AI-Powered-Lawyer-Assistance-Chatbot

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```env
GEMINI_KEY=your_google_gemini_api_key_here
```

### Run

```bash
python chatbot.py
```

On first run, the system will:
1. Parse the legislation document and extract all articles
2. Process all court ruling PDFs and extract metadata
3. Generate embedding vectors for every text chunk (cached for future runs)
4. Start the interactive chat session

---

## Usage Example

```
Kanun ve Karar Chatbot'una hos geldiniz!
Çıkmak için 'quit' yazın.

Soru: Ayıplı mal aldım, garanti belgem yok. Haklarım nelerdir?

Dusunuyorum...

Yanıt:
YANIT:
Garanti belgeniz olmasa bile 6502 sayılı Tüketicinin Korunması Hakkında
Kanun kapsamında haklarınız mevcuttur...

KAYNAKLAR:
1. 6502 sayılı Kanun - Madde 11 (Seçimlik Haklar)
2. Yargıtay 3. Hukuk Dairesi, E.2024/1038, K.2025/272
```

---

## How It Works

### 1. Document Ingestion
- **Legislation (`.docx`)**: Parses structured law documents, extracting metadata (law number, publication date) and individual articles with their content and auto-generated keywords.
- **Court Rulings (`.pdf`)**: Extracts text via `pdfplumber`, then applies regex patterns to identify case numbers, decision numbers, dates, case types, summaries, and reasoning sections. Automatically distinguishes between Yargıtay and Danıştay rulings.

### 2. Embedding & Indexing
Each article and ruling is encoded into a 768-dimensional vector using Google's `embedding-001` model. Embeddings are cached in `kanun_embeddings.json` — only new or modified documents trigger re-embedding on subsequent runs.

### 3. Retrieval
When a user asks a question:
- **Metadata matching** checks for explicit references (case numbers, dates) and scores matches
- **Semantic search** computes dot-product similarity between the query embedding and all document embeddings
- Returns the top-k most relevant sources (default: 4)

### 4. Generation
A structured prompt is assembled with conversation summaries, pinned context, retrieved sources, and domain-specific instructions, then sent to Gemini 2.0 Flash. A multi-pass validation pipeline ensures responses are complete and include proper source citations.

---

## Knowledge Base Summary

| Source | Count | Description |
|---|---|---|
| **Law Articles** | ~90 | Consumer Protection Law No. 6502 — all articles |
| **Temporary Articles** | ~5 | Transitional provisions |
| **Annexes** | Variable | Supplementary schedules |
| **Court Rulings** | 10 | Yargıtay decisions (2024–2025) from 3rd & 5th Civil Chambers and General Assembly |

---
