# SYLLABUSIQ — Universal Syllabus Intelligence & Reference Book RAG Engine

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/Framework-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![FAISS](https://img.shields.io/badge/Vector%20Store-FAISS-green.svg)](https://github.com/facebookresearch/faiss)
[![BM25](https://img.shields.io/badge/Lexical%20Search-Rank--BM25-orange.svg)](https://github.com/dorianbrown/rank_bm25)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

**SyllabusIQ** is an enterprise-grade academic curriculum platform and **Hybrid RAG Reference Book Topic Locator**. It combines high-speed deterministic syllabus parsing, dual-layer vector search (FAISS + BM25), monotonic textbook page resolution, and dynamic LLM validation to extract academic curriculums and pinpoint exact printed and PDF textbook page ranges for any engineering or science topic with **zero false positives**.

---

## 🌟 Key Capabilities

### 1. Hybrid RAG Reference Book Topic Locator (`Task 3`)
* **Dual Dense & Sparse Retrieval**: Combines FAISS vector embeddings (`sentence-transformers/all-MiniLM-L6-v2`) with BM25 sparse keyword ranking, applying heading boosts to textbook sections and table of contents.
* **Bounded Section Boundary Detection**: Implements forward page scanning to locate termination headings on forward pages ($\ge \text{start\_page} + 2$), capping unnumbered sections to 3–4 pages and preventing multi-chapter topic sprawling.
* **Monotonic Printed Page Mapping**: Accurately handles compound chapter-page notation (e.g. `I – 1.27` to `I – 1.45`), eliminating OCR noise (e.g. math matrices misparsed as single-digit page numbers) and mapping PDF pages directly to physical textbook pages.
* **Zero False-Positive Concept Verification**: Evaluates semantic similarity against domain-specific stems. Unrelated queries (e.g., searching "DNA sequencing" in an engineering mathematics or probability book) return `found: false` with 100% precision.
* **Typo-Resilient Search**: Query normalization and concept stem matching identify topics even when queries include typos (e.g., `"charctersitic equatiojn in mtrices"`).

### 2. Deep RAG Topic Intelligence & Pedagogical Flow
* **Authoritative Textbook Definition**: Extracts introductory textbook definitions and core significance directly from textbook anchor passages.
* **Structured Subtopic Hierarchy**: Discovers child subtopics alongside their exact printed page numbers and descriptions.
* **Key Mathematical Formulas & Theorems**: Dynamically extracts LaTeX-formatted equations and theorem statements (e.g., Characteristic Equation $|A - \lambda I| = 0$, Cayley-Hamilton Theorem, Standard Error $S.E.(\bar{x}) = \sigma/\sqrt{n}$, Gaussian PDF).
* **Pedagogical Learning Flow**: Synthesizes a structured instructional roadmap:
  $$\text{Prerequisites} \longrightarrow \text{Core Concepts} \longrightarrow \text{Worked Examples} \longrightarrow \text{Practice Problems}$$

### 3. Universal Syllabus Extraction (`Tasks 1 & 2`)
* **Multi-Course Detection**: Automatically segments multi-course academic curriculum PDFs without hardcoded university templates.
* **Structured Academic Entities**: Deterministically parses Units, Unit Titles, Topics, Subtopics, Course Objectives, Course Outcomes (COs), Bloom's Taxonomy levels, CO-PO Articulation Matrices, Textbooks, and Reference Books.
* **Smart Delimiter & Technical Term Preservation**: Separates topics accurately across commas, semicolons, and dashes while safeguarding compound technical terms (e.g., `Micro-controller`, `multi-dimensional`, `e-Health`).

### 4. Interactive Dashboard & In-Browser PDF Viewer
* **Responsive Web Dashboard**: Single-page application built with modern vanilla CSS/JS (glassmorphic styling, dark accents, responsive formula cards).
* **Direct Page-Jumping PDF Viewer**: An integrated modal PDF viewer opens directly to the exact textbook start page with zoom, pan, and page navigation controls.

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    subgraph Client ["Frontend User Interface"]
        UI["Web Dashboard (templates/index.html)"]
        PDFView["Integrated PDF Viewer Modal"]
    end

    subgraph API ["FastAPI Routing Layer (app/main.py)"]
        HealthRoute["GET /health"]
        BookRoute["GET /books"]
        LocateRoute["POST /books/locate-topic"]
        ExtractRoute["POST /extract/course"]
        DocRoute["GET /documents"]
    end

    subgraph RAG ["Hybrid RAG Engine (app/book_index_service.py)"]
        MiniLM["Dense Vectors: SentenceTransformer (all-MiniLM-L6-v2)"]
        BM25Idx["Sparse Keywords: Rank-BM25"]
        FAISSIdx["Dense Vector Store: FAISS Index"]
        PageMap["Monotonic Printed-to-PDF Page Mapper"]
        BoundsDetect["Bounded Section Scanner (_find_section_bounds)"]
    end

    subgraph Intel ["RAG Intelligence & Validation (app/topic_validation_service.py)"]
        DetRAG["Deterministic RAG Parser (Formulas, Subtopics, Quotes)"]
        LLMVal["NVIDIA NIM / GLM-5.3 Semantic Validator"]
    end

    subgraph Syllabus ["Syllabus Parser (app/extraction_service.py)"]
        CourseDet["Multi-Course Boundary Detector"]
        UnitParser["Unit, Topic, CO & Bloom Extractor"]
        MatrixParser["CO-PO Articulation Matrix Parser"]
    end

    UI -->|Locate Topic Query| LocateRoute
    LocateRoute --> FAISSIdx & BM25Idx
    FAISSIdx & BM25Idx --> BoundsDetect
    BoundsDetect --> PageMap
    PageMap --> DetRAG
    DetRAG --> LLMVal
    LLMVal --> UI
    UI -->|Open at Page| PDFView
    
    UI -->|Upload Syllabus| ExtractRoute
    ExtractRoute --> CourseDet --> UnitParser --> MatrixParser
```

---

## 📂 Project Structure

```text
syllabus-backend/
├── app/
│   ├── __init__.py
│   ├── book_index_service.py        # Hybrid RAG engine (FAISS + BM25 + Bounded Scanner)
│   ├── config.py                    # Environment bindings & path configuration
│   ├── database.py                  # Document session storage helper
│   ├── export_service.py            # Multi-format export (JSON, CSV, Excel)
│   ├── extraction_service.py        # Universal syllabus parser (Units, COs, CO-PO, Books)
│   ├── main.py                      # FastAPI routes & request orchestrator
│   ├── nvidia_client.py             # NVIDIA NIM LLM client with fail-fast network abort
│   ├── pdf_service.py               # Text and image extraction (PyMuPDF / OCR)
│   ├── prompts.py                   # Dynamic system & instruction prompts
│   ├── topic_validation_service.py  # Deep RAG intelligence, formulas, subtopics & roadmap
│   └── validation_service.py        # Schema and data fidelity validator
├── book_indexes/                    # Pre-indexed reference textbooks (FAISS, BM25, Metadata)
│   ├── 3a9395d5b235.../             # Veerarajan - Probability & Statistics
│   ├── 95e938792841.../             # BS Grewal - Higher Engineering Mathematics
│   └── b1ae098f3b41.../             # Veerarajan - Engineering Mathematics 1
├── outputs/                         # Extracted course JSON databases
├── templates/
│   └── index.html                   # Single-page interactive frontend with PDF modal
├── tests/
│   ├── test_e2e_rebuild.py          # 5-stage master verification regression suite
│   └── ...                          # Component unit tests
├── .env.example                     # Environment configuration template
├── README.md                        # Platform documentation
└── requirements.txt                 # Python dependencies
```

---

## 🚀 Getting Started

### 1. Prerequisites
* **Python**: 3.10 or 3.11 recommended.
* **Operating System**: Windows, Linux, or macOS.

### 2. Installation
Clone the repository and set up a virtual environment:

```bash
# Clone the repository
git clone https://github.com/mudimukkunaveenkumarbeaiml24-blip/syllubus-extract.git
cd syllabus-backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Linux / macOS:
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration
Create a `.env` file in the root directory:

```ini
NVIDIA_API_KEY="your_nvidia_api_key_here"
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
NVIDIA_MODEL=meta/llama-3.2-11b-vision-instruct
HF_HUB_OFFLINE=1
```

> **Note**: Setting `HF_HUB_OFFLINE=1` instructs HuggingFace and SentenceTransformers to use local cached weights, preventing unnecessary external network stalls.

### 4. Running the Application
Start the Uvicorn server:

```bash
# Windows / Linux / macOS:
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Access the application in your browser:
* **Dashboard & Topic Locator**: [http://127.0.0.1:8000/#/books](http://127.0.0.1:8000/#/books)
* **Swagger API Documentation**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
* **Service Health Check**: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

---

## 📡 API Reference

### 1. Locate Topic in Reference Books
`POST /books/locate-topic`

Locates an academic topic across indexed textbooks using Hybrid RAG and synthesizes deep topic intelligence.

**Request Form Data:**
| Field | Type | Description |
| :--- | :--- | :--- |
| `topic` | `string` | **Required**. Topic query (e.g. `"Characteristic equation in matrices"`). |
| `book_ids` | `string` | *Optional*. Comma-separated textbook IDs or hashes to search. If omitted, searches all books. |
| `validate_with_llm`| `boolean`| *Optional*. Whether to run LLM semantic enrichment (defaults to `true`). |

**Sample Response:**
```json
{
  "success": true,
  "topic": "Characteristic equation in matrices",
  "match_count": 1,
  "results": [
    {
      "book_id": "b1ae098f3b414d7ca656f5eefba6244f",
      "filename": "engineering-mathematics-1-by-t-veerarajan.pdf",
      "total_pages": 400,
      "pdf_type": "Scanned PDF",
      "book_url": "/books-static/394c8544bb554801851fc87d4e7a8362.pdf",
      "result": {
        "found": true,
        "start_page": 37,
        "end_page": 55,
        "printed_start": "I – 1.27",
        "printed_end": "I – 1.45",
        "page_display": "Page I – 1.27 – I – 1.45 (PDF 37 – 55)",
        "section_title": "1.6 EIGENVALUES AND EIGENVECTORS",
        "chapter_title": "CHAPTER 1 MATRICES",
        "match_confidence": 0.88,
        "supporting_passage": "The equation |A - λI| = 0 is called the characteristic equation of the matrix A...",
        "subtopics": [
          { "name": "Characteristic Equation", "printed_page": "I – 1.27" },
          { "name": "Eigenvalues and Spectrum", "printed_page": "I – 1.28" },
          { "name": "Properties of Eigenvalues", "printed_page": "I – 1.47" }
        ],
        "key_formulas": [
          { "name": "Characteristic Equation", "formula": "|A - \\lambda I| = 0" },
          { "name": "Sum of Eigenvalues", "formula": "\\sum \\lambda_i = \\text{tr}(A)" },
          { "name": "Product of Eigenvalues", "formula": "\\prod \\lambda_i = |A|" }
        ],
        "learning_flow": [
          { "step": "1. Prerequisites", "action": "Review matrix determinants, trace, and identity matrices." },
          { "step": "2. Core Concept", "action": "Formulate characteristic polynomial and solve characteristic roots." },
          { "step": "3. Verification", "action": "Verify using trace and determinant properties." }
        ]
      }
    }
  ]
}
```

### 2. List Indexed Reference Books
`GET /books`

Returns metadata, page count, chunk volume, and extraction format for all indexed books.

### 3. Detailed Course Syllabus Extraction
`POST /extract/course`

Extracts Units, Topics, Subtopics, Course Outcomes, Bloom Levels, CO-PO Matrix, and Textbooks for a specified course code.

### 4. Course Reference Topics
`GET /courses/{document_id}/reference-topics?course={course_code}`

Extracts all searchable reference topics from a specific syllabus course for one-click reference textbook location.

---

## 🧪 Automated Testing & Verification

Run the master 5-stage verification suite to validate system health, zero false positives, and RAG accuracy:

```bash
python tests/test_e2e_rebuild.py
```

### Verification Suite Coverage:
1. **Health & LLM Readiness**: Validates `/health` returns `200 OK` and active model configuration.
2. **Reference Inventory**: Confirms pre-indexed textbooks (Veerarajan Probability, BS Grewal, Veerarajan Mathematics 1) are online.
3. **Hybrid RAG Positive Precision**: Verifies `"Cayley-Hamilton Theorem"` pinpoints exact Section 1.7 on Page I–1.46 (PDF 56) with verbatim textbook evidence.
4. **Zero False-Positive Negative Retrieval**: Confirms completely out-of-domain queries (`"DNA sequencing and CRISPR gene editing"`) return `found: false` across all reference textbooks.
5. **Syllabus Multi-Course Extraction**: Verifies course extraction of Units, COs, CO-PO Articulation Matrices, and Reference Books.

---

## 📄 License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
