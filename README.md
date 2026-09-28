# SYLLABUSIQ — Universal Syllabus Intelligence Platform

An intelligent, enterprise-ready syllabus extraction engine powered by FastAPI, PyMuPDF, Tesseract OCR, and NVIDIA NIM LLM (`meta/llama-3.2-11b-vision-instruct`).

---

## 🌟 Key Features

- **Universal Syllabus Detection:** Accurately detects and segments multiple courses across complex academic documents without format-specific hardcoding.
- **Unit & Topic Parsing:** Intelligent extraction of Units, Topics, Subtopics, Contact Hours, Course Objectives, Course Outcomes (with Bloom's Taxonomy levels), Textbooks, and Reference Books.
- **Smart Delimiter & Compound Word Protection:** Accurately separates multi-topic strings (hyphens, commas, semicolons, sentences) while strictly preserving technical terms (`Micro-controller`, `multi-dimensional`, `e-Health`, etc.).
- **Multi-Modal PDF & OCR Support:** PyMuPDF text extraction with automatic Tesseract OCR fallback for scanned and rasterized PDFs.
- **LLM-Powered Detail Extraction:** Powered by NVIDIA NIM API (`meta/llama-3.2-11b-vision-instruct`) with automatic rate limiting and exponential backoff retry.
- **Interactive Single-Page UI:** Modern dashboard with live page preview, side-by-side evidence matching, human-in-the-loop review, and multi-format exports (JSON, CSV, Excel).

---

## 🏗️ Architecture & Project Structure

```text
syllabus-backend/
├── app/
│   ├── __init__.py
│   ├── config.py                 # Configuration loader and environment bindings
│   ├── database.py               # Document session storage helper
│   ├── export_service.py         # Multi-format exports (JSON, Excel, CSV)
│   ├── extraction_service.py     # Universal course segmentation, table parser & regex
│   ├── main.py                   # FastAPI application & REST endpoints
│   ├── nvidia_client.py          # NVIDIA NIM LLM client with retry & rate limiting
│   ├── pdf_service.py            # PDF text and image extraction (PyMuPDF / OCR)
│   ├── prompts.py                # System prompts for course detail extraction
│   └── validation_service.py     # Schema and text fidelity validation
├── templates/
│   └── index.html                # Single-page application frontend
├── .env.example                  # Template for environment variables
├── .gitignore                    # Git ignore file
├── README.md                     # Documentation
└── requirements.txt              # Project dependencies
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+
- Tesseract OCR (optional, for scanned documents)

### 2. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/mudimukkunaveenkumarbeaiml24-blip/syllubus-extract.git
cd syllubus-extract

# Create and activate virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\Activate.ps1
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration
Create a `.env` file from the example:
```bash
cp .env.example .env
```
Fill in your NVIDIA NIM API Key in `.env`:
```ini
NVIDIA_API_KEY="your_nvidia_nim_api_key_here"
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
NVIDIA_MODEL=meta/llama-3.2-11b-vision-instruct
```

### 4. Run the Application
Start the FastAPI server with Uvicorn:
```bash
uvicorn app.main:app --reload
```
Open your browser and navigate to:
**`http://127.0.0.1:8000/`**

---

## 📡 Key API Endpoints

- `GET /health` — Check backend status and active LLM configuration.
- `POST /extract/full` — Upload a syllabus PDF to extract structured academic data.
- `GET /api/documents` — List all processed documents.
- `GET /api/documents/{doc_id}` — Get extracted details for a specific document.
- `POST /export/{doc_id}` — Download extracted data in JSON, CSV, or Excel format.
