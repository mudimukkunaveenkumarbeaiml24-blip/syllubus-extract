# app/book_index_service.py
"""
SyllabusIQ — Reference Book Topic Locator
=========================================

Page-aware, persistent, hybrid-RAG index for reference books.
Works on any PDF (text-native or scanned), any chapter numbering
scheme, any topic.

Pipeline
--------
PDF → extract_pdf() → LaTeX stripping → page-aware chunking
    → heading detection (single- and two-line)
    → SentenceTransformer embeddings + BM25
    → FAISS + BM25 retrieval → heading boost
    → bounded section boundary detection
    → GLM-5.3 validation (via main.py)

Public API
----------
build_index(book_path, book_id, filename="", on_progress=None) -> BookIndex
get_index(book_path, book_id, filename="", on_progress=None)   -> BookIndex
locate_topic(index, topic, top_k=8)                            -> dict
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("syllabusiq.book_index")

# Keep PyTorch from exploding thread counts on Windows.
os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 4))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ---------------------------------------------------------------------------
# Lazy singletons
# ---------------------------------------------------------------------------
_faiss = None
_embedder = None
_bm25_cls = None
_index_lock = threading.Lock()
_embed_lock = threading.Lock()


def _get_faiss():
    global _faiss
    if _faiss is None:
        import faiss
        _faiss = faiss
    return _faiss


def _get_embedder():
    global _embedder
    if _embedder is None:
        import torch
        from sentence_transformers import SentenceTransformer
        torch.set_num_threads(os.cpu_count() or 4)
        os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            _embedder = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
        except Exception:
            _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def _get_bm25():
    global _bm25_cls
    if _bm25_cls is None:
        from rank_bm25 import BM25Okapi
        _bm25_cls = BM25Okapi
    return _bm25_cls


# ---------------------------------------------------------------------------
# LaTeX cleaning — critical for math-heavy PDFs (B.S. Grewal etc.)
# ---------------------------------------------------------------------------
_LATEX_DISPLAY_BRACKET = re.compile(r"\\\[.*?\\\]", re.DOTALL)
_LATEX_DISPLAY_DOLLAR = re.compile(r"\$\$.*?\$\$", re.DOTALL)
_LATEX_INLINE_PAREN = re.compile(r"\\\(.*?\\\)", re.DOTALL)
_LATEX_INLINE_DOLLAR = re.compile(r"\$[^$\n]{1,120}\$")
_LATEX_CMD_WITH_ARG = re.compile(r"\\[a-zA-Z]+\{[^{}]*\}")
_LATEX_CMD_BARE = re.compile(r"\\[a-zA-Z]+")
_WS_RUNS = re.compile(r"[ \t]+")
_NL_RUNS = re.compile(r"\n{2,}")


def _strip_latex(text: str) -> str:
    """Turn LaTeX-heavy OCR output into natural-language-ish prose."""
    if not text:
        return ""
    text = _LATEX_DISPLAY_BRACKET.sub(" ", text)
    text = _LATEX_DISPLAY_DOLLAR.sub(" ", text)
    text = _LATEX_INLINE_PAREN.sub(" ", text)
    text = _LATEX_INLINE_DOLLAR.sub(" ", text)
    text = _LATEX_CMD_WITH_ARG.sub(" ", text)
    text = _LATEX_CMD_BARE.sub(" ", text)
    text = _WS_RUNS.sub(" ", text)
    text = _NL_RUNS.sub("\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class Chunk:
    chunk_id: int
    page: int
    text: str
    heading: Optional[str] = None
    heading_level: int = 0
    is_front_matter: bool = False


@dataclass
class BookIndex:
    book_id: str
    filename: str
    path: str
    file_hash: str
    total_pages: int
    chunks: List[Chunk] = field(default_factory=list)
    faiss_index: object = None
    bm25: object = None
    embeddings: Optional[np.ndarray] = None
    headings_by_page: Dict[int, List[Tuple[int, str]]] = field(default_factory=dict)
    page_labels: Dict[int, str] = field(default_factory=dict)
    toc_entries: List[Dict[str, Any]] = field(default_factory=list)
    page_to_printed: Dict[int, int] = field(default_factory=dict)
    median_page_offset: int = 0
    from_cache: bool = False
    pdf_type: str = "Text PDF"
    extraction_method: str = "PyMuPDF"

    def get_printed_page(self, pdf_page: int) -> int:
        if self.page_to_printed and pdf_page in self.page_to_printed:
            return self.page_to_printed[pdf_page]
        if self.median_page_offset > 0 and pdf_page > self.median_page_offset and pdf_page <= self.total_pages:
            return pdf_page - self.median_page_offset
        return pdf_page

    def get_printed_label(self, pdf_page: int) -> str:
        if self.page_labels and pdf_page in self.page_labels:
            return str(self.page_labels[pdf_page])
        p = self.get_printed_page(pdf_page)
        return str(p)



# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
_INDEX_ROOT = Path(__file__).resolve().parent.parent / "book_indexes"
_INDEX_ROOT.mkdir(parents=True, exist_ok=True)
_CACHE: Dict[str, BookIndex] = {}


# ---------------------------------------------------------------------------
# Heading detection
# ---------------------------------------------------------------------------
# Matches:
#   "Chapter 5 Hyperbolic functions"
#   "71.2 Exponential or complex notation"
#   "SECTION 12.4 Eigenvalues"
#   "THE BINOMIAL THEOREM"  (all caps, 2+ words)
_HEADING_ONE_LINE = re.compile(
    r"^\s*(?:"
    r"(?:chapter|unit|section|module|part|appendix)\s+[\dIVXivx]+[\.\s:\-]+(.{3,140})"
    r"|(\d+(?:\.\d+){0,3})\s+([A-Z][^\n]{3,140})"
    r"|([A-Z][A-Z0-9 \-]{5,90})$"
    r")\s*$"
)
_NUMBER_ONLY = re.compile(r"^\s*(\d+(?:\.\d+){1,3})\s*$")
_TITLE_AFTER_NUMBER = re.compile(r"^\s*([A-Z][A-Za-z0-9][^\n]{2,140})\s*$")


def _classify_line(line: str) -> Optional[Tuple[int, str]]:
    """Return (level, heading_text) if the line is a heading, else None."""
    s = line.strip()
    if not s:
        return None

    m = _HEADING_ONE_LINE.match(s)
    if not m:
        return None

    if m.group(1):  # Chapter/Unit/Section/Module/Part/Appendix
        return (1, s)

    if m.group(2) and m.group(3):  # "12.4 Eigenvalues ..."
        num = m.group(2)
        level = min(2 + num.count("."), 3)
        return (level, f"{num} {m.group(3).strip()}")

    if m.group(4):  # ALL CAPS
        text = s
        if len(text.split()) < 2:
            return None
        digit_count = sum(c.isdigit() for c in text)
        if digit_count > 0 and digit_count / len(text) > 0.35:
            return None
        # Reject things like "XC5 20V" (short alphanumeric codes)
        if any(
            re.fullmatch(r"[A-Z]{1,3}\d{1,3}", w) or re.fullmatch(r"\d{1,3}[A-Z]{1,3}", w)
            for w in text.split()
        ):
            return None
        return (1, text)

    return None


def _detect_headings_in_page(text: str) -> List[Tuple[int, str]]:
    """Return every heading found in a page, including two-line forms."""
    lines = text.splitlines()
    found: List[Tuple[int, str]] = []
    i = 0
    while i < len(lines):
        line = lines[i]

        # 1) Single-line heading
        h = _classify_line(line)
        if h:
            found.append(h)
            i += 1
            continue

        # 2) Two-line: "71.2" on its own line, followed by the title
        num_match = _NUMBER_ONLY.match(line)
        if num_match and i + 1 < len(lines):
            next_line = lines[i + 1]
            title_match = _TITLE_AFTER_NUMBER.match(next_line)
            if title_match:
                num = num_match.group(1)
                title = title_match.group(1).strip()
                # Reject if the "title" is really the start of a sentence
                if len(title) <= 140 and not title.endswith((",", ";", ":")):
                    level = min(2 + num.count("."), 3)
                    found.append((level, f"{num} {title}"))
                    i += 2
                    continue

        i += 1

    return found


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def _chunk_pages(
    pages: List[Dict],
    target_chars: int = 1800,
    overlap_chars: int = 250,
) -> Tuple[List[Chunk], Dict[int, List[Tuple[int, str]]]]:
    chunks: List[Chunk] = []
    headings_by_page: Dict[int, List[Tuple[int, str]]] = {}
    cid = 0
    current_heading: Optional[str] = None
    current_level = 0

    for p in pages:
        try:
            page_num = int(p.get("page", 1))
        except Exception:
            page_num = 1

        raw = p.get("text", "") or ""
        text = _strip_latex(raw)
        if not text.strip():
            continue

        page_headings = _detect_headings_in_page(text)
        if page_headings:
            headings_by_page[page_num] = page_headings
            current_level, current_heading = page_headings[-1]

        # Detect front matter (preface, contents, copyright, title pages)
        is_front = False
        if page_num <= 20:
            low_text = text.lower()
            if any(fm in low_text for fm in ["table of contents", "contents", "preface", "all rights reserved", "printed in", "published by", "isbn"]):
                if "contents" in low_text or "preface" in low_text or "isbn" in low_text or "copyright" in low_text:
                    is_front = True

        if len(text) <= target_chars:
            chunks.append(Chunk(cid, page_num, text, current_heading, current_level, is_front_matter=is_front))
            cid += 1
        else:
            start = 0
            while start < len(text):
                end = min(start + target_chars, len(text))
                chunks.append(
                    Chunk(
                        cid,
                        page_num,
                        text[start:end],
                        current_heading,
                        current_level,
                        is_front_matter=is_front,
                    )
                )
                cid += 1
                if end == len(text):
                    break
                start = end - overlap_chars

    return chunks, headings_by_page


# ---------------------------------------------------------------------------
# Hashing / persistence
# ---------------------------------------------------------------------------
def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _index_dir(file_hash: str) -> Path:
    d = _INDEX_ROOT / file_hash
    d.mkdir(parents=True, exist_ok=True)
    return d


def _embed_texts(texts: List[str]) -> np.ndarray:
    model = _get_embedder()
    with _embed_lock:
        vecs = model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
    return vecs.astype("float32")


def _extract_toc_and_page_mapping(pdf_path: str, total_pages: int) -> Tuple[List[Dict[str, Any]], Dict[int, int], int, Dict[int, str]]:
    try:
        import fitz
        doc = fitz.open(pdf_path)
    except Exception as exc:
        logger.warning("Could not open %s for TOC: %s", pdf_path, exc)
        return [], {}, 0, {}

    page_to_printed: Dict[int, int] = {}
    page_labels: Dict[int, str] = {}
    offsets: List[int] = []

    sample_pages = min(80, len(doc))
    for p_idx in range(sample_pages):
        pdf_p = p_idx + 1
        try:
            text = doc[p_idx].get_text()
            lines = [l.strip() for l in text.splitlines() if l.strip()]

            # Detect compound section-page labels like "I - 1.21" or "1.21"
            for l in lines[:4] + lines[-3:]:
                m_comp = re.search(r"\b([IVXLCDM]+\s*[-–]\s*\d+\.\d+)\b", l)
                if m_comp:
                    page_labels[pdf_p] = m_comp.group(1).replace(" ", "")
                    break

            for l in lines[:3] + lines[-3:]:
                if re.fullmatch(r"\d{1,4}", l):
                    val = int(l)
                    if 1 <= val <= total_pages:
                        offsets.append(pdf_p - val)
                        break
        except Exception:
            pass

    import collections
    dominant_offset = 0
    if offsets:
        cnts = collections.Counter(offsets)
        top_offset, top_freq = cnts.most_common(1)[0]
        # Only adopt dominant offset if it has solid support (at least 3 pages or 10% of samples)
        if top_freq >= 3 or top_freq >= len(offsets) * 0.15:
            dominant_offset = top_offset

    if dominant_offset > 0:
        for pdf_p in range(1, total_pages + 1):
            if pdf_p > dominant_offset:
                page_to_printed[pdf_p] = pdf_p - dominant_offset

    def printed_of(p: int) -> int:
        if p in page_to_printed:
            return page_to_printed[p]
        if dominant_offset > 0 and p > dominant_offset:
            return p - dominant_offset
        return p

    raw_toc = []
    try:
        raw_toc = doc.get_toc()  # [level, title, page]
    except Exception as exc:
        logger.warning("Failed to read TOC: %s", exc)

    toc_entries = []
    current_chapter = None
    for i, item in enumerate(raw_toc):
        lvl, raw_t, p = item[0], item[1], item[2]
        clean_t = re.sub(r"\s+", " ", str(raw_t)).strip()
        if lvl <= 1:
            current_chapter = clean_t

        end_p = total_pages
        for j in range(i + 1, len(raw_toc)):
            nlvl, nt, np = raw_toc[j][0], raw_toc[j][1], raw_toc[j][2]
            if nlvl <= lvl and np >= p:
                end_p = np - 1 if np > p else np
                break

        next_sec = None
        if i + 1 < len(raw_toc):
            next_sec = re.sub(r"\s+", " ", str(raw_toc[i + 1][1])).strip()

        toc_entries.append({
            "level": int(lvl),
            "title": clean_t,
            "page": int(p),
            "end_page": max(int(p), int(end_p)),
            "chapter": current_chapter or clean_t,
            "next_section": next_sec,
            "printed_start": printed_of(p),
            "printed_end": printed_of(max(int(p), int(end_p))),
        })

    return toc_entries, page_to_printed, dominant_offset, page_labels


def _save_to_disk(index: BookIndex) -> None:
    d = _index_dir(index.file_hash)
    meta = {
        "book_id": index.book_id,
        "filename": index.filename,
        "path": index.path,
        "file_hash": index.file_hash,
        "total_pages": index.total_pages,
        "chunk_count": len(index.chunks),
        "headings_by_page": {str(k): v for k, v in index.headings_by_page.items()},
        "toc_entries": index.toc_entries,
        "page_to_printed": {str(k): v for k, v in index.page_to_printed.items()},
        "page_labels": {str(k): v for k, v in index.page_labels.items()},
        "median_page_offset": index.median_page_offset,
        "pdf_type": index.pdf_type,
        "extraction_method": index.extraction_method,
    }
    (d / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    chunks_payload = [
        {
            "chunk_id": c.chunk_id,
            "page": c.page,
            "text": c.text,
            "heading": c.heading,
            "heading_level": c.heading_level,
        }
        for c in index.chunks
    ]
    (d / "chunks.json").write_text(
        json.dumps(chunks_payload, ensure_ascii=False), encoding="utf-8"
    )

    if index.embeddings is not None:
        np.save(d / "embeddings.npy", index.embeddings)

    if index.faiss_index is not None:
        _get_faiss().write_index(index.faiss_index, str(d / "faiss.index"))


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _load_from_disk(file_hash: str) -> Optional[BookIndex]:
    d = _index_dir(file_hash)
    meta_p = d / "meta.json"
    chunks_p = d / "chunks.json"
    emb_p = d / "embeddings.npy"
    faiss_p = d / "faiss.index"

    if not all(p.exists() for p in (meta_p, chunks_p, emb_p, faiss_p)):
        return None

    try:
        meta = json.loads(meta_p.read_text(encoding="utf-8", errors="ignore"))
        chunks_raw = json.loads(chunks_p.read_text(encoding="utf-8", errors="ignore"))
        embeddings = np.load(emb_p)
        faiss = _get_faiss()
        faiss_index = faiss.read_index(str(faiss_p))
    except Exception as exc:
        logger.warning("Failed to load index %s: %s", file_hash, exc)
        return None

    chunks = []
    for c in chunks_raw:
        h = c.get("heading")
        hl = c.get("heading_level", 0)
        # Fix inverted fields from previous index build
        if isinstance(h, int) and isinstance(hl, str):
            h, hl = hl, h
        elif isinstance(h, int) and not isinstance(hl, str):
            h = None
        chunks.append(
            Chunk(
                chunk_id=c["chunk_id"],
                page=c["page"],
                text=c["text"],
                heading=str(h) if (h is not None and not isinstance(h, int)) else None,
                heading_level=int(hl) if (isinstance(hl, int) or (isinstance(hl, str) and hl.isdigit())) else 0,
            )
        )

    BM25 = _get_bm25()
    bm25 = BM25([_tokenize(c.text) for c in chunks])

    headings_by_page = {
        int(k): v for k, v in (meta.get("headings_by_page") or {}).items()
    }

    toc_entries = meta.get("toc_entries") or []
    page_to_printed = {
        int(k): v for k, v in (meta.get("page_to_printed") or {}).items()
    }
    page_labels = {
        int(k): v for k, v in (meta.get("page_labels") or {}).items()
    }
    median_offset = meta.get("median_page_offset", 0)

    # Lazily backfill TOC & printed page mappings once if missing or unpopulated
    if not meta.get("page_mapping_checked") and (not toc_entries or not page_labels) and Path(meta.get("path", "")).exists():
        try:
            toc_entries, page_to_printed, median_offset, page_labels = _extract_toc_and_page_mapping(
                meta["path"], meta.get("total_pages", 0)
            )
            meta["toc_entries"] = toc_entries
            meta["page_to_printed"] = {str(k): v for k, v in page_to_printed.items()}
            meta["page_labels"] = {str(k): v for k, v in page_labels.items()}
            meta["median_page_offset"] = median_offset
            meta["page_mapping_checked"] = True
            meta_p.write_text(
                json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except Exception as exc:
            logger.warning("Could not backfill TOC on load: %s", exc)

    return BookIndex(
        book_id=meta["book_id"],
        filename=meta["filename"],
        path=meta["path"],
        file_hash=meta["file_hash"],
        total_pages=meta["total_pages"],
        chunks=chunks,
        faiss_index=faiss_index,
        bm25=bm25,
        embeddings=embeddings,
        headings_by_page=headings_by_page,
        page_labels=page_labels,
        toc_entries=toc_entries,
        page_to_printed=page_to_printed,
        median_page_offset=median_offset,
        from_cache=True,
        pdf_type=meta.get("pdf_type", "Text PDF"),
        extraction_method=meta.get("extraction_method", "PyMuPDF"),
    )


load_from_disk = _load_from_disk


# ---------------------------------------------------------------------------
# Public: build / get
# ---------------------------------------------------------------------------
def build_index(
    book_path: str,
    book_id: str,
    filename: str = "",
    on_progress: Optional[Callable[[str, float], None]] = None,
) -> BookIndex:
    path = Path(book_path)
    if not path.exists():
        raise FileNotFoundError(f"Book not found: {book_path}")

    file_hash = _file_hash(path)

    # 1) Memory cache
    with _index_lock:
        cached = _CACHE.get(file_hash)
    if cached is not None:
        if on_progress:
            on_progress("Loaded from memory cache", 1.0)
        return cached

    # 2) Disk cache
    disk = _load_from_disk(file_hash)
    if disk is not None:
        with _index_lock:
            _CACHE[file_hash] = disk
        if on_progress:
            on_progress("Loaded from disk cache", 1.0)
        logger.info("[book_index] Loaded from disk: %s", path.name)
        return disk

    # 3) Fresh build
    if on_progress:
        on_progress("Extracting PDF text", 0.05)
    from app.pdf_service import extract_pdf  # type: ignore

    logger.info("[book_index] Extracting PDF: %s", path.name)
    extraction = extract_pdf(str(path))
    pages = extraction.get("pages", []) or []
    if not pages:
        raise ValueError(f"No pages extracted from {path.name}.")

    if on_progress:
        on_progress("Chunking pages", 0.20)
    chunks, headings_by_page = _chunk_pages(pages)
    if not chunks:
        raise ValueError(f"No text chunks produced for {path.name}.")

    if on_progress:
        on_progress(f"Embedding {len(chunks)} chunks", 0.25)
    logger.info("[book_index] Embedding %d chunks from %s", len(chunks), path.name)
    embeddings = _embed_texts([c.text for c in chunks])

    if on_progress:
        on_progress("Building FAISS index", 0.90)
    faiss = _get_faiss()
    faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
    faiss_index.add(embeddings)

    BM25 = _get_bm25()
    bm25 = BM25([c.text.lower().split() for c in chunks])

    ocr_pages_count = int(extraction.get("ocr_pages", 0))
    is_scanned = ocr_pages_count > 0 or (len(pages) > 0 and (ocr_pages_count / len(pages)) > 0.25)
    pdf_type_label = "Scanned PDF" if is_scanned else "Text PDF"
    extraction_method_label = "OCR (PaddleOCR / Tesseract)" if is_scanned else "PyMuPDF"

    toc_entries, page_to_printed, median_offset, page_labels = _extract_toc_and_page_mapping(
        str(path), int(extraction.get("total_pages", len(pages)))
    )

    book_index = BookIndex(
        book_id=book_id,
        filename=filename or path.name,
        path=str(path),
        file_hash=file_hash,
        total_pages=int(extraction.get("total_pages", len(pages))),
        chunks=chunks,
        faiss_index=faiss_index,
        bm25=bm25,
        embeddings=embeddings,
        headings_by_page=headings_by_page,
        page_labels=page_labels,
        toc_entries=toc_entries,
        page_to_printed=page_to_printed,
        median_page_offset=median_offset,
        from_cache=False,
        pdf_type=pdf_type_label,
        extraction_method=extraction_method_label,
    )

    if on_progress:
        on_progress("Saving to disk", 0.97)
    _save_to_disk(book_index)

    with _index_lock:
        _CACHE[file_hash] = book_index

    if on_progress:
        on_progress("Ready", 1.0)
    return book_index


def get_index(
    book_path: str,
    book_id: str,
    filename: str = "",
    on_progress: Optional[Callable[[str, float], None]] = None,
) -> BookIndex:
    return build_index(book_path, book_id, filename, on_progress)


# ---------------------------------------------------------------------------
# Retrieval helpers
# ---------------------------------------------------------------------------
CANONICAL_MATH_TERMS = [
    "characteristic", "equation", "matrices", "matrix", "eigenvalues", "eigenvalue",
    "eigenvectors", "eigenvector", "sampling", "distribution", "distributions",
    "hypotheses", "hypothesis", "testing", "probability", "statistics", "cayley",
    "hamilton", "diagonalization", "orthogonal", "transformation", "poisson",
    "binomial", "normal", "continuous", "discrete", "random", "variable", "variables",
    "standard", "error", "variance", "correlation", "regression", "estimation",
    "inference", "markov", "chain", "chains", "fourier", "series", "integral",
    "laplace", "transform", "transforms", "differential"
]


def _correct_query_spelling(q: str) -> str:
    import difflib
    words = q.lower().split()
    corrected = []
    for w in words:
        clean_w = re.sub(r"[^a-z0-9]", "", w)
        if len(clean_w) > 4:
            matches = difflib.get_close_matches(clean_w, CANONICAL_MATH_TERMS, n=1, cutoff=0.72)
            if matches:
                corrected.append(matches[0])
                continue
        corrected.append(w)
    return " ".join(corrected)


def _normalize_query(q: str) -> str:
    """Fuse common multi-word forms and normalize technical terms."""
    if not q:
        return ""
    q_fixed = _correct_query_spelling(q)
    s = q_fixed.strip().lower()
    s = re.sub(r"[\-_–—]+", " ", s)
    s = re.sub(r"\bis(ation|ed|ing|e\b)", r"iz\1", s)
    for spaced, fused in (
        ("eigen values", "eigenvalues"),
        ("eigen value", "eigenvalue"),
        ("eigen vectors", "eigenvectors"),
        ("eigen vector", "eigenvector"),
        ("cayley hamilton", "cayley-hamilton"),
    ):
        s = s.replace(spaced, fused)
    return re.sub(r"\s+", " ", s).strip()


GENERIC_STOP_WORDS = {
    "a", "an", "the", "and", "or", "of", "in", "on", "at", "to", "for", "with", "by", "from",
    "as", "into", "through", "during", "including", "until", "against", "among", "throughout",
    "chapter", "section", "part", "unit", "theory", "method", "methods", "introduction",
    "properties", "concept", "concepts", "definition", "definitions", "theorem", "theorems",
    "rule", "rules", "law", "laws", "problems", "problem", "exercise", "exercises", "example", "examples",
    "study", "overview", "analysis", "applications", "application", "basic", "basics"
}


def _get_concept_stems(q: str) -> List[str]:
    q_fixed = _correct_query_spelling(q)
    s = q_fixed.strip().lower()
    s = re.sub(r"[\-_–—]+", " ", s)
    words = [w for w in re.findall(r"[a-z0-9]+", s) if len(w) > 2 and w not in GENERIC_STOP_WORDS]
    stems = []
    for w in words:
        if w.startswith("eigen"):
            stems.append("eigen")
        elif "diagon" in w:
            stems.append("diagonali")
        elif "cayley" in w:
            stems.append("cayley")
        elif "hamilton" in w:
            stems.append("hamilton")
        elif w.endswith(("ies", "ses")):
            stems.append(w[:-3])
        elif w.endswith(("ing", "ed", "es")):
            stems.append(w[:-2])
        elif w.endswith("s") and len(w) > 3:
            stems.append(w[:-1])
        else:
            stems.append(w)
    out = []
    for st in stems:
        if st not in out:
            out.append(st)
    return out


def _get_distinctive_tokens(q: str) -> List[str]:
    return _get_concept_stems(q)


def _is_front_matter_chunk(c: Chunk) -> bool:
    if getattr(c, "is_front_matter", False):
        return True
    if c.page <= 25:
        txt_low = (c.text or "").lower()[:500]
        h_low = (c.heading or "").lower()
        if any(fm in txt_low for fm in ["contents", "table of contents", "preface", "all rights reserved", "isbn", "published by"]):
            return True
        if any(fm in h_low for fm in ["contents", "preface"]):
            return True
    return False


def _extract_sentence_evidence(text: str, topic: str) -> str:
    stems = _get_concept_stems(topic)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    best_sent = ""
    best_cnt = 0
    for s in sentences:
        s_clean = s.strip().replace("\n", " ")
        if len(s_clean) < 20 or len(s_clean) > 350:
            continue
        s_norm = _normalize_query(s_clean)
        cnt = sum(1 for st in stems if st in s_norm)
        if cnt > best_cnt:
            best_cnt = cnt
            best_sent = s_clean
    if best_sent:
        return best_sent
    clean_all = " ".join(text.split())
    return clean_all[:220] + "..." if len(clean_all) > 220 else clean_all


def _verify_page_content(
    text: str,
    heading: Optional[str],
    page_headings: List[str],
    topic: str
) -> Tuple[bool, float, str]:
    stems = _get_concept_stems(topic)
    if not stems:
        return False, 0.0, ""

    text_norm = _normalize_query(text)
    all_hdgs = [heading or ""] + page_headings
    hdgs_norm = " ".join([_normalize_query(h) for h in all_hdgs if h])

    hdg_matches = sum(1 for st in stems if st in hdgs_norm)
    txt_matches = sum(1 for st in stems if st in text_norm)

    # 1. Check if exact contiguous phrase appears in headings or body text
    t_clean = re.sub(r"[\-_–—]+", " ", topic.lower().strip())
    pattern = re.escape(t_clean)
    pattern = pattern.replace(r"\ distribution", r"\s+distributions?")
    pattern = pattern.replace(r"\ equation", r"\s+equations?")
    pattern = pattern.replace(r"\ eigenvalue", r"\s+eigenvalues?")
    pattern = pattern.replace(r"\ matrix", r"\s+(?:matrix|matrices)")

    if re.search(r"\b" + pattern + r"\b", hdgs_norm):
        passage = _extract_sentence_evidence(text, topic)
        return True, 0.98, passage

    if re.search(r"\b" + pattern + r"\b", text_norm):
        passage = _extract_sentence_evidence(text, topic)
        return True, 0.94, passage

    # 2. For multi-word topics (e.g. 2 or more distinctive stems), require that all stems
    # appear together within a tight word window (<= 3 words) in at least ONE single sentence
    if len(stems) >= 2:
        sentences = re.split(r"(?<=[.!?])\s+|\n{2,}", text)
        for s in sentences:
            s_clean = s.strip()
            if len(s_clean) < 15:
                continue
            s_words = re.findall(r"[a-z0-9]+", s_clean.lower())
            idx_map = {}
            for i, w in enumerate(s_words):
                for st in stems:
                    if w.startswith(st) or st in w:
                        if st not in idx_map:
                            idx_map[st] = []
                        idx_map[st].append(i)
            if len(idx_map) == len(stems):
                min_dist = min(abs(i1 - i2) for i1 in idx_map[stems[0]] for i2 in idx_map[stems[1]])
                if min_dist <= 3:
                    return True, 0.88, s_clean
        return False, 0.0, ""

    # 3. Single distinctive stem (len(stems) == 1)
    if len(stems) == 1 and (hdg_matches == 1 or txt_matches == 1):
        passage = _extract_sentence_evidence(text, topic)
        return True, 0.84, passage

    return False, 0.0, ""


def _expand_tokens(query: str) -> List[str]:
    raw = _tokenize(query)
    expanded = set(raw)
    for t in list(raw):
        if t in ("eigenvalues", "eigenvalue"):
            expanded.update(["eigenvalues", "eigenvalue", "eigen", "values", "value"])
        elif t in ("eigen", "values", "value"):
            expanded.update(["eigenvalues", "eigenvalue"])
        elif t.endswith("ies") and len(t) > 4:
            expanded.add(t[:-3] + "y")
        elif t.endswith("s") and len(t) > 3:
            expanded.add(t[:-1])
        elif len(t) > 3 and not t.endswith("s"):
            expanded.add(t + "s")
    return list(expanded)


def _keyword_scores(bm25, query: str, n: int) -> np.ndarray:
    tokens = _expand_tokens(query)
    if not tokens:
        return np.zeros(n, dtype="float32")
    scores = np.asarray(bm25.get_scores(tokens), dtype="float32")
    if scores.max() > 0:
        scores = scores / scores.max()
    return scores


def _semantic_scores(index: BookIndex, query: str, n: int) -> np.ndarray:
    model = _get_embedder()
    qv = model.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True
    ).astype("float32")
    sims, ids = index.faiss_index.search(qv, min(n, len(index.chunks)))
    out = np.zeros(n, dtype="float32")
    for sim, cid in zip(sims[0], ids[0]):
        if cid >= 0:
            out[cid] = float(sim)
    return out


def _heading_match_scores(index: BookIndex, query: str, n: int) -> np.ndarray:
    q_tokens = set(_get_distinctive_tokens(query))
    if not q_tokens:
        q_tokens = set(_expand_tokens(query.lower()))
    if not q_tokens:
        return np.zeros(n, dtype="float32")

    out = np.zeros(n, dtype="float32")
    for c in index.chunks:
        candidate_headings = []
        if c.heading:
            candidate_headings.append((c.heading_level, c.heading))
        for lvl, h in index.headings_by_page.get(c.page, []):
            if (lvl, h) not in candidate_headings:
                candidate_headings.append((lvl, h))
        for entry in index.toc_entries:
            if entry["page"] == c.page:
                candidate_headings.append((entry["level"], entry["title"]))

        for lvl, heading in candidate_headings:
            h_tokens = set(re.findall(r"[a-z0-9]+", heading.lower()))
            if not h_tokens:
                continue
            overlap = len(q_tokens & h_tokens)
            if overlap == 0:
                continue
            score = overlap / len(q_tokens)
            score *= 1.0 / (1.0 + 0.005 * len(heading))
            if lvl >= 2:
                score *= 1.25
            if score > out[c.chunk_id]:
                out[c.chunk_id] = score

    return out


def _merge_scores(
    kw: np.ndarray,
    sem: np.ndarray,
    head: np.ndarray,
    alpha: float = 0.35,
    beta: float = 0.25,
) -> np.ndarray:
    """Heading-match, semantic, keyword — weighted."""
    gamma = max(0.0, 1.0 - alpha - beta)
    return alpha * sem + beta * kw + gamma * head


def _match_toc(index: BookIndex, query: str) -> Optional[Tuple[Dict[str, Any], float]]:
    """High-precision match directly against the book's Table of Contents / bookmark hierarchy."""
    if not index.toc_entries:
        return None

    q_norm = _normalize_query(query)
    distinctive_q = set(_get_distinctive_tokens(query))
    if not distinctive_q:
        distinctive_q = set(re.findall(r"[a-z0-9]+", q_norm))
    if not distinctive_q:
        return None

    best_entry = None
    best_score = 0.0

    for entry in index.toc_entries:
        title = entry["title"]
        t_norm = _normalize_query(title)
        t_tokens = set(_get_distinctive_tokens(title))
        if not t_tokens:
            t_tokens = set(re.findall(r"[a-z0-9]+", t_norm))
        if not t_tokens:
            continue

        distinctive_overlap = distinctive_q & t_tokens
        if not distinctive_overlap:
            continue

        overlap_ratio = len(distinctive_overlap) / len(distinctive_q)
        # If multi-token query (e.g. 'Characteristic equation' or 'Sampling distribution'),
        # NEVER accept a partial 1-word match like 'The equation of a circle'
        if len(distinctive_q) >= 2 and overlap_ratio < 0.80:
            continue
        if len(distinctive_q) == 1 and len(distinctive_overlap) == 0:
            continue

        score = 0.0
        # Exact full match
        if q_norm == t_norm:
            score = 2.0
        elif q_norm in t_norm:
            score = 1.5 + (len(q_norm) / max(1, len(t_norm)))
        elif t_norm in q_norm:
            score = 1.3
        else:
            score = overlap_ratio
            if entry.get("level", 1) >= 2:
                score += 0.25

        if score > best_score:
            best_score = score
            best_entry = entry

    if best_entry and best_score >= 0.70:
        return best_entry, min(1.0, round(best_score if best_score <= 1.0 else 0.98, 4))

    return None


def _extract_related_and_subtopics(
    index: BookIndex,
    start_page: int,
    end_page: int,
    section_title: Optional[str] = None,
    chapter_title: Optional[str] = None,
) -> Tuple[List[str], List[str]]:
    """
    Extracts real sibling sections from the TOC under this chapter, and scans
    the actual text in [start_page, end_page] for definitions, bold terms,
    and mathematical concepts, filtering out noise words like Hence/Since/Thus.
    """
    subtopics: List[str] = []
    related_topics: List[str] = []
    seen = set()
    norm_sec = (section_title or "").lower().strip()
    norm_chap = (chapter_title or "").lower().strip()

    # 1. Sibling sections from TOC under this chapter
    if index.toc_entries and chapter_title:
        for entry in index.toc_entries:
            t = entry["title"]
            t_low = t.lower()
            if (
                entry["chapter"].lower() == norm_chap
                and t_low != norm_sec
                and t_low != norm_chap
                and entry["level"] >= 2
            ):
                clean_t = re.sub(r"^\d+(?:\.\d+)*\s*", "", t).strip()
                if clean_t and clean_t.lower() not in seen:
                    seen.add(clean_t.lower())
                    related_topics.append(t)

    # 2. Extract concepts from chunks and bold spans in the page range
    concept_stop_words = {
        "hence", "since", "thus", "therefore", "where", "when", "which",
        "example", "problem", "exercise", "figure", "table", "solution",
        "from", "now", "note", "given", "let", "the", "and", "or", "for",
        "with", "that", "this", "can", "all", "each", "case", "step",
        "these", "those", "such", "also", "then", "into", "onto", "both",
        "about", "after", "before", "between", "during", "under", "above",
        "chapter", "section", "page", "part", "unit"
    }

    span_chunks = [c for c in index.chunks if start_page <= c.page <= end_page]
    for c in span_chunks:
        for line in c.text.splitlines():
            s = line.strip()
            if not s:
                continue

            # Check quoted terms: 'general solution', 'boundary conditions'
            for m in re.finditer(r"['\u2018\u2019\"]([a-zA-Z\s]{4,45})['\u2018\u2019\"]", s):
                term = m.group(1).strip()
                t_lower = term.lower()
                if (
                    len(term) >= 4
                    and t_lower not in concept_stop_words
                    and t_lower not in seen
                    and len(term.split()) <= 4
                ):
                    seen.add(t_lower)
                    subtopics.append(term.title())

            # Check bullet lines or numbered items
            m = re.match(r"^\s*(?:[•\-\*]|\([a-z0-9]+\)|\d+\.\d+|\d+\.)\s*([A-Z][A-Za-z0-9\s\-,/]{3,50})", s)
            if m:
                cand = m.group(1).strip()
                cand_clean = re.sub(r"^[0-9\.\s\-]+", "", cand).strip()
                k = cand_clean.lower()
                first_word = k.split()[0] if k.split() else ""
                if (
                    len(cand_clean) >= 4
                    and first_word not in concept_stop_words
                    and k != norm_sec
                    and k not in seen
                    and not any(w in k for w in ["chapter", "table", "figure", "page", "isbn", "exercise"])
                ):
                    seen.add(k)
                    subtopics.append(cand_clean)

    # Check headings on these pages
    for p in range(start_page, end_page + 1):
        for lvl, h in index.headings_by_page.get(p, []):
            clean_h = re.sub(r"^\d+(?:\.\d+)*\s*", "", h).strip()
            k = clean_h.lower()
            if k not in seen and k != norm_sec and len(clean_h) >= 4:
                seen.add(k)
                subtopics.append(clean_h)

    subtopics = [st for st in subtopics if len(st) >= 3][:8]
    related_topics = [rt for rt in related_topics if len(rt) >= 3][:8]
    if not related_topics:
        related_topics = subtopics[:6]
    if not subtopics:
        subtopics = related_topics[:6]

    return subtopics, related_topics


_RH_EVEN = re.compile(r"^\s*(\d{1,4})\s+([A-Za-z\s,\-–—\(\)\.\/]{3,})\s*$")
_RH_ODD = re.compile(r"^\s*([A-Za-z\s,\-–—\(\)\.\/]{3,})\s+(\d{1,4})\s*$")


def _is_running_header(h: Optional[str]) -> bool:
    if not h:
        return False
    s = h.strip()
    return bool(_RH_EVEN.match(s) or _RH_ODD.match(s))


def _is_math_noise_heading(h: Optional[str]) -> bool:
    if not h:
        return False
    if any(ch in h for ch in ["=", "~", ">", "<", "{", "}", "\\", "[", "]"]):
        return True
    if re.search(r"\b(?:ad|be|at\+b|hereN)\b", h):
        return True
    return False


def _extract_chapter_from_header(h: Optional[str]) -> Optional[str]:
    if not h:
        return None
    m = _RH_ODD.match(h.strip())
    if m:
        c = m.group(1).strip()
        if not any(w in c.lower() for w in ["probability, statistics", "higher engineering", "advanced engineering"]):
            return c
    return None


def _find_section_bounds(
    index: BookIndex,
    top_chunk: Chunk,
    topic: str = "",
    max_walk_back: int = 25,
    max_walk_fwd: int = 50,
) -> Tuple[int, int, Optional[str], Optional[str], Optional[str], List[str], List[str]]:
    chunks = index.chunks
    if not chunks:
        return top_chunk.page, top_chunk.page, None, None, None, [], []

    idx = top_chunk.chunk_id
    section_title: Optional[str] = None
    chapter_title: Optional[str] = None
    matched_level: int = 2
    start_chunk_idx: int = idx

    # Check running headers from top chunk or nearby chunks to grab chapter title
    for k in range(max(0, idx - 10), min(len(chunks), idx + 10)):
        chk = chunks[k]
        if chk.heading:
            ch_cand = _extract_chapter_from_header(chk.heading)
            if ch_cand and not chapter_title:
                chapter_title = ch_cand
        for line in chk.text.splitlines()[:3]:
            ch_cand = _extract_chapter_from_header(line)
            if ch_cand and not chapter_title:
                chapter_title = ch_cand

    # 1. First, check if any chunk on top_chunk.page or nearby pages
    # contains an exact standalone title line matching the topic (e.g. "Sampling Distribution")
    topic_stems = _get_concept_stems(topic) if topic else []
    found_exact_line_heading = None
    found_line_page = None

    if topic_stems:
        search_start_cid = max(0, idx - 15)
        search_end_cid = min(len(chunks), idx + 5)
        best_line_score = 0
        for c in chunks[search_start_cid:search_end_cid]:
            for line in c.text.splitlines():
                s = line.strip()
                if 4 <= len(s) <= 65 and not s.endswith((".", ",", ";", ":", "-", "—")):
                    if _is_running_header(s):
                        continue
                    s_stems = _get_concept_stems(s)
                    if not s_stems:
                        continue
                    overlap = sum(1 for st in topic_stems if st in s_stems)
                    if overlap == len(topic_stems) and len(topic_stems) >= 1:
                        score = overlap / len(s_stems) if len(s_stems) else 0
                        if score > best_line_score:
                            best_line_score = score
                            found_exact_line_heading = s
                            found_line_page = c.page
                            start_chunk_idx = c.chunk_id

    # 2. Walk backward to nearest non-running-header heading (max 4 pages)
    i = idx
    steps = 0
    candidate_subheadings = []

    while i >= 0 and steps < max_walk_back:
        c = chunks[i]
        if top_chunk.page - c.page > 4:
            break
        if c.heading and not _is_running_header(c.heading) and not _is_math_noise_heading(c.heading):
            if c.heading_level >= 2:
                if section_title is None:
                    section_title = c.heading
                    matched_level = c.heading_level
                    start_chunk_idx = i
                candidate_subheadings.append((c.chunk_id, c.page, c.heading_level, c.heading))
            elif c.heading_level == 1:
                chapter_title = c.heading
                if section_title is None:
                    section_title = c.heading
                    matched_level = 1
                    start_chunk_idx = i
                break
        i -= 1
        steps += 1

    # Check for hierarchical parent section (e.g., if 1.6.4 Corollary was found, find 1.6)
    if section_title:
        m_num = re.search(r"^(\d+\.\d+)(?:\.\d+)+", section_title)
        if m_num:
            major_pfx = m_num.group(1)  # e.g. "1.6"
            # Walk back further to find the major section heading (e.g. "1.6 EIGENVALUES AND EIGENVECTORS")
            for cand_id, cand_p, cand_lvl, cand_h in candidate_subheadings:
                if re.match(r"^" + re.escape(major_pfx) + r"\s+[A-Za-z]", cand_h):
                    section_title = cand_h
                    matched_level = cand_lvl
                    start_chunk_idx = cand_id
                    break
            # Or scan headings_by_page for the major section heading
            for p in range(max(1, chunks[start_chunk_idx].page - 15), chunks[start_chunk_idx].page + 1):
                for lvl, h in index.headings_by_page.get(p, []):
                    if re.match(r"^" + re.escape(major_pfx) + r"\s+[A-Za-z]", h):
                        section_title = h
                        matched_level = lvl
                        for chk_i in range(max(0, start_chunk_idx - 30), start_chunk_idx + 1):
                            if chunks[chk_i].page == p:
                                start_chunk_idx = chk_i
                                break
                        break

    # Prioritize exact standalone topic heading found nearby over random/noise headings
    if found_exact_line_heading and not any(eq in found_exact_line_heading for eq in ["=", " is ", " are "]):
        if not section_title or _is_running_header(section_title) or _is_math_noise_heading(section_title) or (abs(top_chunk.page - (found_line_page or top_chunk.page)) <= 1):
            section_title = found_exact_line_heading

    # Check if a TOC entry matches this page
    matched_toc = None
    for entry in index.toc_entries:
        if entry["page"] <= top_chunk.page <= entry["end_page"] and entry.get("level", 1) >= 2:
            matched_toc = entry
            break

    if matched_toc:
        if not chapter_title:
            chapter_title = matched_toc["chapter"]
        if not section_title:
            section_title = matched_toc["title"]

    start_page = chunks[start_chunk_idx].page if start_chunk_idx >= 0 else top_chunk.page
    if found_line_page and found_line_page < start_page:
        start_page = found_line_page

    # Extract numerical prefix from section title (e.g., "1.6" from "1.6 EIGENVALUES AND EIGENVECTORS")
    sec_num_match = re.search(r"(\d+(?:\.\d+)*)", section_title or "")
    sec_num = sec_num_match.group(1) if sec_num_match else None

    # 3. Walk forward to next major heading / next section
    end_page = top_chunk.page
    next_section_title: Optional[str] = None
    j = max(idx, start_chunk_idx) + 1
    found_next = False

    # 3a. For unnumbered sections (sec_num is None), scan forward chunks for stand-alone section headings
    if not sec_num:
        for k in range(j, min(j + 30, len(chunks))):
            fc = chunks[k]
            if fc.page >= start_page + 2 and fc.page <= start_page + 6:
                for line in fc.text.splitlines():
                    line_s = line.strip()
                    if (
                        len(line_s) >= 4
                        and len(line_s) <= 60
                        and not _is_running_header(line_s)
                        and not _is_math_noise_heading(line_s)
                        and not any(line_s.endswith(punc) for punc in [".", ",", ";", ":", "=", "+", "-"])
                        and not any(eq in line_s for eq in ["=", "<", ">", "{", "}", "(", ")"])
                    ):
                        words = [w for w in line_s.split() if w.lower() not in {"and", "or", "of", "in", "on", "for", "the", "a", "an", "to"}]
                        if words and (line_s.isupper() or sum(1 for w in words if w[0].isupper()) >= len(words) * 0.7):
                            if line_s.lower() != (section_title or "").lower() and line_s.lower() != (chapter_title or "").lower():
                                if any(kw in line_s.lower() for kw in [
                                    "procedure for", "procedure of", "critical values", "tests of significance",
                                    "large sample", "small sample", "test 1", "test 2", "test 3"
                                ]):
                                    next_section_title = line_s
                                    end_page = max(start_page, fc.page - 1)
                                    found_next = True
                                    break
                if found_next:
                    break

    # 3b. Scan chunks forward
    if not found_next:
        fwd_steps = 0
        while j < len(chunks) and fwd_steps < max_walk_fwd:
            c = chunks[j]
            if (
                c.heading
                and not _is_running_header(c.heading)
                and not _is_math_noise_heading(c.heading)
                and c.heading != section_title
                and c.heading != chapter_title
            ):
                is_next = False
                next_num_match = re.search(r"(\d+(?:\.\d+)*)", c.heading)
                if sec_num and next_num_match:
                    next_num = next_num_match.group(1)
                    if sec_num.count(".") == 1:
                        if next_num != sec_num and not next_num.startswith(sec_num + "."):
                            is_next = True
                    else:
                        if next_num != sec_num:
                            is_next = True
                elif not sec_num and c.heading_level <= matched_level:
                    if c.page - start_page <= 5:
                        is_next = True
                elif sec_num and c.heading_level <= matched_level and (c.page - start_page <= 25):
                    is_next = True

                if is_next:
                    next_section_title = c.heading
                    end_page = max(start_page, c.page - 1 if c.page > start_page else c.page)
                    found_next = True
                    break
            j += 1
            fwd_steps += 1

    # 3c. If no next section was found in chunks, try forward headings_by_page
    if not found_next:
        if matched_toc and matched_toc.get("level", 1) >= 2 and matched_toc["end_page"] > start_page:
            end_page = matched_toc["end_page"]
        else:
            found_fwd = False
            for p in range(start_page + 1, min(start_page + 30, index.total_pages + 1)):
                hdgs = index.headings_by_page.get(p, [])
                for lvl, h in hdgs:
                    if _is_running_header(h) or _is_math_noise_heading(h):
                        continue
                    m_num = re.search(r"(\d+(?:\.\d+)*)", h)
                    if sec_num and m_num:
                        next_num = m_num.group(1)
                        if sec_num.count(".") == 1:
                            if next_num != sec_num and not next_num.startswith(sec_num + "."):
                                end_page = max(start_page, p - 1)
                                next_section_title = h
                                found_fwd = True
                                break
                        elif next_num != sec_num:
                            end_page = max(start_page, p - 1)
                            next_section_title = h
                            found_fwd = True
                            break
                    elif any(h.lower().startswith(pfx) for pfx in ["chapter", "part", "unit"]):
                        if not sec_num:
                            end_page = min(start_page + 3, p - 1)
                        else:
                            end_page = max(start_page, p - 1)
                        next_section_title = h
                        found_fwd = True
                        break
                if found_fwd:
                    break
            if not found_fwd:
                end_page = min(start_page + 3, chunks[-1].page)

    # 3d. Enforce tight bounds on unnumbered sections so they don't sprawl
    if not sec_num and end_page > start_page + 4:
        end_page = min(start_page + 3, chunks[-1].page)

    if end_page < start_page:
        end_page = start_page

    subtopics, related_topics = _extract_related_and_subtopics(
        index, start_page, end_page, section_title, chapter_title
    )

    return start_page, end_page, section_title, chapter_title, next_section_title, subtopics, related_topics


# ---------------------------------------------------------------------------
# Public: locate_topic (Steps 4 & 5)
# ---------------------------------------------------------------------------
def locate_topic(index: BookIndex, topic: str, top_k: int = 8) -> Dict:
    n = len(index.chunks)
    if n == 0:
        return {"found": False, "reason": "empty_index"}

    topic = _correct_query_spelling(topic)
    normalized = _normalize_query(topic)

    # --- STAGE 1: Check High-Precision Table of Contents Match ---
    toc_res = _match_toc(index, topic)
    if toc_res:
        toc_entry, toc_score = toc_res
        start_p = toc_entry["page"]
        chap = toc_entry["chapter"]
        sec = toc_entry["title"]
        next_sec = toc_entry["next_section"]

        # Ensure TOC entry points to body content, not front matter
        is_front_toc = (start_p <= 12 and any(entry["page"] > 12 for entry in index.toc_entries))
        if not is_front_toc:
            toc_passage = ""
            for c in index.chunks:
                if c.page == start_p and c.text:
                    toc_passage = _extract_sentence_evidence(c.text, topic)
                    if toc_passage:
                        break
            if not toc_passage:
                toc_passage = f"Section '{sec}' under chapter '{chap}' in prescribed reference text."

            end_p = toc_entry["end_page"]
            if end_p == start_p or (toc_entry.get("level", 1) <= 1):
                chap_end = start_p
                for e in index.toc_entries:
                    if e["chapter"] == chap and e["end_page"] > chap_end and e.get("level", 1) >= 2:
                        chap_end = e["end_page"]
                end_p = max(start_p, min(start_p + 3, chap_end))

            printed_start = index.get_printed_label(start_p)
            printed_end = index.get_printed_label(end_p)
            try:
                if int(printed_end) < int(printed_start):
                    printed_start = str(start_p)
                    printed_end = str(end_p)
            except (ValueError, TypeError):
                pass

            subtopics, related_topics = _extract_related_and_subtopics(
                index, start_p, end_p, sec, chap
            )

            page_display = (
                f"Page {printed_start} – {printed_end} (PDF {start_p} – {end_p})"
                if printed_start != printed_end
                else f"Page {printed_start} (PDF {start_p})"
            )

            cand_pages = [p for p in range(start_p, min(start_p + 8, end_p + 1))]
            candidate_pages_detail = [
                {
                    "page": p,
                    "similarity": round(toc_score, 2) if p == start_p else round(toc_score * 0.9, 2),
                    "is_top": (p == start_p),
                }
                for p in cand_pages
            ]

            return {
                "found": True,
                "verified": True,
                "start_page": start_p,
                "end_page": end_p,
                "printed_start_page": printed_start,
                "printed_end_page": printed_end,
                "page_display": page_display,
                "section_title": sec,
                "chapter_title": chap,
                "next_section_title": next_sec,
                "supporting_passage": toc_passage,
                "subtopics": subtopics,
                "related_topics": related_topics,
                "match_confidence": round(toc_score, 4),
                "candidate_pages": cand_pages,
                "candidate_pages_detail": candidate_pages_detail,
                "section_boundary": {
                    "start_heading": sec,
                    "start_page": start_p,
                    "end_page": end_p,
                    "printed_start_page": printed_start,
                    "printed_end_page": printed_end,
                    "next_heading": next_sec,
                    "subtopics": subtopics,
                    "related_topics": related_topics,
                    "chapter": chap,
                },
                "pdf_type": getattr(index, "pdf_type", "Text PDF"),
                "extraction_method": getattr(index, "extraction_method", "PyMuPDF"),
                "top_chunks": [
                    {
                        "chunk_id": c.chunk_id,
                        "page": c.page,
                        "similarity": round(toc_score, 4),
                        "preview": c.text[:200],
                    }
                    for c in index.chunks if c.page == start_p
                ][:top_k],
            }

    # --- STAGE 2: Hybrid RAG Search (FAISS + BM25 + Heading Boost) ---
    kw = _keyword_scores(index.bm25, normalized, n)
    sem = _semantic_scores(index, normalized, n)
    head = _heading_match_scores(index, normalized, n)
    merged = _merge_scores(kw, sem, head)

    order = np.argsort(-merged)
    if len(order) == 0 or merged[order[0]] <= 0:
        return {
            "found": False,
            "reason": "topic_not_found_in_book",
            "candidate_pages": [],
            "candidate_pages_detail": [],
            "top_chunks": [],
            "pdf_type": getattr(index, "pdf_type", "Text PDF"),
            "extraction_method": getattr(index, "extraction_method", "PyMuPDF"),
        }

    # Strict Page-Level Verification Gate
    verified_chunk = None
    verified_passage = ""
    verified_conf = 0.0

    for idx_i in order[:min(35, len(order))]:
        cand_c = index.chunks[int(idx_i)]
        if _is_front_matter_chunk(cand_c):
            continue
        page_hdgs = [h for _, h in index.headings_by_page.get(cand_c.page, [])]
        is_valid, conf_val, passage = _verify_page_content(
            cand_c.text, cand_c.heading, page_hdgs, topic
        )
        if is_valid:
            verified_chunk = cand_c
            verified_passage = passage
            sim_score = float(merged[int(idx_i)])
            verified_conf = min(0.98, max(0.80, (conf_val * 0.7) + (sim_score * 0.3)))
            break

    if verified_chunk is None:
        return {
            "found": False,
            "reason": "topic_not_found_in_book",
            "candidate_pages": [],
            "candidate_pages_detail": [],
            "top_chunks": [],
            "pdf_type": getattr(index, "pdf_type", "Text PDF"),
            "extraction_method": getattr(index, "extraction_method", "PyMuPDF"),
        }

    top_chunk = verified_chunk
    start_page, end_page, section, chapter, next_section, subtopics, related_topics = _find_section_bounds(
        index, top_chunk, topic=topic
    )

    printed_start = index.get_printed_label(start_page)
    printed_end = index.get_printed_label(end_page)
    try:
        if int(printed_end) < int(printed_start):
            printed_start = str(start_page)
            printed_end = str(end_page)
    except (ValueError, TypeError):
        pass

    page_display = (
        f"Page {printed_start} – {printed_end} (PDF {start_page} – {end_page})"
        if printed_start != printed_end
        else f"Page {printed_start} (PDF {start_page})"
    )

    page_score_map: Dict[int, float] = {}
    for i in order[:min(top_k * 2, len(order))]:
        cid = int(i)
        p = index.chunks[cid].page
        sc = float(merged[cid])
        if p not in page_score_map or sc > page_score_map[p]:
            page_score_map[p] = sc

    sorted_pages = sorted(page_score_map.keys())
    top_page = top_chunk.page
    candidate_pages_detail = [
        {
            "page": p,
            "similarity": round(page_score_map[p], 2),
            "is_top": (p == top_page),
        }
        for p in sorted_pages
    ]

    return {
        "found": True,
        "verified": True,
        "start_page": start_page,
        "end_page": end_page,
        "printed_start_page": printed_start,
        "printed_end_page": printed_end,
        "page_display": page_display,
        "section_title": section,
        "chapter_title": chapter,
        "next_section_title": next_section,
        "supporting_passage": verified_passage,
        "subtopics": subtopics,
        "related_topics": related_topics,
        "match_confidence": round(verified_conf, 4),
        "candidate_pages": sorted_pages,
        "candidate_pages_detail": candidate_pages_detail,
        "section_boundary": {
            "start_heading": section,
            "start_page": start_page,
            "end_page": end_page,
            "printed_start_page": printed_start,
            "printed_end_page": printed_end,
            "next_heading": next_section,
            "subtopics": subtopics,
            "related_topics": related_topics,
            "chapter": chapter,
        },
        "pdf_type": getattr(index, "pdf_type", "Text PDF"),
        "extraction_method": getattr(index, "extraction_method", "PyMuPDF"),
        "top_chunks": [
            {
                "chunk_id": int(i),
                "page": index.chunks[int(i)].page,
                "similarity": round(float(merged[int(i)]), 4),
                "preview": index.chunks[int(i)].text[:200],
            }
            for i in order[:top_k]
        ],
    }