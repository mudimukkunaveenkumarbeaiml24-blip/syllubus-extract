import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Union

import fitz  # PyMuPDF
import pytesseract
from PIL import Image

from app.config import (
    OCR_DPI,
    OCR_LANGUAGE,
    OCR_MIN_TEXT_LENGTH,
    OCR_MAX_WORKERS,
)

# Configure Tesseract binary path on Windows if installed
if hasattr(pytesseract, "pytesseract"):
    _tess_candidates = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
    ]
    for _tc in _tess_candidates:
        if os.path.exists(_tc):
            pytesseract.pytesseract.tesseract_cmd = _tc
            break


# =========================================================
# OCR SINGLE PAGE FUNCTION (THREAD-SAFE)
# =========================================================

def _ocr_page_image(page: fitz.Page, dpi: int = OCR_DPI, lang: str = OCR_LANGUAGE) -> str:
    """
    Convert a fitz.Page to an image and run Tesseract OCR.
    """
    matrix = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    text = pytesseract.image_to_string(image, lang=lang)
    return text.strip()


def extract_page_by_index(
    pdf_path_str: str,
    page_index: int,
    dpi: int = OCR_DPI,
    lang: str = OCR_LANGUAGE,
    min_length: int = OCR_MIN_TEXT_LENGTH,
) -> Dict[str, Any]:
    """
    Extract text from a specific page index in a thread-safe manner
    by opening the document instance within the thread.
    """
    page_number = page_index + 1
    used_ocr = False
    final_text = ""
    ocr_error = None

    try:
        with fitz.open(pdf_path_str) as doc:
            if page_index < 0 or page_index >= len(doc):
                return {
                    "page": page_number,
                    "text": "",
                    "text_length": 0,
                    "used_ocr": False,
                    "status": "unreadable",
                    "error": "Page index out of range",
                }

            page = doc[page_index]
            pymupdf_text = page.get_text("text").strip()

            # If embedded text is sufficient, use it.
            # Otherwise, perform OCR on the page.
            if len(pymupdf_text) >= min_length:
                final_text = pymupdf_text
            else:
                try:
                    ocr_text = _ocr_page_image(page, dpi=dpi, lang=lang)
                    if len(ocr_text) >= len(pymupdf_text):
                        final_text = ocr_text
                        used_ocr = True
                    else:
                        final_text = pymupdf_text
                except Exception as exc:
                    ocr_error = str(exc)
                    final_text = pymupdf_text

    except Exception as exc:
        ocr_error = str(exc)

    return {
        "page": page_number,
        "text": final_text,
        "text_length": len(final_text),
        "used_ocr": used_ocr,
        "status": "ok" if final_text else "unreadable",
        "ocr_error": ocr_error,
    }


# =========================================================
# EXTRACT COMPLETE PDF (PARALLEL & EFFICIENT)
# =========================================================

def extract_pdf(
    pdf_path: Union[str, Path]
) -> Dict[str, Any]:
    """
    Extract all pages from a syllabus PDF efficiently using parallel processing.
    Preserves exact page numbering with '===== PAGE N =====' headers.
    """
    pdf_path = Path(pdf_path).resolve()

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported.")

    pdf_path_str = str(pdf_path)

    # 1. Get total page count
    with fitz.open(pdf_path_str) as doc:
        total_pages = len(doc)

    if total_pages == 0:
        return {
            "filename": pdf_path.name,
            "total_pages": 0,
            "ocr_pages": 0,
            "unreadable_pages": [],
            "pages": [],
            "text": "",
        }

    # 2. Extract pages in parallel
    num_workers = min(OCR_MAX_WORKERS, total_pages)
    page_results = [None] * total_pages
    completed_count = 0

    print(f"[OCR] Starting extraction for {pdf_path.name}: {total_pages} pages using {num_workers} workers")

    if total_pages == 1 or num_workers <= 1:
        for idx in range(total_pages):
            res = extract_page_by_index(pdf_path_str, idx)
            page_results[idx] = res
            print(f"[OCR] Processing page {idx + 1}/{total_pages} (chars: {res.get('text_length', 0)})")
    else:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_idx = {
                executor.submit(extract_page_by_index, pdf_path_str, idx): idx
                for idx in range(total_pages)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                completed_count += 1
                try:
                    res = future.result()
                    page_results[idx] = res
                    if completed_count % 10 == 0 or completed_count == total_pages:
                        print(f"[OCR] Processing page {completed_count}/{total_pages}")
                except Exception as exc:
                    page_results[idx] = {
                        "page": idx + 1,
                        "text": "",
                        "text_length": 0,
                        "used_ocr": False,
                        "status": "unreadable",
                        "ocr_error": str(exc),
                    }

    # 3. Assemble results and summary
    total_ocr_pages = 0
    unreadable_pages = []
    combined_parts = []

    for page_res in page_results:
        if page_res.get("used_ocr"):
            total_ocr_pages += 1
        if page_res.get("status") == "unreadable":
            unreadable_pages.append(page_res["page"])

        combined_parts.append(
            f"===== PAGE {page_res['page']} =====\n{page_res['text']}\n"
        )

    combined_text = "\n".join(combined_parts).strip()
    print(f"[OCR] OCR completed: {total_pages} pages ({total_ocr_pages} used OCR)")

    return {
        "filename": pdf_path.name,
        "total_pages": total_pages,
        "ocr_pages": total_ocr_pages,
        "unreadable_pages": unreadable_pages,
        "pages": page_results,
        "text": combined_text,
    }


# =========================================================
# TEST FUNCTION
# =========================================================

def test_pdf(
    pdf_path: Union[str, Path]
):
    """
    Print a simple extraction report.
    """
    result = extract_pdf(pdf_path)

    print()
    print("=" * 60)
    print("SYLLABUSIQ PDF EXTRACTION TEST")
    print("=" * 60)
    print("File       :", result["filename"])
    print("Pages      :", result["total_pages"])
    print("OCR pages  :", result["ocr_pages"])
    print("Unreadable :", result["unreadable_pages"])
    print("=" * 60)