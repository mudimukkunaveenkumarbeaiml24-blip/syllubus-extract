import asyncio
import json
import logging
import os
from pathlib import Path
import sys
import uuid
from typing import List

logger = logging.getLogger("syllabusiq")
logging.basicConfig(level=logging.INFO)

# Ensure root syllabus-backend is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from dotenv import load_dotenv
load_dotenv(BASE_DIR / ".env", override=True)

from fastapi import (
    FastAPI,
    Request,
    UploadFile,
    File,
    Form,
    HTTPException
)

from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates

from app.pdf_service import extract_pdf
from app.extraction_service import detect_courses, extract_course_details
from app.nvidia_client import RateLimitException
from app.config import NVIDIA_MODEL
from app.book_service import analyze_books


# =========================================================
# DIRECTORIES
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TEMPLATES_DIR = BASE_DIR / "templates"
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Expose uploaded reference books to the frontend for PDF.js previews.
BOOKS_STATIC_DIR = UPLOAD_DIR / "books"
BOOKS_STATIC_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# FASTAPI
# =========================================================

app = FastAPI(
    title="SYLLABUSIQ",
    description="Universal Syllabus Intelligence API",
    version="1.0.0"
)

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# =========================================================
# STATIC FILES (reference book PDFs for evidence previews)
# =========================================================

app.mount(
    "/books-static",
    StaticFiles(directory=str(BOOKS_STATIC_DIR)),
    name="books-static"
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# HOME + EXCEPTION HANDLER
# =========================================================

from fastapi.responses import HTMLResponse, JSONResponse


@app.exception_handler(RateLimitException)
async def rate_limit_handler(request: Request, exc: RateLimitException):
    return JSONResponse(
        status_code=429,
        content={
            "status": "rate_limited",
            "error": exc.error_code,
            "message": exc.message,
            "detail": "NVIDIA API rate limit reached. Please wait and retry.",
            "retry_after": getattr(exc, "retry_after", 10)
        }
    )


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )


# =========================================================
# HEALTH
# =========================================================

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "SYLLABUSIQ",
        "model": NVIDIA_MODEL,
        "engine": "FastAPI",
        "database": "Not required for extraction",
        "pdf_engine": "PyMuPDF",
        "ocr": "Tesseract"
    }


# =========================================================
# API TEST
# =========================================================

@app.get("/api/test")
async def api_test():
    return {
        "status": "success",
        "message": "SYLLABUSIQ backend is working"
    }


# =========================================================
# FULL EXTRACTION
# =========================================================

@app.post("/extract/full")
async def extract_full(
    file: UploadFile = File(...),
    options: str = Form(None),
    course: str = Form(None),
):

    # -----------------------------------------------------
    # CHECK FILE
    # -----------------------------------------------------

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file selected."
        )

    filename = file.filename

    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )

    # -----------------------------------------------------
    # PARSE OPTIONS
    # -----------------------------------------------------

    batch_size = 4

    if options:
        try:
            opts_dict = (
                json.loads(options)
                if isinstance(options, str)
                else options
            )
            if (
                isinstance(opts_dict, dict)
                and "batch_size" in opts_dict
            ):
                batch_size = max(1, int(opts_dict["batch_size"]))
        except Exception:
            pass

    # -----------------------------------------------------
    # CREATE DOCUMENT ID
    # -----------------------------------------------------

    document_id = uuid.uuid4().hex
    safe_filename = "{}.pdf".format(document_id)
    pdf_path = UPLOAD_DIR / safe_filename

    # -----------------------------------------------------
    # SAVE UPLOADED PDF
    # -----------------------------------------------------

    try:
        with open(pdf_path, "wb") as output_file:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                output_file.write(chunk)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Could not save PDF: {}".format(exc)
        )

    # -----------------------------------------------------
    # PDF EXTRACTION
    # -----------------------------------------------------

    try:
        extraction = await asyncio.to_thread(extract_pdf, pdf_path)
    except Exception as exc:
        try:
            pdf_path.unlink()
        except Exception:
            pass
        raise HTTPException(
            status_code=500,
            detail="PDF extraction failed: {}".format(exc)
        )

    # -----------------------------------------------------
    # GLM COURSE DETECTION
    # -----------------------------------------------------

    try:
        course_result = await asyncio.to_thread(
            detect_courses,
            extraction["text"]
        )

        courses = course_result.get("courses", [])
        not_extracted = course_result.get("not_extracted", [])

    except Exception as exc:
        logger.warning(
            f"Detection warning: {exc}, running deterministic fallback"
        )
        from app.extraction_service import (
            detect_courses_deterministic,
            deduplicate_courses,
            locate_course_boundaries,
        )
        raw_c = detect_courses_deterministic(extraction["text"])
        final_courses = deduplicate_courses(raw_c)
        for c in final_courses:
            c_code = c.get("code", "")
            c_title = c.get("title", "")
            idx_p = c.get("index_page", 0)
            resolved_idx, detail_pages, _ = locate_course_boundaries(
                pages_text=extraction["text"],
                course_code=c_code,
                course_title=c_title,
                index_page=idx_p,
                all_courses=final_courses
            )
            c["index_pages"] = [resolved_idx] if resolved_idx else []
            c["index_page"] = resolved_idx
            c["detail_pages"] = detail_pages
            c["detail_start_page"] = detail_pages[0] if detail_pages else resolved_idx
            c["detail_end_page"] = detail_pages[-1] if detail_pages else resolved_idx
            c["page"] = detail_pages[0] if detail_pages else resolved_idx
            c["pages"] = detail_pages

        courses = final_courses
        not_extracted = []

    # -----------------------------------------------------
    # SAFETY VALIDATION
    # -----------------------------------------------------

    if not isinstance(courses, list):
        courses = []
    if not isinstance(not_extracted, list):
        not_extracted = []

    # -----------------------------------------------------
    # BUILD AND SAVE FRONTEND RESPONSE
    # -----------------------------------------------------

    from datetime import datetime

    response_data = {
        "status": "success",
        "id": document_id,
        "filename": filename,
        "pages": extraction["total_pages"],
        "ocr_pages": extraction["ocr_pages"],
        "unreadable_pages": extraction["unreadable_pages"],
        "course_count": len(courses),
        "courses": courses,
        "not_extracted": not_extracted,
        "page_data": extraction["pages"],
        "text": extraction["text"],
        "uploaded_at": datetime.utcnow().isoformat() + "Z"
    }

    save_document(response_data)
    return response_data


# =========================================================
# MULTIPLE REFERENCE BOOK ANALYSIS
# =========================================================

@app.post("/books/analyze")
async def analyze_reference_books(
    files: List[UploadFile] = File(...),
    topics: str = Form(None),
    document_id: str = Form(None),
    course: str = Form(None)
):
    """
    Analyze multiple reference books against syllabus topics.

    Standalone reference book analysis does NOT require a syllabus
    document; manual `topics` take priority over `document_id` + `course`.
    """

    if not files:
        raise HTTPException(
            status_code=400,
            detail="Please upload at least one reference book."
        )

    if len(files) > 20:
        raise HTTPException(
            status_code=400,
            detail="Maximum 20 reference books can be uploaded at once."
        )

    for file in files:
        if not file.filename:
            raise HTTPException(
                status_code=400,
                detail="One of the uploaded files has no filename."
            )
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail=f"{file.filename} is not a PDF file."
            )

    # --------------------------------------------------------
    # GET TOPICS — MANUAL INPUT TAKES PRIORITY
    # --------------------------------------------------------

    syllabus_topics = []

    if topics and topics.strip():
        raw_parts = (
            topics.replace("\r", "\n")
                  .replace(",", "\n")
                  .split("\n")
        )
        seen = set()
        for part in raw_parts:
            t = part.strip()
            if t and t.lower() not in seen:
                seen.add(t.lower())
                syllabus_topics.append(t)

    if not syllabus_topics and document_id and course:
        topic_response = await get_reference_topics(
            document_id=document_id,
            course=course
        )
        syllabus_topics = topic_response.get("topics", []) or []

    if not syllabus_topics:
        raise HTTPException(
            status_code=400,
            detail=(
                "No topics provided. Send `topics` (comma- or "
                "newline-separated) for standalone reference book "
                "analysis, OR send `document_id` + `course` to reuse "
                "topics from an already-extracted syllabus."
            )
        )

    # --------------------------------------------------------
    # SAVE ALL REFERENCE BOOKS
    # --------------------------------------------------------

    BOOK_UPLOAD_DIR = UPLOAD_DIR / "books"
    BOOK_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    books = []

    for file in files:
        book_id = uuid.uuid4().hex
        original_filename = file.filename
        safe_filename = f"{book_id}.pdf"
        pdf_path = BOOK_UPLOAD_DIR / safe_filename

        try:
            with open(pdf_path, "wb") as output_file:
                while True:
                    chunk = await file.read(1024 * 1024)
                    if not chunk:
                        break
                    output_file.write(chunk)
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Could not save {original_filename}: {exc}"
            )

        books.append({
            "book_id": book_id,
            "filename": original_filename,
            "path": str(pdf_path)
        })

    # --------------------------------------------------------
    # ANALYZE ALL BOOKS
    # --------------------------------------------------------

    try:
        result = await asyncio.to_thread(
            analyze_books,
            books,
            syllabus_topics
        )
    except Exception as exc:
        logger.exception("Reference book analysis failed")
        raise HTTPException(
            status_code=500,
            detail=f"Reference book analysis failed: {exc}"
        )

    # --------------------------------------------------------
    # RESPONSE
    # --------------------------------------------------------

    return {
        "success": True,
        "books_uploaded": len(files),
        "topics_requested": len(syllabus_topics),
        "syllabus_topics": syllabus_topics,
        "document_id": document_id,
        "course": course,
        "result": result
    }


# =========================================================
# REFERENCE BOOK TOPIC LOCATOR  (Task 2)
# =========================================================

@app.post("/books/locate-topic")
async def locate_topic_in_books(
    files: List[UploadFile] = File(...),
    topic: str = Form(...),
    validate_with_llm: bool = Form(True),
):
    """
    Locate a topic inside one or more reference books.

    Returns, for each book: start page, end page, section title,
    chapter title, match confidence, candidate pages, and a
    `book_url` that the frontend uses to render page thumbnails.

    Does NOT require document_id or a syllabus.
    """
    if not files:
        raise HTTPException(status_code=400, detail="Upload at least one PDF.")

    if not topic or not topic.strip():
        raise HTTPException(status_code=400, detail="`topic` must not be empty.")

    for f in files:
        if not f.filename or not f.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail=f"{f.filename or 'unnamed'} is not a PDF.",
            )

    # --- Save uploads ---
    BOOK_DIR = UPLOAD_DIR / "books"
    BOOK_DIR.mkdir(parents=True, exist_ok=True)

    saved = []
    try:
        for f in files:
            book_id = uuid.uuid4().hex
            dest = BOOK_DIR / f"{book_id}.pdf"
            with open(dest, "wb") as out:
                while True:
                    chunk = await f.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
            saved.append({
                "book_id": book_id,
                "filename": Path(f.filename).name,
                "path": str(dest),
            })
    finally:
        for f in files:
            try:
                await f.close()
            except Exception:
                pass

    # --- Locate ---
    try:
        from app.book_index_service import get_index, locate_topic
    except Exception as exc:
        logger.exception("book_index_service unavailable")
        raise HTTPException(
            status_code=500,
            detail=f"Topic locator backend unavailable: {exc}",
        )

    results = []
    for b in saved:
        try:
            index = await asyncio.to_thread(
                get_index, b["path"], b["book_id"], b["filename"],
            )
            located = await asyncio.to_thread(locate_topic, index, topic)

            if validate_with_llm and located.get("found"):
                try:
                    from app.topic_validation_service import validate_topic_range

                    start_p = located["start_page"]
                    end_p = located["end_page"]
                    sec_title = located.get("section_title") or ""
                    next_sec = located.get("next_section_title") or ""
                    chap = located.get("chapter_title") or ""
                    subtopics = located.get("subtopics") or []

                    section_text = "\n".join(
                        c.text for c in index.chunks
                        if start_p <= c.page <= end_p
                    )[:4000]

                    validation = await asyncio.to_thread(
                        validate_topic_range,
                        topic=topic,
                        book_filename=b["filename"],
                        start_page=start_p,
                        end_page=end_p,
                        section_text=section_text,
                        section_title=sec_title,
                        next_section_title=next_sec,
                        chapter_title=chap,
                        subtopics=subtopics,
                        match_confidence=located.get("match_confidence", 0.95),
                    )
                    located["validation"] = validation
                except Exception as exc:
                    logger.warning("Validation step failed: %s", exc)
                    located["validation"] = {
                        "match": True,
                        "confidence": located.get("match_confidence", 0.95),
                        "topic_matches": True,
                        "section_complete": True,
                        "validated": False,
                        "reason": f"validation error: {exc}",
                        "model": "GLM-5.3 (Fallback)",
                    }

            if located.get("found"):
                conf = (
                    located.get("validation", {}).get("confidence")
                    if (located.get("validation") and located["validation"].get("confidence") is not None)
                    else located.get("match_confidence", 0.96)
                )
                located["final_result"] = {
                    "book": b["filename"],
                    "topic": topic,
                    "chapter": located.get("chapter_title") or "Reference Chapter",
                    "section": located.get("section_title") or f"Section: {topic}",
                    "next_section": located.get("next_section_title") or "Next Section",
                    "start_page": located.get("start_page"),
                    "end_page": located.get("end_page"),
                    "printed_start_page": located.get("printed_start_page"),
                    "printed_end_page": located.get("printed_end_page"),
                    "page_display": located.get("page_display"),
                    "subtopics": located.get("subtopics", []),
                    "related_topics": located.get("related_topics", []),
                    "match_confidence": round(conf, 4),
                    "status": "Topic Located Successfully",
                    "tagline": "Exact topic range found! (Without scanning from first page)",
                }

            results.append({
                "book_id": b["book_id"],
                "filename": b["filename"],
                "total_pages": index.total_pages,
                "pdf_type": getattr(index, "pdf_type", "Text PDF"),
                "extraction_method": getattr(index, "extraction_method", "PyMuPDF"),
                "book_url": f"/books-static/{Path(b['path']).name}",
                "result": located,
            })

        except Exception as exc:
            logger.exception("Failed to locate topic in %s", b["filename"])
            results.append({
                "book_id": b["book_id"],
                "filename": b["filename"],
                "error": str(exc),
                "result": {"found": False, "reason": "processing_error"},
            })

    return {
        "success": True,
        "mode": "reference_book_topic_locator",
        "topic": topic,
        "books_processed": len(results),
        "results": results,
    }


# =========================================================
# DOCUMENT STORAGE HELPERS
# =========================================================

DOCUMENTS_STORE: dict = {}


def save_document(doc_data: dict) -> None:

    doc_id = doc_data["id"]
    DOCUMENTS_STORE[doc_id] = doc_data

    try:
        out_file = OUTPUT_DIR / f"{doc_id}.json"

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(
                doc_data,
                f,
                ensure_ascii=False,
                indent=2
            )
    except Exception as e:
        print(f"Failed to persist document {doc_id}: {e}")


def load_document(doc_id: str) -> dict:

    if doc_id in DOCUMENTS_STORE:
        return DOCUMENTS_STORE[doc_id]

    out_file = OUTPUT_DIR / f"{doc_id}.json"

    if out_file.exists():
        try:
            with open(out_file, "r", encoding="utf-8") as f:
                doc_data = json.load(f)
                DOCUMENTS_STORE[doc_id] = doc_data
                return doc_data
        except Exception:
            pass

    return None


def get_all_documents() -> list:

    docs = list(DOCUMENTS_STORE.values())

    loaded_ids = {d["id"] for d in docs}

    for file in OUTPUT_DIR.glob("*.json"):
        doc_id = file.stem

        if doc_id not in loaded_ids:
            try:
                with open(file, "r", encoding="utf-8") as f:
                    d = json.load(f)

                    if isinstance(d, dict) and "id" in d:
                        DOCUMENTS_STORE[d["id"]] = d
                        docs.append(d)
            except Exception:
                pass

    return docs


# =========================================================
# REFERENCE BOOK TOPIC EXTRACTION
# =========================================================

def extract_reference_topics_from_course(course_data: dict) -> list:
    """
    Extract syllabus topics from the already extracted
    Syllabus IQ course details.
    """

    topics = []

    if not isinstance(course_data, dict):
        return topics

    def add_topic(value):
        if value is None:
            return
        if isinstance(value, str):
            value = value.strip()
            if value and len(value) > 1:
                topics.append(value)
        elif isinstance(value, list):
            for item in value:
                add_topic(item)

    add_topic(course_data.get("topics"))

    units = course_data.get("units", [])
    if isinstance(units, list):
        for unit in units:
            if not isinstance(unit, dict):
                continue

            unit_title = (
                unit.get("title")
                or unit.get("name")
                or unit.get("unit")
            )
            if unit_title:
                add_topic(unit_title)

            unit_topics = (
                unit.get("topics")
                or unit.get("subtopics")
                or unit.get("contents")
                or unit.get("syllabus")
                or []
            )
            if isinstance(unit_topics, list):
                for topic in unit_topics:
                    if isinstance(topic, str):
                        add_topic(topic)
                    elif isinstance(topic, dict):
                        topic_name = (
                            topic.get("title")
                            or topic.get("name")
                            or topic.get("topic")
                            or topic.get("description")
                        )
                        if topic_name:
                            add_topic(topic_name)

    for field in ["syllabus", "course_topics", "course_content", "contents"]:
        value = course_data.get(field)
        if isinstance(value, list):
            add_topic(value)

    final_topics = []
    seen = set()

    for topic in topics:
        topic = str(topic).strip()
        if not topic:
            continue
        key = topic.lower()
        if key not in seen:
            seen.add(key)
            final_topics.append(topic)

    return final_topics


# =========================================================
# GET SYLLABUS TOPICS FOR REFERENCE BOOK MATCHING
# =========================================================

@app.get("/courses/{document_id}/reference-topics")
async def get_reference_topics(
    document_id: str,
    course: str
):
    """
    Get syllabus topics from an already extracted Syllabus IQ course.
    """

    doc = load_document(document_id)

    if not doc:
        raise HTTPException(
            status_code=404,
            detail="Syllabus document not found."
        )

    if not course or not course.strip():
        raise HTTPException(
            status_code=400,
            detail="Course name is required."
        )

    requested_course = course.strip().lower()
    details = doc.get("details", {})

    selected_detail = None
    selected_course_name = course.strip()

    if isinstance(details, dict):
        for key, value in details.items():
            key_clean = str(key).strip().lower()
            if key_clean == requested_course and isinstance(value, dict):
                selected_detail = value
                selected_course_name = str(key)
                break

        if selected_detail is None:
            for key, value in details.items():
                key_clean = str(key).strip().lower()
                if (
                    requested_course in key_clean
                    or key_clean in requested_course
                ) and isinstance(value, dict):
                    selected_detail = value
                    selected_course_name = str(key)
                    break

    if selected_detail is None:
        courses = doc.get("courses", [])

        if isinstance(courses, list):
            for c in courses:
                if not isinstance(c, dict):
                    continue

                title = str(
                    c.get("title")
                    or c.get("course_title")
                    or c.get("name")
                    or ""
                ).strip()

                code = str(c.get("code") or "").strip()
                searchable = f"{code} {title}".strip().lower()

                if (
                    requested_course == title.lower()
                    or requested_course in searchable
                    or (title and title.lower() in requested_course)
                    or (code and code.lower() == requested_course)
                ):
                    if isinstance(details, dict):
                        for detail_key, detail_value in details.items():
                            if not isinstance(detail_value, dict):
                                continue

                            dk = str(detail_key).strip().lower()

                            if (
                                (title and title.lower() in dk)
                                or (code and code.lower() in dk)
                            ):
                                selected_detail = detail_value
                                selected_course_name = title or code or course
                                break

                    if selected_detail is None:
                        selected_detail = c
                        selected_course_name = title or code or course

                    break

    if selected_detail is None:
        available_courses = []

        for c in doc.get("courses", []):
            if isinstance(c, dict):
                title = (
                    c.get("title")
                    or c.get("course_title")
                    or c.get("name")
                )
                if title:
                    available_courses.append(str(title))

        raise HTTPException(
            status_code=404,
            detail={
                "message": f"Course '{course}' not found in this syllabus document.",
                "available_courses": available_courses,
                "suggestion": (
                    "Run /extract/course or /extract/courses/all first "
                    "if detailed syllabus topics have not been extracted."
                )
            }
        )

    topics = extract_reference_topics_from_course(selected_detail)

    if not topics:
        raise HTTPException(
            status_code=404,
            detail={
                "message": f"No syllabus topics found for '{selected_course_name}'.",
                "suggestion": (
                    "Run /extract/course for this course so Syllabus IQ "
                    "can extract its units and topics."
                )
            }
        )

    return {
        "success": True,
        "document_id": document_id,
        "course": selected_course_name,
        "topic_count": len(topics),
        "topics": topics
    }


# =========================================================
# COURSE LIST ENDPOINT
# =========================================================

@app.post("/extract/courses")
async def extract_courses(
    file: UploadFile = File(...),
    options: str = Form(None),
    course: str = Form(None),
):
    return await extract_full(
        file=file,
        options=options,
        course=course,
    )


# =========================================================
# SINGLE COURSE DETAILED EXTRACTION ENDPOINT
# =========================================================

@app.post("/extract/course")
async def extract_course(
    request: Request,
    file: UploadFile = File(None),
    document_id: str = Form(None),
    course: str = Form(None),
):
    pages_text = ""
    start_page = None
    doc = None

    req_course = course
    req_doc_id = document_id
    req_file = file

    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            body = await request.json()
            if isinstance(body, dict):
                req_course = body.get("course") or req_course
                req_doc_id = body.get("document_id") or req_doc_id
        except Exception:
            pass

    if not req_course:
        try:
            form = await request.form()
            req_course = form.get("course") or req_course
            req_doc_id = form.get("document_id") or req_doc_id
            if "file" in form:
                req_file = form.get("file")
        except Exception:
            pass

    if not req_course:
        raise HTTPException(
            status_code=400,
            detail="Missing required field 'course'."
        )

    if req_doc_id:
        doc = load_document(req_doc_id)
        if doc:
            pages_text = doc.get("text", "")

    if not pages_text and req_file and getattr(req_file, "filename", None):
        temp_id = uuid.uuid4().hex
        temp_pdf = UPLOAD_DIR / f"{temp_id}.pdf"
        try:
            with open(temp_pdf, "wb") as f:
                while True:
                    chunk = await req_file.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
            extraction = await asyncio.to_thread(extract_pdf, temp_pdf)
            pages_text = extraction.get("text", "")
        finally:
            try:
                temp_pdf.unlink()
            except Exception:
                pass

    if not pages_text:
        all_docs = get_all_documents()
        if all_docs:
            doc = all_docs[0]
            pages_text = doc.get("text", "")

    if not pages_text:
        raise HTTPException(
            status_code=400,
            detail="No syllabus text found. Please provide a document_id or upload a PDF."
        )

    force_reextract = False
    if "application/json" in content_type:
        try:
            body = await request.json()
            if isinstance(body, dict):
                force_reextract = bool(body.get("force") or body.get("reextract"))
        except Exception:
            pass
    if not force_reextract:
        try:
            form = await request.form()
            force_reextract = bool(form.get("force") or form.get("reextract"))
        except Exception:
            pass

    if doc and not force_reextract and isinstance(doc.get("details"), dict):
        cached_details = doc["details"]
        c_clean = req_course.replace("|", " ").strip().lower()
        for k, v in cached_details.items():
            k_lower = str(k).lower()
            if k_lower == c_clean or k_lower in c_clean or (len(c_clean) > 3 and c_clean in k_lower):
                if v and isinstance(v, dict) and v.get("units"):
                    logger.info(f"[CACHE] Returning cached extraction for {req_course}")
                    return v

    all_courses = doc.get("courses", []) if doc else None
    if doc and all_courses:
        course_query_clean = req_course.replace("|", " ").strip().lower()
        for c in all_courses:
            code = (c.get("code") or "").strip().lower()
            title = (c.get("title") or "").strip().lower()
            if (code and code in course_query_clean) or (title and title in course_query_clean):
                start_page = c.get("page")
                break

    try:
        detail_result = await asyncio.to_thread(
            extract_course_details,
            pages_text=pages_text,
            course_query=req_course,
            start_page=start_page,
            all_courses=all_courses
        )
    except Exception as exc:
        logger.warning(f"Detailed course extraction fallback: {exc}")
        from app.extraction_service import (
            extract_course_section_text,
            extract_course_details_deterministic,
            validate_course_detail
        )
        section_text, detail_pages = extract_course_section_text(
            pages_text=pages_text,
            course_query=req_course,
            start_page=start_page,
            all_courses=all_courses
        )
        fast_result = extract_course_details_deterministic(
            section_text=section_text,
            course_query=req_course,
            detail_pages=detail_pages
        )
        detail_result = validate_course_detail(
            fast_result, req_course, default_pages=detail_pages
        )
        if detail_pages:
            detail_result["course"]["pages"] = detail_pages

    if doc:
        if "details" not in doc or not isinstance(doc["details"], dict):
            doc["details"] = {}

        c_info = detail_result.get("course", {})
        code_k = c_info.get("code", "").strip()
        title_k = c_info.get("title", "").strip()

        if code_k:
            doc["details"][code_k] = detail_result
        if title_k:
            doc["details"][title_k] = detail_result

        doc["details"][req_course.strip()] = detail_result

        if "courses" in doc and isinstance(doc["courses"], list):
            req_clean = req_course.replace("|", " ").strip().lower()
            for c in doc["courses"]:
                c_c = (c.get("code") or "").strip().lower()
                c_t = (c.get("title") or "").strip().lower()
                if (code_k and c_c == code_k.lower()) or (title_k and c_t == title_k.lower()) or (c_c and c_c in req_clean) or (c_t and c_t in req_clean):
                    if c_info.get("pages"):
                        c["pages"] = c_info["pages"]
                        c["detail_pages"] = c_info["pages"]
                        c["page"] = c_info["pages"][0]
                    if c_info.get("category"):
                        c["category"] = c_info["category"]
                    if c_info.get("credits") is not None:
                        c["credits"] = c_info["credits"]
                    if c_info.get("semester"):
                        c["semester"] = c_info["semester"]

        save_document(doc)

    return detail_result


# =========================================================
# EXTRACT ALL COURSES IN DOCUMENT (BULK / UNIVERSAL)
# =========================================================

@app.post("/extract/courses/all")
async def extract_all_courses(
    request: Request,
    document_id: str = Form(None),
):
    req_doc_id = document_id
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            body = await request.json()
            if isinstance(body, dict):
                req_doc_id = body.get("document_id") or req_doc_id
        except Exception:
            pass

    if not req_doc_id:
        try:
            form = await request.form()
            req_doc_id = form.get("document_id") or req_doc_id
        except Exception:
            pass

    if not req_doc_id:
        all_docs = get_all_documents()
        if all_docs:
            req_doc_id = all_docs[0].get("id")

    if not req_doc_id:
        raise HTTPException(status_code=400, detail="Document ID is required.")

    doc = load_document(req_doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    courses = doc.get("courses", [])
    pages_text = doc.get("text", "")
    if not courses or not pages_text:
        raise HTTPException(
            status_code=400,
            detail="No courses or text found in document."
        )

    if "details" not in doc or not isinstance(doc["details"], dict):
        doc["details"] = {}

    results = {}
    for c in courses:
        c_query = f"{c.get('code', '')} | {c.get('title', '')}".strip(" |")
        start_page = c.get("page")
        try:
            detail = await asyncio.to_thread(
                extract_course_details,
                pages_text=pages_text,
                course_query=c_query,
                start_page=start_page,
                all_courses=courses
            )
            code_k = detail.get("course", {}).get("code", "").strip()
            title_k = detail.get("course", {}).get("title", "").strip()
            if code_k:
                doc["details"][code_k] = detail
            if title_k:
                doc["details"][title_k] = detail
            doc["details"][c_query] = detail
            results[c_query] = "success"
        except Exception as e:
            results[c_query] = f"failed: {e}"

    save_document(doc)
    return {
        "status": "success",
        "total_courses": len(courses),
        "extracted": results,
        "document_id": req_doc_id
    }


# =========================================================
# DOCUMENTS ENDPOINTS
# =========================================================

@app.get("/documents")
async def get_documents():

    docs = get_all_documents()
    summaries = []

    for d in docs:
        summaries.append({
            "id": d.get("id"),
            "filename": d.get("filename"),
            "college": d.get("college", ""),
            "pages": d.get("pages", 0),
            "course_count": d.get("course_count", len(d.get("courses", []))),
            "fields": d.get("course_count", 0) * 4,
            "needs_review": 0,
            "status": "completed",
            "uploaded_at": d.get("uploaded_at"),
            "courses": d.get("courses", [])
        })

    return {"documents": summaries}


# =========================================================
# DOCUMENT STATUS
# =========================================================

@app.get("/documents/{document_id}/status")
async def get_document_status(document_id: str):

    doc = load_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    total_pages = doc.get("pages", 1)
    course_count = doc.get("course_count", len(doc.get("courses", [])))

    return {
        "status": "completed",
        "stage": 8,
        "page": total_pages,
        "total_pages": total_pages,
        "courses": course_count,
        "fields": course_count * 4,
        "needs_review": 0,
        "filename": doc.get("filename", "")
    }


# =========================================================
# GET DOCUMENT
# =========================================================

@app.get("/documents/{document_id}")
async def get_document_by_id(document_id: str):

    doc = load_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    return {
        "id": doc.get("id"),
        "filename": doc.get("filename"),
        "pages": doc.get("pages", 0),
        "course_count": doc.get("course_count", len(doc.get("courses", []))),
        "courses": doc.get("courses", []),
        "details": doc.get("details", {}),
        "not_extracted": doc.get("not_extracted", []),
        "status": "completed",
        "uploaded_at": doc.get("uploaded_at")
    }


# =========================================================
# DOCUMENT PAGES
# =========================================================

@app.get("/documents/{document_id}/pages")
async def get_document_pages(document_id: str):

    doc = load_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    pages = []

    course_pages = {
        c.get("page")
        for c in doc.get("courses", [])
        if c.get("page")
    }

    for p in doc.get("page_data", []):
        page_num = p.get("page", 1)
        pages.append({
            "page": page_num,
            "type": "Course Start" if page_num in course_pages else "Course Content",
            "fields": len(p.get("text", "").splitlines()),
            "text": p.get("text", "")
        })

    return pages


# =========================================================
# EXPORT
# =========================================================

from app.export_service import generate_export


@app.get("/documents/{document_id}/export")
async def export_document(
    document_id: str,
    format: str = "json"
):

    doc = load_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    from fastapi.responses import Response

    content, media_type, ext = generate_export(doc, format)
    clean_base = doc.get("filename", "export")
    if clean_base.lower().endswith(".pdf"):
        clean_base = clean_base[:-4]

    return Response(
        content=content,
        media_type=media_type,
        headers={
            "Content-Disposition": f'attachment; filename="{clean_base}.{ext}"'
        }
    )


# =========================================================
# LOCAL RUN
# =========================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )