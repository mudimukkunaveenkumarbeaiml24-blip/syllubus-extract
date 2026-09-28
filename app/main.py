import asyncio
import json
import logging
import os
from pathlib import Path
import sys
import uuid

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
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates

from app.pdf_service import extract_pdf
from app.extraction_service import detect_courses, extract_course_details
from app.nvidia_client import RateLimitException
from app.config import NVIDIA_MODEL


# =========================================================
# DIRECTORIES
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TEMPLATES_DIR = BASE_DIR / "templates"
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# FASTAPI
# =========================================================

app = FastAPI(
    title="SYLLABUSIQ",
    description="Universal Syllabus Intelligence API",
    version="1.0.0"
)

templates = Jinja2Templates(
    directory=str(TEMPLATES_DIR)
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
# HOME
# =========================================================

from fastapi.responses import HTMLResponse, JSONResponse
from app.nvidia_client import RateLimitException

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
                batch_size = max(
                    1,
                    int(opts_dict["batch_size"])
                )

        except Exception:
            pass


    # -----------------------------------------------------
    # CREATE DOCUMENT ID
    # -----------------------------------------------------

    document_id = uuid.uuid4().hex

    safe_filename = "{}.pdf".format(
        document_id
    )

    pdf_path = UPLOAD_DIR / safe_filename


    # -----------------------------------------------------
    # SAVE UPLOADED PDF
    # -----------------------------------------------------

    try:

        with open(
            pdf_path,
            "wb"
        ) as output_file:

            while True:

                chunk = await file.read(
                    1024 * 1024
                )

                if not chunk:
                    break

                output_file.write(
                    chunk
                )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail="Could not save PDF: {}".format(
                exc
            )
        )


    # -----------------------------------------------------
    # PDF EXTRACTION
    # -----------------------------------------------------

    try:

        extraction = await asyncio.to_thread(
            extract_pdf,
            pdf_path
        )

    except Exception as exc:

        try:
            pdf_path.unlink()
        except Exception:
            pass

        raise HTTPException(
            status_code=500,
            detail="PDF extraction failed: {}".format(
                exc
            )
        )


    # -----------------------------------------------------
    # GLM COURSE DETECTION
    # SINGLE REQUEST
    # -----------------------------------------------------

    try:

        course_result = await asyncio.to_thread(
            detect_courses,
            extraction["text"]
        )

        # IMPORTANT:
        # Get the courses from the LLM result.
        courses = course_result.get(
            "courses",
            []
        )

        # Get information that could not be extracted.
        not_extracted = course_result.get(
            "not_extracted",
            []
        )

    except Exception as exc:
        logger.warning(f"Detection warning: {exc}, running deterministic fallback")
        from app.extraction_service import detect_courses_deterministic, deduplicate_courses, locate_course_boundaries
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

        "pages": extraction[
            "total_pages"
        ],

        "ocr_pages": extraction[
            "ocr_pages"
        ],

        "unreadable_pages": extraction[
            "unreadable_pages"
        ],

        "course_count": len(
            courses
        ),

        "courses": courses,

        "not_extracted": not_extracted,

        "page_data": extraction[
            "pages"
        ],

        "text": extraction[
            "text"
        ],

        "uploaded_at":
            datetime.utcnow().isoformat()
            + "Z"
    }


    # -----------------------------------------------------
    # PERSIST DOCUMENT
    # -----------------------------------------------------

    save_document(
        response_data
    )


    return response_data


# =========================================================
# DOCUMENT STORAGE HELPERS
# =========================================================

DOCUMENTS_STORE: dict = {}


def save_document(
    doc_data: dict
) -> None:

    doc_id = doc_data["id"]

    DOCUMENTS_STORE[
        doc_id
    ] = doc_data

    try:

        out_file = (
            OUTPUT_DIR
            / f"{doc_id}.json"
        )

        with open(
            out_file,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                doc_data,
                f,
                ensure_ascii=False,
                indent=2
            )

    except Exception as e:

        print(
            f"Failed to persist document "
            f"{doc_id}: {e}"
        )


def load_document(
    doc_id: str
) -> dict:

    if doc_id in DOCUMENTS_STORE:

        return DOCUMENTS_STORE[
            doc_id
        ]

    out_file = (
        OUTPUT_DIR
        / f"{doc_id}.json"
    )

    if out_file.exists():

        try:

            with open(
                out_file,
                "r",
                encoding="utf-8"
            ) as f:

                doc_data = json.load(f)

                DOCUMENTS_STORE[
                    doc_id
                ] = doc_data

                return doc_data

        except Exception:

            pass

    return None


def get_all_documents() -> list:

    docs = list(
        DOCUMENTS_STORE.values()
    )

    loaded_ids = {
        d["id"]
        for d in docs
    }

    for file in OUTPUT_DIR.glob(
        "*.json"
    ):

        doc_id = file.stem

        if doc_id not in loaded_ids:

            try:

                with open(
                    file,
                    "r",
                    encoding="utf-8"
                ) as f:

                    d = json.load(f)

                    if (
                        isinstance(d, dict)
                        and "id" in d
                    ):

                        DOCUMENTS_STORE[
                            d["id"]
                        ] = d

                        docs.append(d)

            except Exception:

                pass

    return docs


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

    # -----------------------------------------------------
    # CHECK CACHE FIRST (Process Once, Extract On Demand)
    # -----------------------------------------------------
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
        detail_result = validate_course_detail(fast_result, req_course, default_pages=detail_pages)
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

        # Update course entry in doc["courses"]
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
        raise HTTPException(status_code=400, detail="No courses or text found in document.")

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
    return {"status": "success", "total_courses": len(courses), "extracted": results, "document_id": req_doc_id}


# =========================================================
# DOCUMENTS ENDPOINTS
# =========================================================

@app.get("/documents")
async def get_documents():

    docs = get_all_documents()

    summaries = []

    for d in docs:

        summaries.append({

            "id": d.get(
                "id"
            ),

            "filename": d.get(
                "filename"
            ),

            "college": d.get(
                "college",
                ""
            ),

            "pages": d.get(
                "pages",
                0
            ),

            "course_count": d.get(
                "course_count",
                len(
                    d.get(
                        "courses",
                        []
                    )
                )
            ),

            "fields": d.get(
                "course_count",
                0
            ) * 4,

            "needs_review": 0,

            "status": "completed",

            "uploaded_at": d.get(
                "uploaded_at"
            ),

            "courses": d.get(
                "courses",
                []
            )

        })

    return {
        "documents": summaries
    }


# =========================================================
# DOCUMENT STATUS
# =========================================================

@app.get(
    "/documents/{document_id}/status"
)
async def get_document_status(
    document_id: str
):

    doc = load_document(
        document_id
    )

    if not doc:

        raise HTTPException(
            status_code=404,
            detail="Document not found"
        )

    total_pages = doc.get(
        "pages",
        1
    )

    course_count = doc.get(
        "course_count",
        len(
            doc.get(
                "courses",
                []
            )
        )
    )

    return {

        "status": "completed",

        "stage": 8,

        "page": total_pages,

        "total_pages": total_pages,

        "courses": course_count,

        "fields": course_count * 4,

        "needs_review": 0,

        "filename": doc.get(
            "filename",
            ""
        )
    }


# =========================================================
# GET DOCUMENT
# =========================================================

@app.get(
    "/documents/{document_id}"
)
async def get_document_by_id(
    document_id: str
):

    doc = load_document(
        document_id
    )

    if not doc:

        raise HTTPException(
            status_code=404,
            detail="Document not found"
        )

    return {

        "id": doc.get(
            "id"
        ),

        "filename": doc.get(
            "filename"
        ),

        "pages": doc.get(
            "pages",
            0
        ),

        "course_count": doc.get(
            "course_count",
            len(
                doc.get(
                    "courses",
                    []
                )
            )
        ),

        "courses": doc.get(
            "courses",
            []
        ),

        "details": doc.get(
            "details",
            {}
        ),

        "not_extracted": doc.get(
            "not_extracted",
            []
        ),

        "status": "completed",

        "uploaded_at": doc.get(
            "uploaded_at"
        )

    }


# =========================================================
# DOCUMENT PAGES
# =========================================================

@app.get(
    "/documents/{document_id}/pages"
)
async def get_document_pages(
    document_id: str
):

    doc = load_document(
        document_id
    )

    if not doc:

        raise HTTPException(
            status_code=404,
            detail="Document not found"
        )

    pages = []

    course_pages = {
        c.get("page")
        for c in doc.get(
            "courses",
            []
        )
        if c.get("page")
    }

    for p in doc.get(
        "page_data",
        []
    ):

        page_num = p.get(
            "page",
            1
        )

        pages.append({

            "page": page_num,

            "type":
                "Course Start"
                if page_num in course_pages
                else "Course Content",

            "fields": len(
                p.get(
                    "text",
                    ""
                ).splitlines()
            ),

            "text": p.get(
                "text",
                ""
            )

        })

    return pages


# =========================================================
# EXPORT
# =========================================================

from app.export_service import generate_export

@app.get(
    "/documents/{document_id}/export"
)
async def export_document(
    document_id: str,
    format: str = "json"
):

    doc = load_document(
        document_id
    )

    if not doc:
        raise HTTPException(
            status_code=404,
            detail="Document not found"
        )

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