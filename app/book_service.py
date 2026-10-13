# app/book_service.py

import re
from pathlib import Path
from typing import List, Dict, Any, Tuple

import fitz


# =========================================================
# NORMALIZATION
# =========================================================

def normalize_text(text: str) -> str:
    if not text:
        return ""

    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def clean_heading(text: str) -> str:
    text = normalize_text(text)

    text = re.sub(
        r"[\s:;,.]+$",
        "",
        text
    )

    return text.strip()


# =========================================================
# BAD HEADING FILTER
# =========================================================

BAD_PATTERNS = [
    r"^\d+(\.\d+)?\s*(mm|cm|m|km|kg|g|mg|ml|l)$",
    r"^\d+(\.\d+)?\s*%$",
    r"^[a-zA-Z]\s*=\s*.+$",
    r"^(sin|cos|tan|cot|sec|cosec)\s*x$",
    r"^\d+$",
    r"^\d+\s*(x|×|÷)\s*\d+$",
    r".*@.*\..*",
    r"^https?://",
    r"^page\s+\d+$",
    r".*\b(road|street|suite|avenue|burlington|usa)\b.*",
    r".*\bisbn\b.*",
]


def is_bad_heading(text: str) -> bool:

    text = clean_heading(text)

    if not text:
        return True

    if len(text) < 3:
        return True

    if len(text) > 180:
        return True

    for pattern in BAD_PATTERNS:
        if re.search(
            pattern,
            text,
            re.IGNORECASE
        ):
            return True

    return False


# =========================================================
# HEADING TYPE
# =========================================================

KNOWN_MAJOR_TOPICS = [
    "algebra",
    "partial fractions",
    "logarithms",
    "exponential functions",
    "hyperbolic functions",
    "trigonometry",
    "complex numbers",
    "matrices",
    "determinants",
    "vectors",
    "differentiation",
    "integration",
    "differential equations",
    "ordinary differential equations",
    "partial differentiation",
    "multiple integrals",
    "vector calculus",
    "statistics",
    "probability",
    "random variables",
    "sampling theory",
    "correlation",
    "regression",
    "laplace transforms",
    "fourier series",
    "fourier transforms",
    "numerical methods",
]


def classify_heading(text: str) -> str:

    text = clean_heading(text)
    lower = text.lower()

    # Explicit chapter
    if re.match(
        r"^(chapter|unit)\s+\d+",
        lower
    ):
        return "chapter"

    # Known major subjects
    for topic in KNOWN_MAJOR_TOPICS:

        if lower == topic:
            return "chapter"

        if lower.startswith(
            topic + " "
        ):
            return "chapter"

    # Numbered chapter
    if re.match(
        r"^\d{1,2}\s+[A-Z][A-Za-z]",
        text
    ):
        return "chapter"

    # Numbered section
    if re.match(
        r"^\d{1,2}\.\d{1,2}\s+",
        text
    ):
        return "section"

    return "topic"


# =========================================================
# PDF PAGE EXTRACTION
# =========================================================

def extract_book_pages(
    pdf_path: str
) -> Dict[str, Any]:

    pdf_path = str(pdf_path)

    pages = []

    with fitz.open(pdf_path) as doc:

        total_pages = len(doc)

        pymupdf_pages = 0
        ocr_pages = 0
        failed_pages = 0

        for page_index in range(
            total_pages
        ):

            page_number = page_index + 1

            try:

                page = doc.load_page(
                    page_index
                )

                text = page.get_text(
                    "text"
                )

                text = normalize_text(
                    text
                )

                if len(text) >= 30:

                    method = "pymupdf"

                    pymupdf_pages += 1

                else:

                    method = "ocr_needed"

                    ocr_pages += 1

                pages.append({
                    "page": page_number,
                    "text": text,
                    "method": method
                })

            except Exception:

                failed_pages += 1

                pages.append({
                    "page": page_number,
                    "text": "",
                    "method": "failed"
                })

    return {
        "total_pages": total_pages,
        "pymupdf_pages": pymupdf_pages,
        "ocr_pages": ocr_pages,
        "failed_pages": failed_pages,
        "pages": pages
    }


# =========================================================
# PDF BOOKMARK / TOC EXTRACTION
# =========================================================

def extract_pdf_toc(
    pdf_path: str
) -> List[Dict[str, Any]]:

    toc_items = []

    try:

        with fitz.open(pdf_path) as doc:

            toc = doc.get_toc(
                simple=True
            )

            for item in toc:

                if len(item) < 3:
                    continue

                level = item[0]
                title = clean_heading(
                    str(item[1])
                )
                page = int(item[2])

                if is_bad_heading(
                    title
                ):
                    continue

                heading_type = (
                    "chapter"
                    if level <= 1
                    else "section"
                )

                toc_items.append({
                    "page": page,
                    "heading": title,
                    "type": heading_type,
                    "confidence": 0.98,
                    "source": "pdf_toc",
                    "level": level
                })

    except Exception:
        pass

    return toc_items


# =========================================================
# TEXT HEADING DETECTION
# =========================================================

def detect_text_headings(
    pages: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:

    headings = []

    for page_data in pages:

        page_number = page_data[
            "page"
        ]

        text = page_data.get(
            "text",
            ""
        )

        if not text:
            continue

        raw_lines = text.splitlines()

        for raw_line in raw_lines:

            line = clean_heading(
                raw_line
            )

            if is_bad_heading(
                line
            ):
                continue

            words = line.split()

            # Avoid paragraphs
            if len(words) > 18:
                continue

            alpha_count = sum(
                char.isalpha()
                for char in line
            )

            if alpha_count < 4:
                continue

            # Numbered sections
            numbered_section = re.match(
                r"^\d{1,2}\.\d{1,2}\s+.+",
                line
            )

            # Chapter heading
            chapter_heading = re.match(
                r"^(chapter|unit)\s+\d+",
                line,
                re.IGNORECASE
            )

            # Known topic
            known_topic = any(
                topic in line.lower()
                for topic
                in KNOWN_MAJOR_TOPICS
            )

            if not (
                numbered_section
                or chapter_heading
                or known_topic
            ):
                continue

            heading_type = classify_heading(
                line
            )

            confidence = 0.75

            if chapter_heading:
                confidence = 0.90

            elif numbered_section:
                confidence = 0.82

            elif known_topic:
                confidence = 0.85

            headings.append({
                "page": page_number,
                "heading": line,
                "type": heading_type,
                "confidence": confidence,
                "source": "page_text"
            })

    return headings


# =========================================================
# COMBINE TOC + TEXT HEADINGS
# =========================================================

def detect_headings(
    pages: List[Dict[str, Any]],
    pdf_path: str = None
) -> List[Dict[str, Any]]:

    all_headings = []

    # -----------------------------------------------------
    # 1. PDF TOC
    # -----------------------------------------------------

    if pdf_path:

        toc_headings = extract_pdf_toc(
            pdf_path
        )

        all_headings.extend(
            toc_headings
        )

    # -----------------------------------------------------
    # 2. Page text
    # -----------------------------------------------------

    text_headings = detect_text_headings(
        pages
    )

    all_headings.extend(
        text_headings
    )

    # -----------------------------------------------------
    # 3. Remove duplicates
    # -----------------------------------------------------

    unique = {}

    for heading in all_headings:

        key = (
            heading["page"],
            heading["heading"].lower()
        )

        if key not in unique:

            unique[key] = heading

        else:

            # Keep higher confidence
            if (
                heading["confidence"]
                >
                unique[key]["confidence"]
            ):

                unique[key] = heading

    headings = list(
        unique.values()
    )

    headings.sort(
        key=lambda x: (
            x["page"],
            -x["confidence"]
        )
    )

    return headings


# =========================================================
# MATCHING
# =========================================================

def normalize_for_matching(
    text: str
) -> str:

    text = text.lower()

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def keyword_tokens(
    text: str
) -> List[str]:

    stop_words = {
        "the",
        "and",
        "of",
        "to",
        "in",
        "for",
        "by",
        "a",
        "an",
        "on",
        "with",
        "from",
        "using",
        "unit",
        "chapter",
    }

    words = normalize_for_matching(
        text
    ).split()

    return list(dict.fromkeys([
        word
        for word in words
        if len(word) >= 3
        and word not in stop_words
    ]))


def calculate_match_score(
    syllabus_topic: str,
    heading: str
) -> float:

    topic = normalize_for_matching(
        syllabus_topic
    )

    candidate = normalize_for_matching(
        heading
    )

    if not topic or not candidate:
        return 0.0

    # Exact
    if topic == candidate:
        return 1.0

    # Phrase inside heading
    if topic in candidate:
        return 0.95

    topic_words = set(
        keyword_tokens(topic)
    )

    heading_words = set(
        keyword_tokens(candidate)
    )

    if not topic_words:
        return 0.0

    common = (
        topic_words
        &
        heading_words
    )

    score = (
        len(common)
        /
        len(topic_words)
    )

    if topic_words.issubset(
        heading_words
    ):
        score += 0.20

    return min(
        score,
        0.99
    )


# =========================================================
# FIND TOPIC MATCHES
# =========================================================

def find_topic_matches(
    syllabus_topic: str,
    headings: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:

    matches = []

    for heading in headings:

        score = calculate_match_score(
            syllabus_topic,
            heading["heading"]
        )

        if score < 0.30:
            continue

        final_score = (
            score * 0.75
            +
            heading["confidence"]
            * 0.25
        )

        matches.append({
            "page": heading["page"],
            "heading": heading["heading"],
            "type": heading["type"],
            "source": heading.get(
                "source"
            ),
            "match_score": round(
                final_score,
                3
            )
        })

    matches.sort(
        key=lambda x: x[
            "match_score"
        ],
        reverse=True
    )

    return matches


# =========================================================
# PAGE RANGE
# =========================================================

def calculate_page_range(
    start_page: int,
    headings: List[Dict[str, Any]],
    total_pages: int
) -> Tuple[int, int]:

    next_chapters = []

    for heading in headings:

        if heading["page"] <= start_page:
            continue

        if heading["type"] == "chapter":

            next_chapters.append(
                heading["page"]
            )

    if next_chapters:

        next_page = min(
            next_chapters
        )

        end_page = next_page - 1

    else:

        end_page = total_pages

    if end_page < start_page:
        end_page = start_page

    return (
        start_page,
        end_page
    )


# =========================================================
# SINGLE BOOK
# =========================================================

def analyze_single_book(
    book: Dict[str, Any],
    syllabus_topics: List[str]
) -> Dict[str, Any]:

    pdf_path = book["path"]

    extraction = extract_book_pages(
        pdf_path
    )

    pages = extraction[
        "pages"
    ]

    headings = detect_headings(
        pages,
        pdf_path
    )

    topic_results = []

    for syllabus_topic in syllabus_topics:

        matches = find_topic_matches(
            syllabus_topic,
            headings
        )

        if not matches:

            topic_results.append({
                "syllabus_topic":
                    syllabus_topic,

                "matched": False,

                "best_match": None,

                "all_matches": []
            })

            continue

        best = matches[0]

        start_page, end_page = (
            calculate_page_range(
                best["page"],
                headings,
                extraction[
                    "total_pages"
                ]
            )
        )

        confidence = round(
            best["match_score"] * 100
        )

        topic_results.append({

            "syllabus_topic":
                syllabus_topic,

            "matched": True,

            "best_match": {

                "heading":
                    best["heading"],

                "start_page":
                    start_page,

                "end_page":
                    end_page,

                "confidence":
                    confidence,

                "match_score":
                    best["match_score"],

                "type":
                    best["type"],

                "source":
                    best["source"]
            },

            "all_matches": [

                {
                    **match,

                    "confidence":
                        round(
                            match[
                                "match_score"
                            ] * 100
                        )
                }

                for match
                in matches[:10]
            ]
        })

    return {

        "book_id":
            book.get("book_id"),

        "filename":
            book.get(
                "filename",
                Path(
                    pdf_path
                ).name
            ),

        "total_pages":
            extraction[
                "total_pages"
            ],

        "pymupdf_pages":
            extraction[
                "pymupdf_pages"
            ],

        "ocr_pages":
            extraction[
                "ocr_pages"
            ],

        "failed_pages":
            extraction[
                "failed_pages"
            ],

        "headings_detected":
            len(headings),

        "headings":
            headings,

        "topics":
            topic_results
    }


# =========================================================
# COMPARE BOOKS
# =========================================================

def compare_book_results(
    book_results: List[Dict[str, Any]],
    syllabus_topics: List[str]
) -> List[Dict[str, Any]]:

    comparison = []

    for topic in syllabus_topics:

        references = []

        for book in book_results:

            if "topics" not in book:
                continue

            for result in book[
                "topics"
            ]:

                if result[
                    "syllabus_topic"
                ].lower() != topic.lower():
                    continue

                if not result[
                    "matched"
                ]:
                    continue

                best = result[
                    "best_match"
                ]

                references.append({

                    "book_id":
                        book.get(
                            "book_id"
                        ),

                    "book_name":
                        book.get(
                            "filename"
                        ),

                    "heading":
                        best[
                            "heading"
                        ],

                    "start_page":
                        best[
                            "start_page"
                        ],

                    "end_page":
                        best[
                            "end_page"
                        ],

                    "confidence":
                        best[
                            "confidence"
                        ],

                    "match_score":
                        best[
                            "match_score"
                        ]
                })

        references.sort(
            key=lambda x:
                x["confidence"],
            reverse=True
        )

        comparison.append({

            "syllabus_topic":
                topic,

            "books_found":
                len(references),

            "best_reference":
                (
                    references[0]
                    if references
                    else None
                ),

            "all_references":
                references
        })

    return comparison


# =========================================================
# MULTIPLE BOOKS
# =========================================================

def analyze_multiple_books(
    books: List[Dict[str, Any]],
    syllabus_topics: List[str]
) -> Dict[str, Any]:

    results = []

    completed = 0
    failed = 0

    for book in books:

        try:

            result = analyze_single_book(
                book,
                syllabus_topics
            )

            results.append(
                result
            )

            completed += 1

        except Exception as exc:

            failed += 1

            results.append({

                "book_id":
                    book.get(
                        "book_id"
                    ),

                "filename":
                    book.get(
                        "filename"
                    ),

                "error":
                    str(exc)
            })

    comparison = compare_book_results(
        results,
        syllabus_topics
    )

    return {

        "books_processed":
            len(books),

        "books_completed":
            completed,

        "books_failed":
            failed,

        "syllabus_topics":
            syllabus_topics,

        "books":
            results,

        "comparison":
            comparison
    }


# =========================================================
# PUBLIC API
# =========================================================

def analyze_books(
    books: List[Dict[str, Any]],
    syllabus_topics: List[str]
) -> Dict[str, Any]:

    if not books:
        raise ValueError(
            "At least one reference book is required."
        )

    if not syllabus_topics:
        raise ValueError(
            "At least one syllabus topic is required."
        )

    return analyze_multiple_books(
        books,
        syllabus_topics
    )