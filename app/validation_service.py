"""
SYLLABUSIQ - Syllabus Extraction Validation Service
Validates structured syllabus extractions for:
  - Schema integrity and required fields
  - Text fidelity / verbatim source match (anti-hallucination)
  - Bloom's Taxonomy classification validity
  - CO-PO matrix mapping consistency
  - Data completeness and confidence scoring
"""

import re
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("syllabusiq.validation")

# Standard Bloom's Taxonomy Cognitive Levels
VALID_BLOOM_LEVELS = {
    "remember", "understand", "apply", "analyze", "evaluate", "create",
    "k1", "k2", "k3", "k4", "k5", "k6",
    "l1", "l2", "l3", "l4", "l5", "l6"
}


# =========================================================
# STRING NORMALIZATION HELPER
# =========================================================

def _normalize_for_match(text: str) -> str:
    """Normalize text by stripping punctuation and extra whitespace for fuzzy comparison."""
    if not text:
        return ""
    cleaned = re.sub(r'[^a-zA-Z0-9\s]', ' ', str(text).lower())
    return re.sub(r'\s+', ' ', cleaned).strip()


# =========================================================
# UNIT & TOPIC FIDELITY VALIDATOR
# =========================================================

def validate_unit_topics_fidelity(unit: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Validates whether extracted topics appear verbatim in the unit's source text.
    Returns a list of validation results per topic.
    """
    results = []
    unit_text = unit.get("text", "") or ""
    norm_unit_text = _normalize_for_match(unit_text)
    topics = unit.get("topics", [])

    for topic in topics:
        t_id = topic.get("id", "")
        t_text = topic.get("text", "") or ""
        norm_t_text = _normalize_for_match(t_text)

        # Exact match or normalized substring match
        is_exact = t_text in unit_text if (t_text and unit_text) else False
        is_normalized = norm_t_text in norm_unit_text if (norm_t_text and norm_unit_text) else False
        is_valid = is_exact or is_normalized or (not unit_text)

        bloom = str(topic.get("bloom", "")).lower().strip()
        bloom_valid = bloom in VALID_BLOOM_LEVELS or not bloom

        results.append({
            "topic_id": t_id,
            "topic_text": t_text,
            "verbatim_in_unit": is_valid,
            "exact_match": is_exact,
            "bloom_level": bloom,
            "bloom_valid": bloom_valid
        })

    return results


# =========================================================
# CO-PO MATRIX VALIDATOR
# =========================================================

def validate_copo_matrix(
    course_outcomes: List[Dict[str, Any]],
    co_po_matrix: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Validates consistency between defined Course Outcomes and the CO-PO mapping matrix.
    """
    co_ids = {co.get("id") or co.get("code") for co in course_outcomes if (co.get("id") or co.get("code"))}
    matrix_cos = set(co_po_matrix.keys()) if isinstance(co_po_matrix, dict) else set()

    unmapped_cos = list(co_ids - matrix_cos)
    extra_matrix_cos = list(matrix_cos - co_ids) if co_ids else []

    return {
        "is_valid": len(extra_matrix_cos) == 0,
        "co_count": len(co_ids),
        "mapped_co_count": len(matrix_cos),
        "unmapped_cos": unmapped_cos,
        "extra_matrix_cos": extra_matrix_cos
    }


# =========================================================
# COURSE EXTRACTION VALIDATOR
# =========================================================

def validate_course_detail(detail: Dict[str, Any]) -> Dict[str, Any]:
    """
    Performs comprehensive validation on a single course extraction object.
    Calculates completeness score, missing field warnings, and topic fidelity.
    """
    issues = []
    warnings = []
    course_meta = detail.get("course", {})

    # 1. Metadata Checks
    if not course_meta.get("code"):
        warnings.append("Course code is missing or unprinted in the source document.")
    if not course_meta.get("title"):
        issues.append("Course title is missing.")

    # 2. Units & Topics Checks
    units = detail.get("units", [])
    total_topics = 0
    verbatim_topics = 0
    unit_validations = []

    if not units:
        warnings.append("No units or modules were extracted for this course.")
    else:
        for u in units:
            u_num = u.get("number")
            u_title = u.get("title", "")
            if not u_title:
                warnings.append(f"Unit {u_num} is missing a descriptive title.")

            topic_results = validate_unit_topics_fidelity(u)
            unit_validations.append({
                "unit_number": u_num,
                "unit_title": u_title,
                "hours": u.get("hours"),
                "topic_results": topic_results
            })
            total_topics += len(topic_results)
            verbatim_topics += sum(1 for t in topic_results if t["verbatim_in_unit"])

    # 3. Course Outcomes Checks
    cos = detail.get("course_outcomes", [])
    if not cos:
        warnings.append("No course outcomes (COs) were found in the syllabus for this course.")

    # 4. Books Checks
    books = detail.get("books", [])
    if not books and not detail.get("textbooks"):
        warnings.append("No textbooks or reference books were extracted.")

    # 5. Calculate Fidelity & Completeness Scores
    topic_fidelity_pct = round((verbatim_topics / max(total_topics, 1)) * 100, 1)
    
    completeness_factors = [
        bool(course_meta.get("title")),
        bool(course_meta.get("code")),
        len(units) > 0,
        total_topics > 0,
        len(cos) > 0,
        len(books) > 0 or len(detail.get("textbooks", [])) > 0
    ]
    completeness_pct = round((sum(completeness_factors) / len(completeness_factors)) * 100, 1)

    return {
        "status": "verified" if len(issues) == 0 and len(warnings) == 0 else "needs_review",
        "completeness_score": completeness_pct,
        "topic_fidelity_score": topic_fidelity_pct,
        "total_units": len(units),
        "total_topics": total_topics,
        "verbatim_topics_count": verbatim_topics,
        "total_outcomes": len(cos),
        "total_books": len(books) or len(detail.get("textbooks", [])),
        "issues": issues,
        "warnings": warnings,
        "unit_details": unit_validations
    }


# =========================================================
# DOCUMENT-LEVEL VALIDATOR
# =========================================================

def validate_document(doc: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validates an entire syllabus document containing multiple courses.
    """
    courses = doc.get("courses", [])
    details = doc.get("details", {})
    course_reports = []

    total_issues = 0
    total_warnings = 0

    for c in courses:
        code = (c.get("code") or "").strip()
        title = (c.get("title") or "").strip()
        det = details.get(code) or details.get(title) or details.get(f"{code} | {title}".strip())

        if not det:
            for k, v in details.items():
                if code and k.startswith(code):
                    det = v
                    break
                if title and title.lower() in k.lower():
                    det = v
                    break

        if det:
            c_val = validate_course_detail(det)
            course_reports.append({
                "code": code,
                "title": title,
                "validation": c_val
            })
            total_issues += len(c_val["issues"])
            total_warnings += len(c_val["warnings"])
        else:
            course_reports.append({
                "code": code,
                "title": title,
                "validation": {
                    "status": "index_only",
                    "completeness_score": 20.0,
                    "issues": [],
                    "warnings": ["Course details have not been extracted yet (index only)."]
                }
            })
            total_warnings += 1

    overall_status = "verified" if total_issues == 0 and total_warnings == 0 else "needs_review"

    return {
        "document_id": doc.get("id", ""),
        "filename": doc.get("filename", ""),
        "total_courses": len(courses),
        "overall_status": overall_status,
        "total_critical_issues": total_issues,
        "total_warnings": total_warnings,
        "course_validations": course_reports
    }
