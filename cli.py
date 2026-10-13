import argparse
import contextlib
import json
import sys
from pathlib import Path

from app.pdf_service import extract_pdf
from app.extraction_service import detect_courses, extract_course_details


def main():
    parser = argparse.ArgumentParser(description="Academic Syllabus Extraction CLI Tool")
    parser.add_argument("--pdf", required=True, help="Path to syllabus PDF file")
    parser.add_argument("--course", required=False, default=None, help="Course code to extract")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        sys.stderr.write(f"Error: PDF file not found at {args.pdf}\n")
        return 1

    # Redirect logging and stdout to stderr so that stdout contains only pure JSON
    with contextlib.redirect_stdout(sys.stderr):
        doc = extract_pdf(str(pdf_path))
        courses = detect_courses(doc.get("text", "")).get("courses", [])

        if args.course is None:
            # Task 1: List courses in the syllabus
            result = {
                "document": pdf_path.name,
                "course_count": len(courses),
                "courses": [
                    {
                        "code": c.get("code", ""),
                        "title": c.get("title", ""),
                        "page": c.get("page", 1)
                    }
                    for c in courses
                ]
            }
        else:
            # Task 2: Extract specific course
            target_norm = "".join(args.course.split()).upper()
            matched = any("".join((c.get("code") or "").split()).upper() == target_norm for c in courses)
            
            if not matched:
                sys.stderr.write(f"Course not found: {args.course}\n")
                return 1

            result = extract_course_details(
                doc.get("text", ""),
                args.course,
                all_courses=courses
            )

    # Print final result strictly to stdout
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
