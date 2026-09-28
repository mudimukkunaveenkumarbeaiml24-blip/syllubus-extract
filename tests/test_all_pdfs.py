import glob
import os
import fitz
from app.extraction_service import detect_courses, extract_course_details
from app.pdf_service import extract_pdf

seen_sizes = set()
unique_pdfs = []
for p in glob.glob('uploads/*.pdf'):
    size = os.path.getsize(p)
    if size not in seen_sizes:
        seen_sizes.add(size)
        doc = fitz.open(p)
        unique_pdfs.append((p, len(doc)))
        doc.close()

print(f"Testing dynamic extraction across {len(unique_pdfs)} distinct PDFs:")
print("=" * 70)

for p, pages in sorted(unique_pdfs, key=lambda x: x[1]):
    ext = extract_pdf(p)
    res = detect_courses(ext['text'])
    courses = res['courses']
    print(f"\n>>> PDF: {p} ({pages} pages) -> Detected: {len(courses)} courses")
    for c in courses[:4]:
        print(f"     [{c.get('code', '')}] {c.get('title', '')} (page {c.get('page', 1)})")
