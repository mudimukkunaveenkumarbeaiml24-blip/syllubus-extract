import fitz

target_pdfs = [
    'uploads/1e5bbc6cdc1447aa95b325456ff54f83.pdf',
    'uploads/542d75443bb248198b1f81610f6a5f7f.pdf',
    'uploads/6bdd5bc8340746a3962d4461e3f4f93e.pdf',
]

for p in target_pdfs:
    doc = fitz.open(p)
    print(f"\n==================== {p} (pages: {len(doc)}) ====================")
    # Check table of contents or pages 7-15
    for page_idx in range(min(15, len(doc))):
        text = doc[page_idx].get_text()
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        for l in lines:
            if any(k in l.lower() for k in ["sub code", "course code", "paper code", "unit 1", "unit i", "teaching scheme", "syllabus"]):
                print(f"  [P{page_idx+1}] {l[:100]}")
    # Also check a syllabus detail page (e.g. page 16 to 25)
    for page_idx in range(15, min(25, len(doc))):
        text = doc[page_idx].get_text()
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if lines:
            print(f"  [Detail P{page_idx+1}] top 3 lines: {lines[:3]}")
