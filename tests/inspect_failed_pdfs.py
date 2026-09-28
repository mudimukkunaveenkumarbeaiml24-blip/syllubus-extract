import fitz

failed_pdfs = [
    'uploads/0c1959bff1af4ae5ace20586b53fbb28.pdf',
    'uploads/150d6d74006043c8ab5e6b6bf595b495.pdf',
    'uploads/1e5bbc6cdc1447aa95b325456ff54f83.pdf',
    'uploads/542d75443bb248198b1f81610f6a5f7f.pdf',
    'uploads/6bdd5bc8340746a3962d4461e3f4f93e.pdf',
]

for p in failed_pdfs:
    doc = fitz.open(p)
    print("\n" + "="*70)
    print(f"PDF: {p} (Total pages: {len(doc)})")
    for i in range(min(15, len(doc))):
        page_text = doc[i].get_text()
        lines = [l.strip() for l in page_text.splitlines() if l.strip()]
        if lines:
            line_summary = [l.encode('ascii', 'replace').decode('ascii') for l in lines[:4]]
            print(f"  Page {i+1} [lines: {len(lines)}]: {line_summary}")
            for l in lines:
                l_asc = l.encode('ascii', 'replace').decode('ascii')
                if any(k in l.lower() for k in ["course title", "course name", "scheme of", "teaching scheme", "unit i", "unit 1", "module 1", "course outcomes", "course code", "semester", "syllabus"]):
                    print(f"     -> Match: {l_asc[:100]}")
    doc.close()
