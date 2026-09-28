import fitz

doc = fitz.open('uploads/5589e18f6d184e429571fb5df9f0be69.pdf')

sample_courses = [
    ('SMS 2101', 'Discrete Mathematical Structures'),
    ('CES 2101', 'Data Structures'),
    ('CES 2102', 'Data Communication and Computer Networks'),
    ('CES 2203', 'Introduction to Artificial Intelligence'),
    ('CES 3101', 'Finite Automata and Compiler Design')
]

for code, title in sample_courses:
    print(f"\n=== FINDING PAGES FOR: {code} - {title} ===")
    for idx in range(len(doc)):
        txt = doc[idx].get_text()
        if code in txt or title.upper() in txt.upper():
            preview = " ".join(txt[:120].split())
            print(f"  Page {idx+1}: {preview}")
