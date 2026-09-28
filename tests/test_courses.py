import fitz
from app.extraction_service import locate_course_boundaries, extract_course_details_deterministic, validate_course_detail

doc = fitz.open('uploads/5589e18f6d184e429571fb5df9f0be69.pdf')
txt = '\n\n'.join([f'===== PAGE {i+1} =====\n{doc[i].get_text()}' for i in range(len(doc))])

for code, title, initial_page in [
    ('SMS 2101', 'Discrete Mathematical Structures', 2),
    ('CES 2101', 'Data Structures', 2),
    ('CES 2203', 'Introduction to Artificial Intelligence', 2),
    ('CES 3101', 'Finite Automata and Compiler Design', 2)
]:
    print("=" * 70)
    print(f"Testing: {code} - {title}")
    idx, pages, section = locate_course_boundaries(txt, code, title, initial_page)
    print(f"Index Page: {idx}, Detail Pages: {pages}")
    
    res = extract_course_details_deterministic(section, f"{code} | {title}", pages)
    print(f"Course: {res['course']['code']} - {res['course']['title']}")
    print(f"Credits: {res['course']['credits']}, L: {res['course']['L']}, T: {res['course']['T']}, P: {res['course']['P']}")
    print(f"Units count: {len(res['units'])}")
    for u in res['units']:
        print(f"  {u['unit']} - {u['title']} (topics: {len(u['topics'])})")
        for t in u['topics'][:2]:
            print(f"     * {t['text']}")
    print(f"Books/References count: {len(res['books'])}")
    for b in res['books'][:5]:
        print(f"  {b['number']}. [{b['type']}] {b['title']}")