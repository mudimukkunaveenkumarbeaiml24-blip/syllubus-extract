from fastapi.testclient import TestClient
from app.main import app
import time

def main():
    client = TestClient(app)
    
    print("Extracting 'CSE_Scheme_2026-2027.pdf' full document...")
    with open('uploads/150d6d74006043c8ab5e6b6bf595b495.pdf', 'rb') as f:
        r = client.post('/extract/full', files={'file': ('CSE_Scheme_2026-2027.pdf', f, 'application/pdf')})
    
    data = r.json()
    doc_id = data.get('id')
    print(f"Document ID: {doc_id}, Detected courses: {data.get('course_count')}")
    
    # Extract details for a specific course
    target_course = "26CE101T | Engineering Chemistry"
    print(f"\nRequesting detailed extraction for: '{target_course}'...")
    t1 = time.time()
    r_detail = client.post('/extract/course', data={'document_id': doc_id, 'course': target_course})
    t2 = time.time()
    
    print(f"Detail extraction status: {r_detail.status_code} in {t2 - t1:.2f}s")
    detail_data = r_detail.json()
    c = detail_data.get('course', {})
    print(f"Course: {c.get('code')} - {c.get('title')}")
    print(f"Credits: {c.get('credits')}, L: {c.get('L')}, T: {c.get('T')}, P: {c.get('P')}")
    print(f"Pages: {c.get('pages')}")
    print(f"Units: {len(detail_data.get('units', []))}, COs: {len(detail_data.get('course_outcomes', []))}")

if __name__ == '__main__':
    main()
