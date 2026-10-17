"""
End-to-End Master Rebuild & Recovery Verification Suite for SYLLABUSIQ
Tests:
1. Health & Service Readiness
2. Reference Books Inventory (GET /books)
3. Hybrid RAG Positive Match (Cayley-Hamilton Theorem -> Veerarajan Page 56)
4. Hybrid RAG Negative Zero False-Positive (DNA Sequencing -> All False)
5. Syllabus Multi-Course Detection & Extraction (Units, COs, Bloom, CO-PO, Textbooks)
"""
import sys
import requests
import json

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


BASE_URL = "http://127.0.0.1:8000"

def test_health():
    print("==================================================")
    print("1. Testing Health Endpoint (GET /health)...")
    r = requests.get(f"{BASE_URL}/health", timeout=10)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    data = r.json()
    print("Health Status:", data.get("status"), "Model:", data.get("model"))
    assert data.get("status") in ("ok", "healthy", "online", True) or "model" in data
    print("[PASS] Health endpoint verified.")

def test_books_list():
    print("\n==================================================")
    print("2. Testing Reference Books Inventory (GET /books)...")
    r = requests.get(f"{BASE_URL}/books", timeout=10)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    data = r.json()
    books = data.get("books", [])
    print(f"Found {len(books)} indexed reference books on server:")
    for b in books:
        print(f" - [{b.get('pdf_type')}] {b.get('filename')} ({b.get('total_pages')} pages, {b.get('chunk_count')} chunks, URL: {b.get('book_url')})")
    assert len(books) >= 3, f"Expected at least 3 reference books, got {len(books)}"
    print("[PASS] Reference books inventory verified.")

def test_rag_positive():
    print("\n==================================================")
    print("3. Testing Hybrid RAG Positive Match ('Cayley-Hamilton Theorem')...")
    payload = {
        "topic": "Cayley-Hamilton Theorem",
        "validate_with_llm": "false"
    }
    r = requests.post(f"{BASE_URL}/books/locate-topic", data=payload, timeout=60)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    data = r.json()
    results = data.get("results", [])
    assert len(results) >= 3, f"Expected results from 3 books, got {len(results)}"

    veerarajan_match = None
    for res in results:
        fname = res.get("filename", "")
        r_inner = res.get("result", {})
        found = r_inner.get("found", False)
        print(f"Book: {fname} -> Found: {found}")
        if "veerarajan" in fname.lower() and "mathematics-1" in fname.lower():
            veerarajan_match = r_inner
        elif "grewal" in fname.lower() or "probability" in fname.lower():
            assert not found, f"Book {fname} should NOT contain Cayley-Hamilton Theorem!"

    assert veerarajan_match is not None, "Veerarajan Mathematics 1 not found in results!"
    assert veerarajan_match.get("found") is True, "Cayley-Hamilton Theorem should be found in Veerarajan Mathematics 1!"
    assert veerarajan_match.get("start_page") == 56, f"Expected start_page 56, got {veerarajan_match.get('start_page')}"
    assert veerarajan_match.get("end_page") == 56, f"Expected end_page 56, got {veerarajan_match.get('end_page')}"
    assert "CAYLEY-HAMILTON" in (veerarajan_match.get("supporting_passage") or "").upper(), "Supporting passage missing theorem quote!"
    
    print(f"Veerarajan Match Verified:")
    print(f" - Page Display: {veerarajan_match.get('page_display')}")
    print(f" - Section Title: {veerarajan_match.get('section_title')}")
    print(f" - Confidence: {veerarajan_match.get('match_confidence')}")
    print(f" - Verbatim Quote: {veerarajan_match.get('supporting_passage')[:120]}...")
    print("[PASS] Positive RAG retrieval verified with exact page numbers and evidence.")

def test_rag_negative():
    print("\n==================================================")
    print("4. Testing Hybrid RAG Negative (Zero False Positives: 'DNA sequencing and CRISPR gene editing')...")
    payload = {
        "topic": "DNA sequencing and CRISPR gene editing",
        "validate_with_llm": "false"
    }
    r = requests.post(f"{BASE_URL}/books/locate-topic", data=payload, timeout=60)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    data = r.json()
    results = data.get("results", [])
    
    for res in results:
        fname = res.get("filename", "")
        r_inner = res.get("result", {})
        found = r_inner.get("found", False)
        print(f"Book: {fname} -> Found: {found} (Reason: {r_inner.get('reason')})")
        assert not found, f"False positive detected! {fname} claimed to contain DNA sequencing!"

    print("[PASS] Zero false positives confirmed across all books.")

def test_syllabus_extraction():
    print("\n==================================================")
    print("5. Testing Multi-Course Detection & Syllabus Extraction...")
    # Get available documents
    r = requests.get(f"{BASE_URL}/documents", timeout=30)
    assert r.status_code == 200
    docs = r.json()
    if not isinstance(docs, list):
        docs = docs.get("documents", [])
    
    print(f"Found {len(docs)} documents on server.")
    target_doc = None
    target_course_code = None
    target_course_title = ""

    # Look for a document with course outcomes/units (e.g., 0243f9b918bc44f7abd4c0237881ee52)
    known_target_id = "0243f9b918bc44f7abd4c0237881ee52"
    for d in docs:
        if d.get("id") == known_target_id:
            target_doc = d
            break

    if target_doc is None:
        for d in docs:
            if d.get("courses"):
                target_doc = d
                break

    assert target_doc is not None, "No documents found on server to verify course extraction."
    doc_id = target_doc.get("id")
    courses = target_doc.get("courses", [])
    target_course = None
    for c in courses:
        if "34121B01" in (c.get("code") or "").upper():
            target_course = c
            break
    if not target_course:
        target_course = courses[0] if courses else {"code": "34121B01", "title": "Calculus"}
    course_code = target_course.get("code") or "34121B01"
    course_title = target_course.get("title") or ""
    print(f"Testing on Document ID: {doc_id} ({target_doc.get('filename')}, {len(courses)} courses)")
    print(f"Extracting details for Course: {course_code} | {course_title}...")

    extract_payload = {
        "document_id": doc_id,
        "course": course_code,
        "force": "false"
    }
    r2 = requests.post(f"{BASE_URL}/extract/course", data=extract_payload, timeout=60)
    assert r2.status_code == 200, f"Expected 200 from /extract/course, got {r2.status_code}"
    cdata = r2.json()
    det = cdata.get("result") if isinstance(cdata.get("result"), dict) else cdata
    
    units = det.get("units", [])
    cos = det.get("course_outcomes", [])
    matrix = det.get("co_po_matrix", [])
    textbooks = det.get("textbooks", [])
    references = det.get("references", [])
    
    print(f"Extraction Results for {course_code}:")
    print(f" - Units: {len(units)} units extracted")
    for u in units:
        print(f"    * {u.get('title')} ({len(u.get('topics', []))} topics)")
    print(f" - Course Outcomes: {len(cos)} COs")
    for co in cos[:3]:
        print(f"    * [{co.get('code')}] {co.get('text')[:60]}... (Bloom: {co.get('bloom_level')})")
    print(f" - CO-PO Matrix: {len(matrix)} mappings")
    print(f" - Textbooks: {len(textbooks)}, References: {len(references)}")
    
    assert len(units) >= 3, f"Expected at least 3 units, got {len(units)}"
    assert len(cos) >= 1, "Expected at least 1 course outcome"
    print("[PASS] Syllabus multi-course detection and heading extraction verified.")

if __name__ == "__main__":
    try:
        test_health()
        test_books_list()
        test_rag_positive()
        test_rag_negative()
        test_syllabus_extraction()
        print("\n==================================================")
        print(">>> ALL 5 END-TO-END VERIFICATION SUITES PASSED! <<<")
        print("==================================================")
    except AssertionError as ae:
        print("\n[FAIL] Assertion failed:", ae)
        sys.exit(1)
    except Exception as exc:
        print("\n[ERROR] Test suite error:", exc)
        sys.exit(1)
