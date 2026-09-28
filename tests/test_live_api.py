import requests
import json

def test_live():
    print("Checking /health...")
    try:
        r = requests.get("http://127.0.0.1:8000/health", timeout=5)
        print("Health response:", r.status_code, r.json())
    except Exception as e:
        print("Health check failed:", e)
        return

    print("\nUploading 'CSE_Scheme_2026-2027.pdf' to http://127.0.0.1:8000/extract/full ...")
    try:
        with open("uploads/150d6d74006043c8ab5e6b6bf595b495.pdf", "rb") as f:
            files = {"file": ("CSE_Scheme_2026-2027.pdf", f, "application/pdf")}
            r = requests.post("http://127.0.0.1:8000/extract/full", files=files, timeout=60)
            print("Status Code:", r.status_code)
            data = r.json()
            print("Document ID:", data.get("id"))
            print("Filename:", data.get("filename"))
            print("Total Pages:", data.get("pages"))
            print("Extracted Courses Count:", data.get("course_count"))
            print("\nCourses List Sample (First 15):")
            for idx, c in enumerate(data.get("courses", [])[:15], 1):
                print(f"  {idx}. [{c.get('code')}] {c.get('title')} (Page {c.get('page')})")
    except Exception as e:
        print("Live extraction test failed:", e)

if __name__ == "__main__":
    test_live()
