import json
import re

with open('outputs/45fc44e2835f46ab953869cde36014b7.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

print("Searching for course patterns in all 48 pages...")
for p in d['page_data']:
    p_num = p['page']
    text = p['text']
    
    # Pattern 1: (PCC) Title or (PEC) Title or (MDM) Title
    m1 = re.findall(r'(\((?:PCC|PEC|BSC|ESC|OE|MDM|VSEC|HSMC|AEC|IKS|VEC|RM|OJT|CEA|CCA|PC|PE|BS|ES|MC|HS|EEC)\)[^\n\r]+)', text)
    if m1:
        print(f"Page {p_num} (Category parens):", m1)
        
    # Pattern 2: Course Code with Title (e.g. CS101 - Title, or MDM-04 Title)
    m2 = re.findall(r'(?:^|\n)\s*([A-Z]{2,6}[-\s]?\d{1,4}[A-Z]?)\s*[\-:]\s*([^\n\r]{3,60})', text)
    if m2:
        print(f"Page {p_num} (Code-Title):", m2)
        
    # Pattern 3: Teaching scheme / Evaluation scheme headers
    if "teaching scheme" in text.lower() or "evaluation scheme" in text.lower() or "course outcomes" in text.lower():
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        print(f"Page {p_num} (Course header candidate):", lines[:3])
