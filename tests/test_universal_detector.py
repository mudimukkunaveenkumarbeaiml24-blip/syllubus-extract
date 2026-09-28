import glob
import re
import fitz

def universal_detect_courses(pages_text: str):
    pattern = re.compile(r"===== PAGE (\d+) =====\s*(.*?)(?=(?:===== PAGE \d+ =====|$))", re.DOTALL)
    matches = pattern.findall(pages_text)
    if not matches:
        return []

    detected = []
    seen = set()

    def add_course(code, title, p, category="", credits=None, semester=""):
        code = str(code or "").strip()
        title = str(title or "").strip()
        if not title:
            return
        
        # Clean title
        title = re.sub(r'^(?:COURSE\s*TITLE|SUBJECT\s*NAME|COURSE\s*NAME)\s*[:\-]?\s*', '', title, flags=re.I).strip()
        title = re.split(r'\s+(?:BS|ES|PC|PE|OE|MC|HS|EEC|FC|SDC|\d+\s+\d+\s+\d+\s+\d+)\b', title)[0].strip(' -:|\t')
        title = ' '.join(title.split())
        
        if len(title) < 3 or len(title) > 100:
            return
            
        # Filter false positives
        invalid_titles = [
            "TOTAL", "SEMESTER", "CURRICULUM", "CREDITS", "HOURS", "PERIODS", "THEORY", "LAB",
            "PRACTICAL", "COURSE TITLE", "CATEGORY", "LIST OF ABBREVIATIONS", "VISION", "MISSION",
            "PROGRAM EDUCATIONAL OBJECTIVES", "PROGRAM OUTCOMES", "COURSE SCHEME", "EVALUATION SCHEME",
            "TEACHING SCHEME", "DEPARTMENT ELECTIVES", "MULTIDISCIPLINARY MINORS", "OPEN ELECTIVES",
            "ABBREVIATION", "TITLE", "FIRST SEMESTER", "SECOND SEMESTER", "THIRD SEMESTER",
            "FOURTH SEMESTER", "FIFTH SEMESTER", "SIXTH SEMESTER", "SEVENTH SEMESTER", "EIGHTH SEMESTER"
        ]
        if title.upper() in invalid_titles:
            return

        # Normalized key for deduplication
        norm_code = re.sub(r"[\s\-_<>]", "", code).upper()
        norm_title = re.sub(r"[^A-Za-z0-9]", "", title).upper()
        
        # If code is generic like TBD or empty, use title as main key
        if norm_code in ("", "TBD", "NONE", "NA", "COURSECODE", "SUBCODE", "CODE"):
            dedup_key = ("TITLE", norm_title)
            if not code or code.upper() in ("<TBD>", "TBD", "NONE", "NA"):
                code = category if category else "COURSE"
        else:
            dedup_key = ("CODE_TITLE", norm_code, norm_title)

        if dedup_key in seen:
            return
        seen.add(dedup_key)

        detected.append({
            "code": code,
            "title": title,
            "index_page": p,
            "page": p,
            "category": category,
            "credits": credits,
            "semester": semester
        })

    # -------------------------------------------------------------
    # PASS 1: Syllabus Section Headers (e.g. (PCC) Compiler Construction, CS801 - AI)
    # -------------------------------------------------------------
    cat_regex = r'(?:PCC|PEC|BSC|ESC|OE|MDM|VSEC|HSMC|AEC|IKS|VEC|RM|OJT|CEA|CCA|PC|PE|BS|ES|MC|HS|EEC|OEC|PROJ|PCC-CS|PEC-CS|ESC-CS)'
    
    for p_num, text in matches:
        p = int(p_num)
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        
        # 1.A: Category in parentheses: e.g. "(PCC) Compiler Construction" or "(PEC) System Administration"
        for i, line in enumerate(lines):
            # Match (PCC) Title or PCC: Title or [PCC] Title
            m = re.match(r'^(?:\(|\b)(' + cat_regex + r')(?:\)|\s*[:\-])\s+([A-Za-z][A-Za-z0-9\s,\-\(\)\&/\.]{3,80})$', line, re.I)
            if m:
                cat = m.group(1).upper()
                raw_t = m.group(2).strip()
                # Check if next line has teaching scheme / course outcomes or unit
                has_subsequent = False
                for nxt in lines[i+1:i+6]:
                    if any(k in nxt.lower() for k in ["teaching scheme", "evaluation scheme", "course outcomes", "course objective", "lectures:", "unit", "module", "credits"]):
                        has_subsequent = True
                        break
                if has_subsequent or len(lines) < 60:
                    add_course(code=f"{cat}", title=raw_t, p=p, category=cat)

            # 1.B: Standard Code - Title: e.g. "ACS101 - Principles of Programming" or "CS801: Artificial Intelligence"
            m_code = re.match(r'^([A-Z]{2,6}[-\s]?\d{2,4}[A-Z]?)\s*[\-–:]\s*([A-Za-z][A-Za-z0-9\s,\-\(\)\&/\.]{3,80})$', line)
            if m_code:
                c_code = m_code.group(1).strip()
                c_title = m_code.group(2).strip()
                add_course(code=c_code, title=c_title, p=p)

            # 1.C: "Course Code: CS101" and "Course Title: Intro to Computing" or "Paper Code: ..."
            m_paper = re.match(r'^(?:Paper|Course|Subject)\s*Code\s*[:\-]\s*([A-Z0-9\-_/]+)', line, re.I)
            if m_paper:
                code_val = m_paper.group(1).strip()
                title_val = ""
                # Look for Paper / Course Title on next 3 lines
                for next_l in lines[i:i+4]:
                    m_title = re.match(r'^(?:Paper|Course|Subject)\s*(?:Title|Name)\s*[:\-]\s*([^\n\r]+)', next_l, re.I)
                    if m_title:
                        title_val = m_title.group(1).strip()
                        break
                    elif re.match(r'^(?:Title|Name)\s*[:\-]\s*([^\n\r]+)', next_l, re.I):
                        title_val = re.sub(r'^(?:Title|Name)\s*[:\-]\s*', '', next_l, flags=re.I).strip()
                        break
                if code_val and title_val:
                    add_course(code=code_val, title=title_val, p=p)

        # -------------------------------------------------------------
        # PASS 2: Curriculum / Scheme Table Lines
        # -------------------------------------------------------------
        for i, line in enumerate(lines):
            # Format: S.No Code Title L T P C
            m = re.search(r'(?:^\d+\s+)?\b([A-Z]{2,6}\d{2,4}[A-Z]?)\b(?:\s*[\-:]\s*|\s+)([A-Za-z][A-Za-z0-9\s,\-\(\)\&/\.]{3,80})', line)
            if m:
                code = m.group(1).strip()
                raw_title = m.group(2).strip()
                clean_title = re.split(r'\s+(?:BS|ES|PC|PE|OE|MC|HS|EEC|FC|SDC|\d+\s+\d+\s+\d+\s+\d+)\b', raw_title)[0].strip(' -:|\t')
                if len(clean_title) < 25 and i + 1 < len(lines):
                    next_l = lines[i + 1]
                    if not re.match(r'^(?:BS|ES|PC|PE|OE|MC|HS|EEC|\d+\s+\d+|THEORY|PRACTICALS|\d+\s+[A-Z]{2,5}\d{3,4})\b', next_l):
                        clean_title += ' ' + next_l.strip()
                        clean_title = re.split(r'\s+(?:BS|ES|PC|PE|OE|MC|HS|EEC|FC|SDC|\d+\s+\d+\s+\d+\s+\d+)\b', clean_title)[0].strip(' -:|\t')
                add_course(code=code, title=clean_title, p=p)

        # -------------------------------------------------------------
        # PASS 3: Multi-Line Table Extraction (e.g. S.No\nCategory\nCode\nTitle)
        # -------------------------------------------------------------
        # Look for sequences like: "01\nPCC\n<TBD>\nCompiler Construction" or "VII\nMDM-04\nLarge Language Models..."
        for i in range(len(lines) - 3):
            # Pattern A: Number, Category, Code/<TBD>, Course Title
            l0 = lines[i]
            l1 = lines[i+1]
            l2 = lines[i+2]
            l3 = lines[i+3]
            
            if (l0.isdigit() or l0 in ("I", "II", "III", "IV", "V", "VI", "VII", "VIII")) and re.match(r'^' + cat_regex + r'$', l1, re.I):
                # l2 might be <TBD> or code like MDM-04, and l3 is title
                if l2 in ("<TBD>", "TBD", "--", "-") or re.match(r'^[A-Z0-9\-_/]{2,10}$', l2):
                    c_code = l1 if l2 in ("<TBD>", "TBD", "--", "-") else l2
                    c_title = l3
                    # Check if next lines have numbers (L T P C)
                    add_course(code=c_code, title=c_title, p=p, category=l1)
            
            # Pattern B: Semester (e.g. VII), Course Code (e.g. MDM-04), Course Title
            if l0 in ("I", "II", "III", "IV", "V", "VI", "VII", "VIII") and re.match(r'^[A-Z]{2,6}[-\s]?\d{1,4}$', l1, re.I):
                add_course(code=l1, title=l2, p=p, semester=l0)

    return detected

# Test across unique PDFs
seen_sizes = set()
unique_pdfs = []
for p in glob.glob('uploads/*.pdf'):
    doc = fitz.open(p)
    size = len(doc)
    doc.close()
    if size not in seen_sizes:
        seen_sizes.add(size)
        unique_pdfs.append(p)

for pdf_path in unique_pdfs:
    doc = fitz.open(pdf_path)
    pages_text = "\n\n".join([f"===== PAGE {i+1} =====\n{doc[i].get_text()}" for i in range(len(doc))])
    doc.close()
    
    courses = universal_detect_courses(pages_text)
    print(f"\nPDF: {pdf_path} (Pages: {len(fitz.open(pdf_path))}) -> Detected: {len(courses)} courses")
    for c in courses[:6]:
        print(f"   [{c['code']}] {c['title']} (Page {c['index_page']})")
