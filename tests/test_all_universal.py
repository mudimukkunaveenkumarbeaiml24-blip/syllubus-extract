import glob
import os
import re
import fitz

def universal_detect_courses(pages_text: str):
    pattern = re.compile(r"===== PAGE (\d+) =====\s*(.*?)(?=(?:===== PAGE \d+ =====|$))", re.DOTALL)
    matches = pattern.findall(pages_text)
    if not matches:
        return []

    detected = []
    seen = set()

    def clean_title_str(t):
        t = re.sub(r'^(?:COURSE\s*TITLE|SUBJECT\s*NAME|COURSE\s*NAME|PAPER\s*NAME|PAPER\s*TITLE|PAPER|COURSE|SUBJECT)\s*[:\-]?\s*', '', t, flags=re.I).strip()
        t = re.split(r'\s+(?:BS|ES|PC|PE|OE|MC|HS|EEC|FC|SDC|PCC|PEC|BSC|ESC|HSMC|AEC|IKS|VEC|VSEC|MDM|RM|OJT|\d+\s+\d+\s+\d+\s+\d+|\d+\(\d+[\-\d]*\))\b', t)[0].strip(' -:|\t')
        t = ' '.join(t.split())
        return t

    def is_invalid_title(t):
        if not t or len(t) < 3 or len(t) > 120:
            return True
        t_up = t.upper()
        invalid_exact = {
            'TOTAL', 'SEMESTER', 'CURRICULUM', 'CREDITS', 'HOURS', 'PERIODS', 'THEORY', 'LAB',
            'PRACTICAL', 'COURSE TITLE', 'CATEGORY', 'LIST OF ABBREVIATIONS', 'VISION', 'MISSION',
            'PROGRAM EDUCATIONAL OBJECTIVES', 'PROGRAM OUTCOMES', 'COURSE SCHEME', 'EVALUATION SCHEME',
            'TEACHING SCHEME', 'DEPARTMENT ELECTIVES', 'MULTIDISCIPLINARY MINORS', 'OPEN ELECTIVES',
            'ABBREVIATION', 'TITLE', 'FIRST SEMESTER', 'SECOND SEMESTER', 'THIRD SEMESTER',
            'FOURTH SEMESTER', 'FIFTH SEMESTER', 'SIXTH SEMESTER', 'SEVENTH SEMESTER', 'EIGHTH SEMESTER',
            'EXIT OPTIONS', 'SCHEME OF EXAMINATION', 'PROGRAMME OUTCOMES', 'COURSE OUTCOMES',
            'COURSES', 'ELECTIVES', 'BASKET', 'OPTION A', 'OPTION B', 'OPTION C', 'PREAMBLE', 'PREREQUISITES',
            'SUB CODE', 'SUBJECT CODE', 'COURSE CODE', 'PAPER CODE', 'SR NO', 'SR. NO', 'S.NO',
            'LIST OF COURSES', 'PROGRAM SPECIFIC OUTCOMES', 'MINIMUM PASSING MARKS', 'MID-SEM EXAMINATION',
            'CONTINUAL ASSESSMENT', 'END SEM EXAMINATION', 'END SEMESTER EXAMINATION', 'MAXIMUM MARKS',
            'HOURS PER WEEK', 'L T P', 'CONTACT HOURS', 'EVALUATION SCHEME', 'SCHEME OF TEACHING',
            'MEDIUM', 'HIGH', 'LOW', 'PROGRAM', 'PROGRAMME', 'ACADEMIC YEAR', 'BATCH', 'OBJECTIVES'
        }
        if t_up in invalid_exact:
            return True
        if re.match(r'^(?:SEMESTER\s+[IVX0-9]+|GROUP\s+[IVX0-9]+|BATCH\s+\d+|OPTION\s+[A-Z]|BASKET\s+FOR|SCHEME\s+OF|SPECIALI[SZ]ATION\s+IN|PO\s*\d+|CO\s*\d+|PSO\s*\d+|PEO\s*\d+)\b', t_up):
            return True
        return False

    def is_invalid_code(c):
        if not c:
            return True
        c_up = re.sub(r'[\s\-_]', '', c).upper()
        if re.match(r'^(?:PO\d+|CO\d+|PSO\d+|PEO\d+|EO\d+|SR|SNO|SRNO|NO|TOTAL|CREDITS|HOURS|PAGE|SEM|MARKS|CODE|SUBCODE)$', c_up):
            return True
        return False

    def add_course(code, title, p, category='', credits=None, semester=''):
        code = str(code or '').strip()
        title = clean_title_str(str(title or ''))
        if not title or is_invalid_title(title):
            return

        # Clean code
        code = re.sub(r'[\s_]+', ' ', code).strip()
        if is_invalid_code(code):
            code = ''

        norm_code = re.sub(r'[\s\-_<>]', '', code).upper()
        norm_title = re.sub(r'[^A-Za-z0-9]', '', title).upper()

        if not norm_title or is_invalid_title(title):
            return

        if not norm_code or norm_code in ('TBD', 'NONE', 'NA', 'COURSECODE', 'SUBCODE', 'CODE', 'COURSE'):
            dedup_key = ('TITLE', norm_title)
            if not code or code.upper() in ('<TBD>', 'TBD', 'NONE', 'NA', '-'):
                code = category if category else ''
        else:
            dedup_key = ('CODE_TITLE', norm_code, norm_title)

        if dedup_key in seen:
            return
        seen.add(dedup_key)

        detected.append({
            'code': code,
            'title': title,
            'index_page': p,
            'page': p,
            'category': category,
            'credits': credits,
            'semester': semester
        })

    CODE_REGEX = r'(?:[A-Z0-9]{1,4}[-\s]?)?[A-Z]{2,6}[-\s]?[0-9]{2,4}[A-Z]?'
    cat_regex = r'(?:PCC|PEC|BSC|ESC|OE|MDM|VSEC|HSMC|AEC|IKS|VEC|RM|OJT|CEA|CCA|PC|PE|BS|ES|MC|HS|EEC|OEC|PROJ|PCC-CS|PEC-CS|ESC-CS|HSSM|CEP|SEC|CC)'

    for p_num, text in matches:
        p = int(p_num)
        lines = [l.strip() for l in text.splitlines() if l.strip()]

        # Skip pure PO / PEO definition pages
        p_head = ' '.join(lines[:6]).lower()
        if 'program outcome' in p_head or 'graduate attributes' in p_head or 'vision and mission' in p_head:
            # Check if there are no course codes on this page
            if not re.search(r'(?:course code|paper code|sub code|course title|unit\s*[i|1])', p_head, re.I):
                continue

        # Strategy 1: Explicit Key-Value Header (Paper Code: ... / Course Code: ...)
        for i, line in enumerate(lines):
            m_code_lbl = re.match(r'^(?:Paper|Course|Subject|Sub\.?|Module)\s*Code(?:\(s\))?\s*[:\-]\s*([A-Za-z0-9\s\-_/]+)', line, re.I)
            if m_code_lbl:
                c_code = m_code_lbl.group(1).strip()
                c_title = ''
                # Look in neighboring 4 lines for Title / Paper / Course
                for next_l in lines[max(0, i-2):min(len(lines), i+5)]:
                    m_title_lbl = re.match(r'^(?:Paper|Course|Subject)\s*(?:Title|Name)?\s*[:\-]\s*([^\n\r]+)', next_l, re.I)
                    if m_title_lbl and not re.search(r'Code|Objective|Outcome|Content', next_l, re.I):
                        c_title = m_title_lbl.group(1).strip()
                        break
                    elif re.match(r'^(?:Title|Name)\s*[:\-]\s*([^\n\r]+)', next_l, re.I):
                        c_title = re.sub(r'^(?:Title|Name)\s*[:\-]\s*', '', next_l, flags=re.I).strip()
                        break
                if c_title and not is_invalid_title(c_title):
                    add_course(code=c_code, title=c_title, p=p)

            # Reverse: Course / Paper / Course Title: ... followed by Course Code: ...
            m_title_lbl = re.match(r'^(?:Course|Subject|Paper)\s*(?:Title|Name)?\s*[:\-]\s*([^\n\r]+)', line, re.I)
            if m_title_lbl and not re.search(r'Code|Objective|Outcome|Content|Structure|Component|Category|Scheme', line, re.I):
                c_title = m_title_lbl.group(1).strip()
                c_code = ''
                for next_l in lines[i:min(len(lines), i+5)]:
                    m_code_lbl = re.match(r'^(?:Paper|Course|Subject|Sub\.?)\s*Code(?:\(s\))?\s*[:\-]\s*([A-Za-z0-9\s\-_/]+)', next_l, re.I)
                    if m_code_lbl:
                        c_code = m_code_lbl.group(1).strip()
                        break
                if c_title and not is_invalid_title(c_title):
                    add_course(code=c_code, title=c_title, p=p)

        # Strategy 2: Code - Title on single line (e.g. ACS101 - Principles of Programming)
        for i, line in enumerate(lines):
            m_inline = re.match(r'^(' + CODE_REGEX + r')\s*[\-–:\|]\s*([A-Za-z][A-Za-z0-9\s,\-\(\)\&/\.\']{3,80})$', line)
            if m_inline:
                c_code = m_inline.group(1).strip()
                c_title = m_inline.group(2).strip()
                if not is_invalid_code(c_code) and not is_invalid_title(c_title):
                    add_course(code=c_code, title=c_title, p=p)

            m_cat = re.match(r'^\(?(' + cat_regex + r')\)?\s*[\-–:]?\s+([A-Za-z][A-Za-z0-9\s,\-\(\)\&/\.\']{3,80})$', line, re.I)
            if m_cat:
                cat = m_cat.group(1).upper()
                raw_t = m_cat.group(2).strip()
                if not is_invalid_title(raw_t):
                    add_course(code=cat, title=raw_t, p=p, category=cat)

        # Strategy 3: Horizontal Table Rows
        for i, line in enumerate(lines):
            m_row = re.search(r'(?:^\d+\s+)?(?:(' + cat_regex + r')\s+)?\b(' + CODE_REGEX + r')\b(?:\s*[\-:]\s*|\s+)([A-Za-z][A-Za-z0-9\s,\-\(\)\&/\.\'#]+)', line)
            if m_row:
                cat = m_row.group(1) or ''
                code = m_row.group(2).strip()
                raw_title = m_row.group(3).strip()
                clean_title = clean_title_str(raw_title)

                if is_invalid_code(code) or is_invalid_title(clean_title):
                    continue

                if len(clean_title) < 25 and i + 1 < len(lines):
                    next_l = lines[i + 1]
                    if not re.match(r'^(?:' + cat_regex + r'|\d+\s+\d+|THEORY|PRACTICALS|\d+\s+' + CODE_REGEX + r'|' + CODE_REGEX + r')\b', next_l, re.I):
                        if not re.match(r'^\d+$', next_l) and not is_invalid_title(next_l):
                            clean_title += ' ' + next_l.strip()
                            clean_title = clean_title_str(clean_title)

                add_course(code=code, title=clean_title, p=p, category=cat)

        # Strategy 4: Multi-Line Table Extraction (Vertical Stacked Columns)
        for i in range(len(lines) - 3):
            l0 = lines[i]
            l1 = lines[i+1]
            l2 = lines[i+2]
            l3 = lines[i+3]

            # Case A: S.No -> Category -> Code -> Title
            if (l0.isdigit() or l0 in ('I','II','III','IV','V','VI','VII','VIII')) and re.match(r'^' + cat_regex + r'$', l1, re.I):
                if re.match(r'^' + CODE_REGEX + r'$', l2) or l2 in ('<TBD>', 'TBD', '--', '-'):
                    c_code = l1 if l2 in ('<TBD>', 'TBD', '--', '-') else l2
                    c_title = l3
                    if not is_invalid_code(c_code) and not is_invalid_title(c_title):
                        add_course(code=c_code, title=c_title, p=p, category=l1)

            # Case B: S.No -> Code -> Title
            if l0.isdigit() and re.match(r'^' + CODE_REGEX + r'$', l1):
                if not is_invalid_code(l1) and not is_invalid_title(l2) and re.match(r'^[A-Za-z]', l2):
                    add_course(code=l1, title=l2, p=p)

            # Case C: Code on line i, Title on line i+1
            if re.match(r'^' + CODE_REGEX + r'$', l0) and not is_invalid_title(l1) and re.match(r'^[A-Za-z][A-Za-z0-9\s,\-\(\)\&/\.\']{3,80}$', l1):
                if re.search(r'\d', l0) and not is_invalid_code(l0):
                    add_course(code=l0, title=l1, p=p)

    return detected

if __name__ == "__main__":
    seen_sizes = set()
    unique_pdfs = []
    for p in glob.glob('uploads/*.pdf'):
        size = os.path.getsize(p)
        if size not in seen_sizes:
            seen_sizes.add(size)
            doc = fitz.open(p)
            unique_pdfs.append((p, len(doc)))
            doc.close()

    print(f"Testing across {len(unique_pdfs)} unique syllabus PDFs:")
    print("=" * 70)
    for pdf_path, pages in sorted(unique_pdfs, key=lambda x: x[1]):
        doc = fitz.open(pdf_path)
        pages_text = "\n\n".join([f"===== PAGE {i+1} =====\n{doc[i].get_text()}" for i in range(len(doc))])
        doc.close()
        courses = universal_detect_courses(pages_text)
        print(f"PDF: {pdf_path} ({pages} pages) -> Detected: {len(courses)} courses")
        for c in courses[:4]:
            print(f"   [{c['code']}] {c['title']} (p{c['index_page']})")
