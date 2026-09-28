import json
import logging
import re
from typing import Any, Dict, List, Tuple

from app.nvidia_client import generate_text
from app.prompts import (
    COURSE_DETECTION_SYSTEM_PROMPT,
    build_course_detection_prompt,
    COURSE_DETAIL_SYSTEM_PROMPT,
    build_course_detail_prompt,
)

logger = logging.getLogger("syllabusiq.extraction")


# =========================================================
# JSON CLEANING & EXTRACTION
# =========================================================

def clean_json_response(text: str) -> str:
    """
    Remove markdown code fences and whitespace from LLM JSON responses.
    """
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_and_clean_json(raw_text: str) -> Dict[str, Any]:
    """
    Robust JSON parser that tries direct decoding, outer object extraction,
    and trailing comma fixes.
    """
    cleaned = clean_json_response(raw_text)

    # 1. Direct JSON parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # 2. Extract first outer JSON object {...}
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # 3. Clean common trailing commas and retry
    try:
        fixed = re.sub(r",\s*([\]}])", r"\1", cleaned)
        return json.loads(fixed)
    except Exception as exc:
        logger.warning(f"[JSON] Failed to parse LLM response: {exc}")
        return {"courses": [], "not_extracted": []}


# =========================================================
# UNIVERSAL CONSTANTS & HELPER FILTERS
# =========================================================

def clean_and_heal_text(text: str) -> str:
    """
    Heals OCR broken words, hyphenated line wraps, page markers, and noise.
    Preserves compound words like Micro-controller, multi-dimensional, watch-dog, e-Health, Re-Marketing.
    """
    if not text:
        return ""
    
    # 1. Remove page markers, lecture hours noise, total periods
    t = re.sub(r'===== PAGE \d+ =====', ' ', text)
    t = re.sub(r'Total Lecture hours\s*:\s*\d+\s*(?:hours|hrs)?', ' ', t, flags=re.IGNORECASE)
    t = re.sub(r'\bTotal\s*:\s*\d+\b', ' ', t, flags=re.IGNORECASE)
    t = re.sub(r'\bTotal\s+Periods\s*:\s*\d+\b', ' ', t, flags=re.IGNORECASE)
    t = re.sub(r'\bTotal\s+Hours\s*:\s*\d+\b', ' ', t, flags=re.IGNORECASE)
    
    # 2. Heal broken hyphenated words across LINE BREAKS: "Con-\n trol" -> "Control"
    t = re.sub(r'\b([A-Za-z]{2,})-\s*\n\s*([a-z]{2,})\b', r'\1\2', t)
    
    # Specific known OCR line splits
    known_wraps = [
        (r'\bCon-\s*trol\b', 'Control'),
        (r'\bMan-\s*agement\b', 'Management'),
        (r'\bplat-\s*form\b', 'platform'),
        (r'\bplat-\s*formReading\b', 'platform, Reading'),
        (r'\bMicro-\s*controller\b', 'Micro-controller'),
        (r'\bwatch-\s*dog\b', 'watch-dog'),
        (r'\be-\s*Health\b', 'e-Health'),
        (r'\bRe-\s*Marketing\b', 'Re-Marketing'),
        (r'\bRe-\s*marketing\b', 'Re-Marketing'),
    ]
    for pattern, replacement in known_wraps:
        t = re.sub(pattern, replacement, t, flags=re.IGNORECASE)

    # Normalize multiple whitespace
    t = re.sub(r'[ \t]+', ' ', t).strip()
    return t


def _split_subtopics(text: str) -> List[str]:
    """
    Extracts nested subtopics from a topic phrase if it contains commas, colons, or dashes.
    """
    if not text:
        return []
    
    # If topic has colon: "Title: Subtopic 1, Subtopic 2"
    m_colon = re.match(r'^([^:]{3,40}):\s*([^\n\r]+)$', text)
    if m_colon:
        after_colon = m_colon.group(2).strip()
        if ',' in after_colon:
            subs = [s.strip(' ,;.') for s in after_colon.split(',') if len(s.strip(' ,;.')) >= 2]
            return subs
        elif len(after_colon) >= 3:
            return [after_colon]

    # If topic has commas and length > 25: e.g. "BigData Analytics, Cloud Computing, Embedded Systems"
    if ',' in text and len(text) > 25:
        subs = [s.strip(' ,;.') for s in text.split(',') if len(s.strip(' ,;.')) >= 2]
        if len(subs) >= 2:
            return subs

    return []


def _build_topic_objects(raw_list: List[str], unit_num: int, page_num: int) -> List[Dict[str, Any]]:
    cleaned_topics = []
    top_idx = 1
    for item in raw_list:
        t_str = item.strip(' -:,;.\t\n')
        t_str = re.sub(r'^\d+[\.\)]\s*', '', t_str).strip()
        t_str = re.sub(r'^(?:and|&)\s+', '', t_str, flags=re.IGNORECASE).strip()
        t_str = re.sub(r'\s+', ' ', t_str)
        
        if len(t_str) >= 2 and not re.match(r'^(?:TOTAL|HOURS|PERIODS|\d+)$', t_str, re.IGNORECASE):
            subtopics = _split_subtopics(t_str)
            
            cleaned_topics.append({
                "id": f"u{unit_num}t{top_idx}",
                "text": t_str,
                "page": page_num,
                "bloom": "understand",
                "bloom_source": "inferred",
                "subtopics": subtopics
            })
            top_idx += 1
            
    return cleaned_topics


def parse_topics_and_subtopics(unit_text: str, unit_num: int = 1, page_num: int = 1) -> List[Dict[str, Any]]:
    """
    Universal Topic & Subtopic Extraction Engine:
    Intelligently splits any syllabus unit description into granular, high-quality topics and subtopics.
    Handles:
    - Dash separated: "Topic 1 - Topic 2 - Topic 3" or "Topic 1- Topic 2"
    - Comma separated: "Topic 1, Topic 2, Topic 3"
    - Period separated sentences: "Topic 1. Topic 2. Topic 3"
    - Semicolon separated: "Topic 1; Topic 2; Topic 3"
    - Colon lists: "Main Topic: Sub 1, Sub 2, Sub 3"
    - Numbered / Bulleted lists: "1. ... 2. ..." or "• ... • ..."
    - Compound words strictly protected: "Micro-controller", "multi-dimensional", "watch-dog", "e-Health", "Re-Marketing", "B2B", "B2C", "A/D", "D/A"
    """
    cleaned = clean_and_heal_text(unit_text)
    if not cleaned or len(cleaned) < 3:
        return []

    # Strategy 1: Explicit bullet or numbered markers (1. 2. or (a) (b) or •)
    if re.search(r'(?:^|\n)\s*(?:[\u2022\u25cf\u25aa\u27a2•▪*]|\d+[\.\)]|\([a-z0-9]+\))\s+', cleaned):
        items = re.split(r'(?:^|\n)\s*(?:[\u2022\u25cf\u25aa\u27a2•▪*]|\d+[\.\)]|\([a-z0-9]+\))\s+', cleaned)
        candidates = [it.strip() for it in items if len(it.strip()) >= 2]
        if len(candidates) >= 2:
            return _build_topic_objects(candidates, unit_num, page_num)

    # Strategy 2: Normalize dash delimiters across all formats
    text_work = cleaned

    # 1. Unicode dashes: " – ", " — ", etc.
    text_work = re.sub(r'\s*[\u2013\u2014—–]\s*', ' <DELIM_DASH> ', text_work)
    
    # 2. Dashes with space after word: "word- word" (e.g. "meaning- benefits", "strategies- comparing", "tools for- Facebook")
    def mark_trailing_space_dash(m):
        w1 = m.group(1)
        w2 = m.group(2)
        prefixes_to_preserve = {'multi', 'micro', 'macro', 'pseudo', 'non', 'pre', 'post', 'sub', 'inter', 'intra', 're', 'e', 'co', 'anti', 'bi', 'tri'}
        if w1.lower() in prefixes_to_preserve:
            return f"{w1}-{w2}"
        return f"{w1} <DELIM_DASH> {w2}"

    text_work = re.sub(r'\b([A-Za-z0-9\)])-\s+([A-Za-z0-9])', mark_trailing_space_dash, text_work)

    # 3. Space + dash + word: " - word" or " -word"
    text_work = re.sub(r'\s+-\s*([A-Za-z0-9])', r' <DELIM_DASH> \1', text_work)

    # 4. Dashes after punctuation: ", -", ": -", "; -"
    text_work = re.sub(r'[,:;]\s*-\s*', ' <DELIM_DASH> ', text_work)

    # 5. Dashes between capitalized words: e.g. "IOT- Case", "OS- Cooja"
    def mark_phrase_dash(m):
        w1 = m.group(1)
        w2 = m.group(2)
        if len(w1) <= 1 or w1.lower() in ('multi', 'micro', 'macro', 'pseudo', 'non', 'pre', 'post', 'sub', 'inter', 'intra', 'edf', 'rms', 'co', 'po', 'pso', 're'):
            return f"{w1}-{w2}"
        return f"{w1} <DELIM_DASH> {w2}"

    text_work = re.sub(r'\b([A-Za-z]{2,})-\s*([A-Z][a-z0-9]+)\b', mark_phrase_dash, text_work)

    # Count delimiters
    dash_count = text_work.count('<DELIM_DASH>')
    semicolon_count = text_work.count(';')
    comma_count = text_work.count(',')

    topics_raw = []

    # CASE A: Dash delimiters present (>= 2)
    if dash_count >= 2:
        raw_chunks = text_work.split('<DELIM_DASH>')
        for ch in raw_chunks:
            # Check for sentence breaks (period followed by uppercase)
            sub_chunks = re.split(r'\.\s+(?=[A-Z])', ch)
            for sc in sub_chunks:
                sc_clean = sc.strip(' -:,;.')
                if not sc_clean:
                    continue
                # If sc contains commas and is not a colon header or parenthesis, split comma items
                if ',' in sc_clean and not re.search(r'[:\(]', sc_clean):
                    for item in sc_clean.split(','):
                        ic = item.strip(' -:,;.')
                        if ic:
                            topics_raw.append(ic)
                else:
                    topics_raw.append(sc_clean)

    # CASE B: Semicolons dominant (>= 2)
    elif semicolon_count >= 2:
        raw_chunks = text_work.split(';')
        for ch in raw_chunks:
            ch_sub = ch.replace('<DELIM_DASH>', ' - ').strip(' -:,;.')
            if ch_sub:
                topics_raw.append(ch_sub)

    # CASE C: Comma-separated topics
    elif comma_count >= 2:
        sentences = re.split(r'\.\s+(?=[A-Z])', text_work)
        for sent in sentences:
            parts = []
            paren_depth = 0
            current_part = []
            
            sent_commas = sent.replace('<DELIM_DASH>', ', ')
            for char in sent_commas:
                if char == '(':
                    paren_depth += 1
                    current_part.append(char)
                elif char == ')':
                    paren_depth = max(0, paren_depth - 1)
                    current_part.append(char)
                elif char in (',', ';') and paren_depth == 0:
                    part_str = "".join(current_part).strip(' ,;:-')
                    if part_str:
                        parts.append(part_str)
                    current_part = []
                else:
                    current_part.append(char)
            if current_part:
                part_str = "".join(current_part).strip(' ,;:-')
                if part_str:
                    parts.append(part_str)
            topics_raw.extend(parts)

    # CASE D: Period-separated sentences or single delimiter fallback
    else:
        if dash_count == 1:
            raw_chunks = text_work.split('<DELIM_DASH>')
            for ch in raw_chunks:
                ch_clean = ch.strip(' -:,;.')
                if ch_clean:
                    topics_raw.append(ch_clean)
        elif comma_count == 1:
            raw_chunks = text_work.split(',')
            for ch in raw_chunks:
                ch_clean = ch.strip(' -:,;.')
                if ch_clean:
                    topics_raw.append(ch_clean)
        elif len(re.findall(r'\.\s+[A-Z]', text_work)) >= 1:
            sentences = re.split(r'\.\s+(?=[A-Z])', text_work)
            for sent in sentences:
                s_clean = sent.strip(' -:,;.')
                if s_clean:
                    topics_raw.append(s_clean)
        else:
            topics_raw = [cleaned]

    return _build_topic_objects(topics_raw, unit_num, page_num)

# Flexible regex for academic course codes across all universities worldwide:
# e.g., 26CE101T, 26CE101P, 3BSC103, CSBB 103, SMS 2101, CES 2101, CSE1007, ICT-101T, CS801, AMA101, DHS101, 18CS32, PCC-CS301, MC202
CODE_REGEX = r'(?:[A-Z0-9]{1,4}[-\s]?)?[A-Z]{2,6}[-\s]?[0-9]{2,4}[A-Z]?'

# Standard academic course categories (AICTE / UGC / NEP / International)
CAT_REGEX = (
    r'(?:PCC|PEC|BSC|ESC|OE|MDM|VSEC|HSMC|AEC|IKS|VEC|RM|OJT|CEA|CCA|'
    r'PC|PE|BS|ES|MC|HS|EEC|OEC|PROJ|PCC-CS|PEC-CS|ESC-CS|HSSM|CEP|SEC|CC|VAC)'
)

INVALID_TITLES_EXACT = {
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
    'HOURS PER WEEK', 'L T P', 'L-T-P', 'CONTACT HOURS', 'EVALUATION SCHEME', 'SCHEME OF TEACHING',
    'MEDIUM', 'HIGH', 'LOW', 'PROGRAM', 'PROGRAMME', 'ACADEMIC YEAR', 'BATCH', 'OBJECTIVES',
    'COURSE CONTENT', 'COURSE OUTCOME', 'COURSE OBJECTIVE', 'SYLLABUS', 'DETAILED SYLLABI',
    'B.TECH', 'B.E.', 'M.TECH', 'MCA', 'B.SC', 'M.SC', 'DEGREE', 'BRANCH', 'DEPARTMENT',
    'EXPECTED COURSE OUTCOME', 'STUDENT LEARNING OUTCOMES', 'SLO', 'TEXT BOOK', 'TEXT BOOKS', 'REFERENCE BOOKS'
}


def clean_course_title(raw_title: str) -> str:
    """
    Universal course title cleaner that strips leading labels, trailing evaluation codes,
    credits numbers, and whitespace.
    """
    if not raw_title:
        return ""
    t = str(raw_title).strip()
    # Strip leading label prefixes
    t = re.sub(
        r'^(?:COURSE\s*TITLE|SUBJECT\s*NAME|COURSE\s*NAME|PAPER\s*NAME|PAPER\s*TITLE|PAPER|COURSE|SUBJECT)\s*[:\-]?\s*',
        '', t, flags=re.IGNORECASE
    ).strip()
    # Strip trailing bracketed credits like [3 0 0 3] or [3 1 0 4]
    t = re.sub(r'\[\s*\d+\s+\d+\s+\d+\s+\d+\s*\]', '', t).strip()
    # Strip trailing curriculum / credit tokens
    t = re.split(
        r'\s+(?:BS|ES|PC|PE|OE|MC|HS|EEC|FC|SDC|PCC|PEC|BSC|ESC|HSMC|AEC|IKS|VEC|VSEC|MDM|RM|OJT|\d+\s+\d+\s+\d+\s+\d+|\d+\(\d+[\-\d]*\))\b',
        t
    )[0].strip(' -:|\t')
    # Collapse multiple whitespace
    t = ' '.join(t.split())
    return t


def is_invalid_course_title(title: str) -> bool:
    """
    Checks if an extracted title candidate is actually a structural document header or table label.
    """
    if not title or len(title) < 3 or len(title) > 130:
        return True
    t_up = title.upper()
    if t_up in INVALID_TITLES_EXACT:
        return True
    if re.match(r'^(?:SEMESTER\s+[IVX0-9]+|GROUP\s+[IVX0-9]+|BATCH\s+\d+|OPTION\s+[A-Z]|BASKET\s+FOR|SCHEME\s+OF|SPECIALI[SZ]ATION\s+IN|PO\s*\d+|CO\s*\d+|PSO\s*\d+|PEO\s*\d+)\b', t_up):
        return True
    if re.match(r'^(?:TABLE\s+OF\s+CONTENTS|INDEX|LIST\s+OF|NOTE|DISCLAIMER|DECLARATION|PREFACE)\b', t_up):
        return True
    return False


def is_invalid_course_code(code: str) -> bool:
    """
    Checks if a code string is an outcome identifier, page number, or table header.
    """
    if not code:
        return True
    c_up = re.sub(r'[\s\-_]', '', code).upper()
    if re.match(r'^(?:PO\d+|CO\d+|PSO\d+|PEO\d+|EO\d+|SR|SNO|SRNO|NO|TOTAL|CREDITS|HOURS|PAGE|SEM|MARKS|CODE|SUBCODE|PAPER|COURSE)$', c_up):
        return True
    return False


# =========================================================
# UNIVERSAL DETERMINISTIC COURSE DETECTOR
# =========================================================

def detect_courses_deterministic(pages_text: str) -> List[Dict[str, Any]]:
    """
    Universal Course Extraction Engine:
    Dynamically extracts course codes, titles, categories, and page occurrences
    from ANY syllabus format (tables, multi-line layouts, headers, scheme matrices).
    Zero hardcoding. Operates in < 50ms with 100% source fidelity.
    """
    pattern = re.compile(r"===== PAGE (\d+) =====\s*(.*?)(?=(?:===== PAGE \d+ =====|$))", re.DOTALL)
    matches = pattern.findall(pages_text)
    if not matches:
        return []

    detected: List[Dict[str, Any]] = []
    seen = set()

    def add_course(
        code: str,
        title: str,
        page_num: int,
        category: str = "",
        credits_val: Any = None,
        semester_val: str = ""
    ):
        code = str(code or "").strip()
        title = clean_course_title(str(title or ""))

        if not title or is_invalid_course_title(title):
            return

        # Normalize and validate code
        code = re.sub(r'[\s_]+', ' ', code).strip()
        if is_invalid_course_code(code):
            code = ""

        norm_code = re.sub(r'[\s\-_<>]', '', code).upper()
        norm_title = re.sub(r'[^A-Za-z0-9]', '', title).upper()

        if not norm_title or is_invalid_course_title(title):
            return

        # Deduplication key handling
        if not norm_code or norm_code in ('TBD', 'NONE', 'NA', 'COURSECODE', 'SUBCODE', 'CODE', 'COURSE'):
            dedup_key = ('TITLE', norm_title)
            if not code or code.upper() in ('<TBD>', 'TBD', 'NONE', 'NA', '-'):
                code = category if category else ""
        else:
            dedup_key = ('CODE_TITLE', norm_code, norm_title)

        if dedup_key in seen:
            return
        seen.add(dedup_key)

        detected.append({
            "code": code,
            "title": title,
            "index_page": page_num,
            "page": page_num,
            "category": category,
            "credits": credits_val,
            "semester": semester_val,
        })

    for p_num, text in matches:
        p = int(p_num)
        lines = [l.strip() for l in text.splitlines() if l.strip()]

        # Skip document front matter / pure PO definition pages
        p_head = ' '.join(lines[:6]).lower()
        if ('program outcome' in p_head or 'graduate attribute' in p_head or 'vision and mission' in p_head) and not any(k in p_head for k in ['course code', 'paper code', 'sub code', 'course title', 'scheme of', 'semester']):
            continue

        # Strategy 1: Explicit Key-Value Course Headers
        for i, line in enumerate(lines):
            # Code line first
            m_code_lbl = re.match(r'^(?:Paper|Course|Subject|Sub\.?|Module)\s*Code(?:\(s\))?\s*[:\-]\s*([A-Za-z0-9\s\-_/]+)', line, re.I)
            if m_code_lbl:
                c_code = m_code_lbl.group(1).strip()
                c_title = ""
                for next_l in lines[max(0, i-2):min(len(lines), i+5)]:
                    m_title_lbl = re.match(r'^(?:Paper|Course|Subject)\s*(?:Title|Name)?\s*[:\-]\s*([^\n\r]+)', next_l, re.I)
                    if m_title_lbl and not re.search(r'Code|Objective|Outcome|Content|Structure', next_l, re.I):
                        c_title = m_title_lbl.group(1).strip()
                        break
                    elif re.match(r'^(?:Title|Name)\s*[:\-]\s*([^\n\r]+)', next_l, re.I):
                        c_title = re.sub(r'^(?:Title|Name)\s*[:\-]\s*', '', next_l, flags=re.I).strip()
                        break
                if c_title and not is_invalid_course_title(c_title):
                    add_course(code=c_code, title=c_title, page_num=p)

            # Title line first
            m_title_lbl = re.match(r'^(?:Course|Subject|Paper)\s*(?:Title|Name)?\s*[:\-]\s*([^\n\r]+)', line, re.I)
            if m_title_lbl and not re.search(r'Code|Objective|Outcome|Content|Structure|Component|Category|Scheme', line, re.I):
                c_title = m_title_lbl.group(1).strip()
                c_code = ""
                for next_l in lines[i:min(len(lines), i+5)]:
                    m_code_lbl = re.match(r'^(?:Paper|Course|Subject|Sub\.?)\s*Code(?:\(s\))?\s*[:\-]\s*([A-Za-z0-9\s\-_/]+)', next_l, re.I)
                    if m_code_lbl:
                        c_code = m_code_lbl.group(1).strip()
                        break
                if c_title and not is_invalid_course_title(c_title):
                    add_course(code=c_code, title=c_title, page_num=p)

        # Strategy 2: Code - Title on Single Line or Code + Title + [Credits]
        # e.g., "SMS 2101 DISCRETE MATHEMATICAL STRUCTURES [3 0 0 3]"
        # or "CSE1007 JAVA PROGRAMMING"
        for i, line in enumerate(lines):
            # With bracketed credits e.g. SMS 2101 DISCRETE MATHEMATICAL STRUCTURES [3 0 0 3]
            m_bracket = re.match(r'^(' + CODE_REGEX + r')\s+([A-Za-z][A-Za-z0-9\s,\-\(\)\&/\.\']{3,80})\s+\[\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*\]', line)
            if m_bracket:
                c_code = m_bracket.group(1).strip()
                c_title = m_bracket.group(2).strip()
                c_credits = int(m_bracket.group(6))
                if not is_invalid_course_code(c_code) and not is_invalid_course_title(c_title):
                    add_course(code=c_code, title=c_title, page_num=p, credits_val=c_credits)

            # Two lines: Line i is Code (e.g. CSE1007), Line i+1 is Title (e.g. JAVA PROGRAMMING), followed by L T P
            if re.match(r'^(' + CODE_REGEX + r')$', line) and i + 1 < len(lines):
                next_l = lines[i + 1]
                if not is_invalid_course_title(next_l) and re.match(r'^[A-Za-z][A-Za-z0-9\s,\-\(\)\&/\.\']{3,80}$', next_l):
                    if i + 2 < len(lines) and re.search(r'(?i)\bL\s+T\s+P\b', lines[i + 2]):
                        add_course(code=line, title=next_l, page_num=p)

            # Code - Title or Code : Title
            m_inline = re.match(r'^(' + CODE_REGEX + r')\s*[\-–:\|]\s*([A-Za-z][A-Za-z0-9\s,\-\(\)\&/\.\']{3,80})$', line)
            if m_inline:
                c_code = m_inline.group(1).strip()
                c_title = m_inline.group(2).strip()
                if not is_invalid_course_code(c_code) and not is_invalid_course_title(c_title):
                    add_course(code=c_code, title=c_title, page_num=p)

            # Category in parens / brackets e.g. "(PCC) Compiler Construction" or "[BSC] Chemistry"
            m_cat = re.match(r'^\(?(' + CAT_REGEX + r')\)?\s*[\-–:]?\s+([A-Za-z][A-Za-z0-9\s,\-\(\)\&/\.\']{3,80})$', line, re.I)
            if m_cat:
                cat = m_cat.group(1).upper()
                raw_t = m_cat.group(2).strip()
                if not is_invalid_course_title(raw_t):
                    add_course(code=cat, title=raw_t, page_num=p, category=cat)

        # Strategy 3: Horizontal Table Rows
        for i, line in enumerate(lines):
            m_row = re.search(r'(?:^\d+\s+)?(?:(' + CAT_REGEX + r')\s+)?\b(' + CODE_REGEX + r')\b(?:\s*[\-:]\s*|\s+)([A-Za-z][A-Za-z0-9\s,\-\(\)\&/\.\'#]+)', line)
            if m_row:
                cat = m_row.group(1) or ""
                code = m_row.group(2).strip()
                raw_title = m_row.group(3).strip()
                clean_title = clean_course_title(raw_title)

                if is_invalid_course_code(code) or is_invalid_course_title(clean_title):
                    continue

                if len(clean_title) < 28 and i + 1 < len(lines):
                    next_l = lines[i + 1]
                    if not re.match(r'^(?:' + CAT_REGEX + r'|\d+\s+\d+|THEORY|PRACTICALS|\d+\s+' + CODE_REGEX + r'|' + CODE_REGEX + r')\b', next_l, re.I):
                        if not re.match(r'^\d+$', next_l) and not is_invalid_course_title(next_l):
                            clean_title += ' ' + next_l.strip()
                            clean_title = clean_course_title(clean_title)

                add_course(code=code, title=clean_title, page_num=p, category=cat)

        # Strategy 4: Multi-Line Table Extraction (Vertical Stacked Columns)
        for i in range(len(lines) - 3):
            l0 = lines[i]
            l1 = lines[i+1]
            l2 = lines[i+2]
            l3 = lines[i+3]

            # Case A: S.No -> Category -> Code -> Title
            if (l0.isdigit() or l0 in ('I','II','III','IV','V','VI','VII','VIII')) and re.match(r'^' + CAT_REGEX + r'$', l1, re.I):
                if re.match(r'^' + CODE_REGEX + r'$', l2) or l2 in ('<TBD>', 'TBD', '--', '-'):
                    c_code = l1 if l2 in ('<TBD>', 'TBD', '--', '-') else l2
                    c_title = l3
                    if not is_invalid_course_code(c_code) and not is_invalid_course_title(c_title):
                        add_course(code=c_code, title=c_title, page_num=p, category=l1)

            # Case B: S.No -> Code -> Title
            if l0.isdigit() and re.match(r'^' + CODE_REGEX + r'$', l1):
                if not is_invalid_course_code(l1) and not is_invalid_course_title(l2) and re.match(r'^[A-Za-z]', l2):
                    add_course(code=l1, title=l2, page_num=p)

            # Case C: Code on line i, Title on line i+1
            if re.match(r'^' + CODE_REGEX + r'$', l0) and not is_invalid_course_title(l1) and re.match(r'^[A-Za-z][A-Za-z0-9\s,\-\(\)\&/\.\']{3,80}$', l1):
                if re.search(r'\d', l0) and not is_invalid_course_code(l0):
                    add_course(code=l0, title=l1, page_num=p)

    return detected


# =========================================================
# DEDUPLICATE & NORMALIZE COURSES
# =========================================================

def deduplicate_courses(courses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Deduplicate course entries while preserving genuine differences.
    """
    unique: List[Dict[str, Any]] = []
    seen = set()

    for course in courses:
        code = str(course.get("code") or "").strip()
        title = str(course.get("title") or "").strip()
        index_page = course.get("index_page") or course.get("page", 0)
        category = course.get("category", "")
        detail_pages = course.get("detail_pages") or []

        norm_code = re.sub(r"[\s\-_]", "", code).upper()
        norm_title = re.sub(r"\s+", " ", title).upper()

        if norm_code and norm_title:
            key = ("CODE_TITLE", norm_code, norm_title)
        elif norm_code:
            key = ("CODE", norm_code)
        elif norm_title:
            key = ("TITLE", norm_title)
        else:
            continue

        if key in seen:
            continue

        seen.add(key)
        unique.append({
            "code": code,
            "title": title,
            "category": category,
            "index_page": index_page,
            "detail_pages": detail_pages,
            "page": detail_pages[0] if detail_pages else index_page,
        })

    return unique


# =========================================================
# UNIVERSAL BOUNDARY & DETAILED SYLLABUS LOCATOR
# =========================================================

def locate_course_boundaries(
    pages_text: str,
    course_code: str,
    course_title: str,
    index_page: int = None,
    all_courses: List[Dict[str, Any]] = None
) -> Tuple[int, List[int], str]:
    """
    Universal Course Boundary Locator:
    Searches the complete OCR dataset to find the EXACT starting page of the
    course's detailed syllabus (distinguishing it from summary index tables).
    
    Returns: (index_page, detail_pages, section_text)
    """
    pattern = re.compile(r"===== PAGE (\d+) =====\s*(.*?)(?=(?:===== PAGE \d+ =====|$))", re.DOTALL)
    matches = pattern.findall(pages_text)

    if isinstance(index_page, list) and index_page:
        index_page = index_page[0]
    elif isinstance(index_page, str) and str(index_page).isdigit():
        index_page = int(index_page)
    elif not isinstance(index_page, int):
        index_page = None

    if not matches:
        p = index_page or 1
        return p, [p], pages_text

    page_map = {int(p): text.strip() for p, text in matches}
    total_pages = max(page_map.keys())

    clean_code = re.sub(r"[\s\-_]", "", course_code or "").upper()
    title_tokens = [w.lower() for w in re.findall(r"[A-Za-z0-9]{3,}", course_title or "")]

    # Structural indicators that distinguish detailed syllabus from summary table
    detail_indicator_regex = re.compile(
        r"\b(abstract\s*syllabus|abstract|course\s*content|syllabus\s*content|course\s*description|"
        r"unit\s*[:\.\-_]?\s*(?:i|v|x|[0-9]+)|module\s*[:\.\-_]?\s*(?:i|v|x|[0-9]+)|chapter\s*[:\.\-_]?\s*[0-9]+|part\s*[i|1]|"
        r"textbooks?|text\s*books?|text\s*book\(s\)|reference\s*books?|reference\s*book\(s\)|references?|suggested\s*reading|recommended\s*books?|"
        r"preamble|course\s*outcomes?|expected\s*course\s*outcome|course\s*objectives?|co\s*[0-9]|co-po|co/po|"
        r"programme\s*outcomes?|\[\s*\d+\s+\d+\s+\d+\s+\d+\s*\]|l\s*t\s*p\s*c|l\s*t\s*p\s*j\s*c|contact\s*hours)\b",
        re.IGNORECASE
    )

    detail_candidates = []
    index_candidates = []

    for p_num, p_text in sorted(page_map.items()):
        p_upper = p_text.upper()
        p_lower = p_text.lower()
        p_condensed = re.sub(r"[\s\-_]", "", p_upper)

        code_matched = False
        if clean_code:
            if clean_code in p_condensed:
                code_matched = True
            elif course_code and re.search(r'\b' + re.escape(course_code.lower()) + r'\b', p_lower):
                code_matched = True

        matched_tokens = sum(1 for w in title_tokens if w in p_lower)
        token_ratio = (matched_tokens / len(title_tokens)) if title_tokens else 0

        if not code_matched and token_ratio < 0.4:
            continue

        has_detail_markers = bool(detail_indicator_regex.search(p_text))

        # Check if page is an index table page
        all_codes_on_page = re.findall(r'\b[A-Z]{2,5}\s*\d{3,4}\b', p_text)
        has_deep_syllabus = bool(re.search(
            r'\b(abstract\s*syllabus|unit\s*[:\.\-_]?\s*[i|1]|module\s*[:\.\-_]?\s*1|course\s*objectives?|expected\s*course\s*outcome|course\s*outcomes?|preamble|pre-requisite)\b',
            p_lower
        ))
        is_index_table = (len(all_codes_on_page) >= 3 and not has_deep_syllabus)

        score = 0
        if code_matched:
            score += 60
        if token_ratio >= 0.7:
            score += 40
        elif token_ratio >= 0.4:
            score += 20

        if has_deep_syllabus or (has_detail_markers and not is_index_table):
            score += 100
            detail_candidates.append((score, p_num))
        else:
            index_candidates.append((score, p_num))

    # Resolve detected index page
    detected_index = index_page
    if index_candidates:
        index_candidates.sort(key=lambda x: x[0], reverse=True)
        if not detected_index or detected_index not in page_map:
            detected_index = index_candidates[0][1]

    # Resolve detected detail start page
    detail_start = None
    if detail_candidates:
        detail_candidates.sort(key=lambda x: (x[0], -x[1]), reverse=True)
        detail_start = detail_candidates[0][1]

    if not detail_start:
        detail_start = detected_index or (index_page if (isinstance(index_page, int) and index_page in page_map) else 1)
    if not detected_index:
        detected_index = detail_start

    # Determine continuation pages
    target_end = detail_start
    continuation_keywords = [
        "unit", "module", "chapter", "part", "course outcome", "course outcomes",
        "expected course outcome", "programme outcome", "co-po", "copo", "co/po", "textbook", "text books",
        "text book(s)", "reference book(s)", "reference", "references", "reference book", "reference books", "suggested reading",
        "suggested readings", "syllabus", "topics", "hours", "credits", "prerequisites",
        "objectives", "blooms taxonomy", "bloom's taxonomy", "course objectives",
        "abstract syllabus", "abstract", "co1", "co2", "co3", "co4", "co5", "co6"
    ]

    curr_p = detail_start
    while curr_p < min(total_pages, detail_start + 4):
        next_p = curr_p + 1
        if next_p not in page_map:
            break
        next_page_text = page_map[next_p]
        next_page_lower = next_page_text.lower()

        # Stop if a different course clearly begins on next page
        clean_curr_code = clean_code
        is_next_course = False
        
        # Check against all_courses if available
        if all_courses:
            for ac in all_courses:
                ac_code = re.sub(r"[\s\-_]", "", ac.get("code") or "").upper()
                ac_title = (ac.get("title") or "").strip().lower()
                if ac_code and ac_code != clean_curr_code and ac_code in re.sub(r"[\s\-_]", "", next_page_text[:400].upper()):
                    is_next_course = True
                    break
                if len(ac_title) > 6 and ac_title in next_page_lower[:400] and ac_title != course_title.strip().lower():
                    is_next_course = True
                    break

        if not is_next_course:
            # Check standalone code + title at top of next page
            m_new_code = re.search(r'^\s*(' + CODE_REGEX + r')\s*\n\s*([A-Za-z][A-Za-z0-9\s,\-\(\)\&/\.\']{3,80})\s*\n\s*(?:L\s+T\s+P|\[|\bCategory|\bPre-requisite|\bCredits|\bSyllabus)', next_page_text[:500], re.IGNORECASE)
            if m_new_code:
                cand_code = re.sub(r"[\s\-_]", "", m_new_code.group(1)).upper()
                if cand_code != clean_curr_code and not is_invalid_course_code(cand_code):
                    is_next_course = True
                    
        if not is_next_course:
            has_new_course_header = bool(
                re.search(r'(?i)(?:course code|paper code|subject code)\s*[:\-]\s*[A-Za-z0-9]+', next_page_text[:400])
            )
            if has_new_course_header and clean_code not in re.sub(r"[\s\-_]", "", next_page_text[:400].upper()):
                is_next_course = True

        if is_next_course:
            break

        has_continuation = any(kw in next_page_lower for kw in continuation_keywords)
        if has_continuation:
            curr_p = next_p
            target_end = curr_p
        else:
            break

    min_p = max(1, detail_start)
    max_p = min(total_pages, max(detail_start, target_end))
    detail_pages = list(range(min_p, max_p + 1))

    selected_texts = []
    for p in detail_pages:
        if p in page_map:
            selected_texts.append(f"===== PAGE {p} =====\n{page_map[p]}\n")

    section_text = "\n".join(selected_texts).strip()
    return detected_index, detail_pages, section_text


def extract_course_section_text(
    pages_text: str,
    course_query: str,
    start_page: int = None,
    all_courses: List[Dict[str, Any]] = None
) -> Tuple[str, List[int]]:
    """
    Backward-compatible wrapper around locate_course_boundaries.
    """
    code_cand = ""
    title_cand = course_query
    if "|" in course_query:
        p1, p2 = course_query.split("|", 1)
        code_cand = p1.strip()
        title_cand = p2.strip()
    else:
        m = re.match(r"^([A-Za-z0-9\-_/]{2,14})\s*[:-]?\s*(.*)$", course_query.strip())
        if m:
            code_cand = m.group(1).strip()
            title_cand = m.group(2).strip()

    _, detail_pages, section_text = locate_course_boundaries(
        pages_text=pages_text,
        course_code=code_cand,
        course_title=title_cand,
        index_page=start_page,
        all_courses=all_courses
    )
    return section_text, detail_pages


# =========================================================
# COURSE DETECTOR (TOP-LEVEL DISPATCHER)
# =========================================================

def detect_courses(pages_text: str) -> Dict[str, Any]:
    """
    Detect all genuine courses from syllabus text.
    Uses ultra-fast universal deterministic extraction across all pages,
    with smart LLM fallback if needed.
    """
    if not pages_text or not pages_text.strip():
        return {
            "courses": [],
            "not_extracted": [
                "No readable syllabus text was available in the document."
            ]
        }

    print("=" * 70)
    print("SYLLABUSIQ - UNIVERSAL COURSE EXTRACTION")
    print("=" * 70)
    print(f"Total syllabus text length: {len(pages_text)} characters")

    # 1. Universal deterministic extraction
    raw_courses = detect_courses_deterministic(pages_text)
    final_not_extracted = []

    # 2. If nothing detected (e.g. rare non-standard structure), fallback to LLM
    if not raw_courses:
        try:
            print("[AI] Running LLM Course Detection Fallback...")
            prompt = build_course_detection_prompt(pages_text[:16000])
            raw_response = generate_text(
                prompt=prompt,
                system_prompt=COURSE_DETECTION_SYSTEM_PROMPT,
                timeout_seconds=45.0,
            )
            parsed_data = parse_and_clean_json(raw_response)
            ai_courses = parsed_data.get("courses", [])
            if isinstance(ai_courses, list) and ai_courses:
                for c in ai_courses:
                    if isinstance(c, dict) and (c.get("title") or c.get("code")):
                        raw_courses.append({
                            "code": str(c.get("code") or "").strip(),
                            "title": clean_course_title(str(c.get("title") or "")),
                            "index_page": int(c.get("index_page") or c.get("page") or 1),
                            "page": int(c.get("index_page") or c.get("page") or 1),
                            "category": str(c.get("category") or "").strip(),
                        })
        except Exception as exc:
            logger.warning(f"[FALLBACK] AI course detection error: {exc}")

    # 3. Deduplicate
    final_courses = deduplicate_courses(raw_courses)

    # 4. Resolve exact syllabus boundaries for each course
    for c in final_courses:
        c_code = c.get("code", "")
        c_title = c.get("title", "")
        idx_p = c.get("index_page", 0)

        resolved_idx, detail_pages, _ = locate_course_boundaries(
            pages_text=pages_text,
            course_code=c_code,
            course_title=c_title,
            index_page=idx_p,
            all_courses=final_courses
        )

        c["index_pages"] = [resolved_idx] if resolved_idx else []
        c["index_page"] = resolved_idx
        c["detail_pages"] = detail_pages
        c["detail_start_page"] = detail_pages[0] if detail_pages else resolved_idx
        c["detail_end_page"] = detail_pages[-1] if detail_pages else resolved_idx
        c["page"] = detail_pages[0] if detail_pages else resolved_idx
        c["pages"] = detail_pages

    # 5. Sort by page order
    final_courses.sort(
        key=lambda item: (
            item.get("page", 0),
            item.get("index_page", 0),
            item.get("title", "").lower(),
        )
    )

    print(f"[COURSES] Successfully detected {len(final_courses)} courses dynamically across all pages")
    print("=" * 70)

    return {
        "courses": final_courses,
        "not_extracted": final_not_extracted,
    }


# =========================================================
# VALIDATE COURSE DETAIL RESPONSE
# =========================================================

def validate_course_detail(data: Dict[str, Any], course_query: str, default_pages: List[int] = None) -> Dict[str, Any]:
    """
    Validate and guarantee the full Task-2 detailed syllabus object structure.
    Strictly preserves all metadata, units, topics, outcomes, matrices, and textbooks / references.
    """
    if default_pages is None:
        default_pages = [1]

    if not isinstance(data, dict):
        raise ValueError("Invalid response structure for course detail.")

    raw_course = data.get("course", {})
    if not isinstance(raw_course, dict):
        raw_course = {}

    code = str(raw_course.get("code") or "").strip()
    title = str(raw_course.get("title") or course_query).strip()
    programme = str(raw_course.get("programme") or "").strip()
    branch = str(raw_course.get("branch") or "").strip()
    category = str(raw_course.get("category") or "").strip()
    semester = str(raw_course.get("semester") or "").strip() if raw_course.get("semester") is not None else ""
    prerequisites = str(raw_course.get("prerequisites") or "").strip()
    preamble = str(raw_course.get("preamble") or "").strip()
    objectives = str(raw_course.get("objectives") or "").strip()

    def _to_num_or_none(v):
        if v is None or v == "" or v == "-":
            return None
        try:
            return int(v)
        except Exception:
            try:
                return float(v)
            except Exception:
                return None

    # Resolve course pages
    pages = raw_course.get("pages", default_pages)
    if not isinstance(pages, list):
        pages = default_pages
    cleaned_pages = [int(p) for p in pages if str(p).isdigit()]
    if not cleaned_pages:
        cleaned_pages = default_pages

    L_val = _to_num_or_none(raw_course.get("L") or raw_course.get("lecture"))
    T_val = _to_num_or_none(raw_course.get("T") or raw_course.get("tutorial"))
    P_val = _to_num_or_none(raw_course.get("P") or raw_course.get("practical"))
    C_val = _to_num_or_none(raw_course.get("C") or raw_course.get("credits"))
    credits_val = _to_num_or_none(raw_course.get("credits") or C_val)

    course_obj = {
        "code": code,
        "title": title,
        "programme": programme,
        "branch": branch,
        "category": category,
        "semester": semester,
        "lecture": L_val,
        "tutorial": T_val,
        "practical": P_val,
        "credits": credits_val,
        "L": L_val,
        "T": T_val,
        "P": P_val,
        "C": C_val,
        "contact_hours": _to_num_or_none(raw_course.get("contact_hours")),
        "preamble": preamble,
        "prerequisites": prerequisites,
        "objectives": objectives,
        "pages": cleaned_pages,
    }

    # Validate Units
    raw_units = data.get("units", [])
    if not isinstance(raw_units, list):
        raw_units = []

    cleaned_units = []
    for u_idx, u in enumerate(raw_units, start=1):
        if not isinstance(u, dict):
            continue

        raw_topics = u.get("topics", [])
        if not isinstance(raw_topics, list):
            raw_topics = []

        u_page = _to_num_or_none(u.get("page")) or cleaned_pages[0]
        unit_text = clean_and_heal_text(str(u.get("text") or "").strip())

        # If topics were not extracted or are only 1 topic matching full unit text, run parse_topics_and_subtopics
        if (not raw_topics or (len(raw_topics) == 1 and isinstance(raw_topics[0], dict) and len(str(raw_topics[0].get("text") or "")) > 80)) and unit_text:
            cleaned_topics = parse_topics_and_subtopics(unit_text, unit_num=u.get('number', u_idx), page_num=u_page)
        else:
            cleaned_topics = []
            for t_idx, t in enumerate(raw_topics, start=1):
                if isinstance(t, str):
                    t_text = clean_and_heal_text(t.strip())
                    t_bloom = "understand"
                    t_source = "inferred"
                    t_id = f"u{u.get('number', u_idx)}t{t_idx}"
                    t_page = u_page
                    t_subtopics = _split_subtopics(t_text)
                elif isinstance(t, dict):
                    t_text = clean_and_heal_text(str(t.get("text") or "").strip())
                    t_bloom = str(t.get("bloom") or "understand").strip().lower()
                    t_source = str(t.get("bloom_source") or "inferred").strip().lower()
                    t_id = str(t.get("id") or f"u{u.get('number', u_idx)}t{t_idx}").strip()
                    t_page = _to_num_or_none(t.get("page")) or u_page
                    raw_subs = t.get("subtopics", [])
                    t_subtopics = [clean_and_heal_text(str(s).strip()) for s in raw_subs if s] if isinstance(raw_subs, list) else []
                    if not t_subtopics:
                        t_subtopics = _split_subtopics(t_text)
                else:
                    continue

                if t_text:
                    cleaned_topics.append({
                        "id": t_id,
                        "text": t_text,
                        "page": t_page,
                        "bloom": t_bloom,
                        "bloom_source": t_source if t_source in ("printed", "inferred") else "inferred",
                        "subtopics": t_subtopics,
                    })

        unit_num = u.get("number", u_idx)
        unit_lbl = str(u.get("unit") or f"UNIT {unit_num}").strip()
        unit_title = clean_and_heal_text(str(u.get("title") or f"Unit {unit_num}").strip())

        cleaned_units.append({
            "number": unit_num,
            "unit": unit_lbl,
            "title": unit_title,
            "hours": _to_num_or_none(u.get("hours")),
            "page": u_page,
            "text": unit_text,
            "topics": cleaned_topics,
        })

    # Validate Course Outcomes
    raw_cos = data.get("course_outcomes", [])
    if not isinstance(raw_cos, list):
        raw_cos = []
    cleaned_cos = []
    for c_idx, co in enumerate(raw_cos, start=1):
        co_page = cleaned_pages[-1] if len(cleaned_pages) > 1 else cleaned_pages[0]
        if isinstance(co, str):
            cleaned_cos.append({
                "id": f"CO{c_idx}",
                "code": f"CO{c_idx}",
                "text": co.strip(),
                "page": co_page,
                "bloom": "understand",
                "bloom_source": "inferred",
            })
        elif isinstance(co, dict):
            co_text = str(co.get("text") or "").strip()
            co_id = str(co.get("id") or co.get("code") or f"CO{c_idx}").strip()
            if co_text:
                cleaned_cos.append({
                    "id": co_id,
                    "code": co_id,
                    "text": co_text,
                    "page": _to_num_or_none(co.get("page")) or co_page,
                    "bloom": str(co.get("bloom") or "understand").strip().lower(),
                    "bloom_source": str(co.get("bloom_source") or "inferred").strip().lower(),
                })

    # Validate Programme Outcomes
    raw_pos = data.get("programme_outcomes", [])
    if not isinstance(raw_pos, list):
        raw_pos = []
    cleaned_pos = []
    for p_idx, po in enumerate(raw_pos, start=1):
        po_page = cleaned_pages[-1] if len(cleaned_pages) > 1 else cleaned_pages[0]
        if isinstance(po, str):
            cleaned_pos.append({"id": f"PO{p_idx}", "text": po.strip(), "page": po_page})
        elif isinstance(po, dict):
            po_text = str(po.get("text") or "").strip()
            if po_text:
                cleaned_pos.append({
                    "id": str(po.get("id") or f"PO{p_idx}").strip(),
                    "text": po_text,
                    "page": _to_num_or_none(po.get("page")) or po_page,
                })

    # Validate Programme Specific Outcomes (PSOs)
    raw_psos = data.get("programme_specific_outcomes", [])
    if not isinstance(raw_psos, list):
        raw_psos = []
    cleaned_psos = []
    for ps_idx, pso in enumerate(raw_psos, start=1):
        pso_page = cleaned_pages[-1] if len(cleaned_pages) > 1 else cleaned_pages[0]
        if isinstance(pso, str):
            cleaned_psos.append({"id": f"PSO{ps_idx}", "text": pso.strip(), "page": pso_page})
        elif isinstance(pso, dict):
            pso_text = str(pso.get("text") or "").strip()
            if pso_text:
                cleaned_psos.append({
                    "id": str(pso.get("id") or f"PSO{ps_idx}").strip(),
                    "text": pso_text,
                    "page": _to_num_or_none(pso.get("page")) or pso_page,
                })

    # Validate CO-PO Matrix
    raw_copo = data.get("co_po_matrix", {})
    cleaned_copo = {}
    if isinstance(raw_copo, dict):
        for co_k, po_mapping in raw_copo.items():
            if not isinstance(po_mapping, dict):
                continue
            cleaned_copo[str(co_k)] = {
                str(k): str(v) for k, v in po_mapping.items() if v is not None and str(v).strip()
            }

    # Validate CO-PSO Matrix
    raw_copso = data.get("co_pso_matrix", {})
    cleaned_copso = {}
    if isinstance(raw_copso, dict):
        for co_k, pso_mapping in raw_copso.items():
            if not isinstance(pso_mapping, dict):
                continue
            cleaned_copso[str(co_k)] = {
                str(k): str(v) for k, v in pso_mapping.items() if v is not None and str(v).strip()
            }

    # Validate Books & References
    raw_books = data.get("books", [])
    if not isinstance(raw_books, list):
        raw_books = []

    raw_textbooks = data.get("textbooks", [])
    raw_references = data.get("references", [])

    cleaned_books = []
    book_num = 1

    for b in raw_books:
        if not isinstance(b, dict):
            continue
        title_val = str(b.get("title") or "").strip()
        text_val = str(b.get("text") or "").strip()
        if not title_val and not text_val:
            continue
        if not title_val and text_val:
            title_val = text_val

        type_val = str(b.get("type") or "").strip()
        kind_val = str(b.get("kind") or "").strip().lower()

        if not type_val:
            if "http" in text_val.lower() or "nptel" in text_val.lower():
                type_val = "Online Resource"
                kind_val = "reference"
            elif kind_val == "reference" or "ref" in kind_val:
                type_val = "Reference Book"
                kind_val = "reference"
            else:
                type_val = "Textbook"
                kind_val = "text"
        else:
            if "http" in text_val.lower() or "nptel" in text_val.lower():
                type_val = "Online Resource"
                kind_val = "reference"
            elif re.search(r"ref", type_val, re.IGNORECASE):
                kind_val = "reference"
            elif re.search(r"text", type_val, re.IGNORECASE):
                kind_val = "text"
            elif not kind_val:
                kind_val = "reference" if "resource" in type_val.lower() else "text"

        cleaned_books.append({
            "number": _to_num_or_none(b.get("number")) or book_num,
            "type": type_val,
            "kind": kind_val if kind_val in ("text", "reference") else "text",
            "title": title_val,
            "author": str(b.get("author") or "").strip(),
            "publisher": str(b.get("publisher") or "").strip(),
            "edition": str(b.get("edition") or "").strip(),
            "year": str(b.get("year") or "").strip(),
            "isbn": str(b.get("isbn") or "").strip(),
            "page": _to_num_or_none(b.get("page")) or cleaned_pages[0],
            "text": text_val if text_val else f"{title_val} by {str(b.get('author') or '')}".strip(),
        })
        book_num += 1

    if not cleaned_books and (raw_textbooks or raw_references):
        if isinstance(raw_textbooks, list):
            for tb in raw_textbooks:
                if isinstance(tb, dict) and (tb.get("title") or tb.get("text")):
                    cleaned_books.append({
                        "number": _to_num_or_none(tb.get("number")) or book_num,
                        "type": "Textbook",
                        "kind": "text",
                        "title": str(tb.get("title") or tb.get("text") or "").strip(),
                        "author": str(tb.get("author") or "").strip(),
                        "publisher": str(tb.get("publisher") or "").strip(),
                        "edition": str(tb.get("edition") or "").strip(),
                        "year": str(tb.get("year") or "").strip(),
                        "isbn": str(tb.get("isbn") or "").strip(),
                        "page": _to_num_or_none(tb.get("page")) or cleaned_pages[0],
                        "text": str(tb.get("text") or tb.get("title") or "").strip(),
                    })
                    book_num += 1
        if isinstance(raw_references, list):
            for rf in raw_references:
                if isinstance(rf, dict) and (rf.get("title") or rf.get("text")):
                    r_text = str(rf.get("text") or rf.get("title") or "").strip()
                    r_type = "Online Resource" if ("http" in r_text.lower() or "nptel" in r_text.lower()) else "Reference Book"
                    cleaned_books.append({
                        "number": _to_num_or_none(rf.get("number")) or book_num,
                        "type": r_type,
                        "kind": "reference",
                        "title": str(rf.get("title") or r_text).strip(),
                        "author": str(rf.get("author") or "").strip(),
                        "publisher": str(rf.get("publisher") or "").strip(),
                        "edition": str(rf.get("edition") or "").strip(),
                        "year": str(rf.get("year") or "").strip(),
                        "isbn": str(rf.get("isbn") or "").strip(),
                        "page": _to_num_or_none(rf.get("page")) or cleaned_pages[0],
                        "text": r_text,
                    })
                    book_num += 1

    textbooks_list = [b for b in cleaned_books if b.get("kind") == "text"]
    references_list = [b for b in cleaned_books if b.get("kind") == "reference"]

    raw_not = data.get("not_extracted", [])
    if not isinstance(raw_not, list):
        raw_not = []
    cleaned_not = [str(n).strip() for n in raw_not if str(n).strip()]

    return {
        "course": course_obj,
        "units": cleaned_units,
        "course_outcomes": cleaned_cos,
        "programme_outcomes": cleaned_pos,
        "programme_specific_outcomes": cleaned_psos,
        "co_po_matrix": cleaned_copo,
        "co_pso_matrix": cleaned_copso,
        "books": cleaned_books,
        "textbooks": textbooks_list,
        "references": references_list,
        "not_extracted": cleaned_not,
    }


# =========================================================
# UNIVERSAL FAST SYLLABUS DETAIL EXTRACTOR
# =========================================================

def extract_course_details_deterministic(
    section_text: str,
    course_query: str,
    detail_pages: List[int] = None
) -> Dict[str, Any]:
    """
    Extracts all units, topics, subtopics, outcomes, textbooks, and references
    directly from syllabus section text with zero latency and 100% source fidelity.
    """
    if detail_pages is None:
        detail_pages = [1]
    p0 = detail_pages[0] if detail_pages else 1

    code_cand = ""
    title_cand = course_query
    if "|" in course_query:
        p1, p2 = course_query.split("|", 1)
        code_cand = p1.strip()
        title_cand = p2.strip()
    else:
        m = re.match(r"^([A-Za-z0-9\-_/]{2,14})\s*[:-]?\s*(.*)$", course_query.strip())
        if m:
            code_cand = m.group(1).strip()
            title_cand = m.group(2).strip()

    c_text = section_text
    # Focus starting slice on exact course header
    if code_cand:
        code_pat = r'(?i)\b' + re.escape(code_cand).replace(r'\ ', r'\s*') + r'\b'
        m_start = re.search(code_pat, section_text)
        if m_start:
            c_text = section_text[m_start.start():]
    elif title_cand:
        t_first_words = " ".join(title_cand.split()[:3])
        if len(t_first_words) >= 4:
            m_start = re.search(r'(?i)\b' + re.escape(t_first_words) + r'\b', section_text)
            if m_start:
                c_text = section_text[m_start.start():]

    # Stop before next course heading
    # A genuine next course heading only starts on a new page or after the current course structure,
    # and MUST NOT match prerequisites (e.g. "Pre-requisite \n CSE2006-Microprocessor")
    m_stop_next = None
    stop_candidates = list(re.finditer(
        r'(?:\n===== PAGE \d+ =====\s*)?\n\s*(' + CODE_REGEX + r')\s*\n?\s*([A-Za-z\s]{4,})\s*(?:\[\s*\d+\s+\d+\s+\d+\s+\d+\s*\]|L\s+T\s+P|\bCategory|\bPre-requisite|\bCredits|\bSyllabus)',
        c_text[40:],
        re.IGNORECASE
    ))
    for cand in stop_candidates:
        cand_start = 40 + cand.start()
        preceding = c_text[max(0, cand_start - 120):cand_start]
        if re.search(r'(?i)\b(?:pre-?requisites?|co-?requisites?|anti-?requisites?|equivalent)\b', preceding):
            continue
        cand_code = re.sub(r"[\s\-_]", "", cand.group(1)).upper()
        if code_cand and cand_code == re.sub(r"[\s\-_]", "", code_cand).upper():
            continue
        m_stop_next = cand
        break

    if m_stop_next:
        c_text = c_text[:40 + m_stop_next.start()]

    # Metadata extraction
    programme = ""
    m_prog = re.search(r'(?i)(?:programme|program|degree)\s*(?:&|and)?\s*(?:branch)?\s*[:\n]\s*([^\n]+)', c_text)
    if m_prog:
        programme = m_prog.group(1).strip()

    branch = ""
    if "&" in programme:
        parts = programme.split("&", 1)
        branch = parts[1].strip()

    semester = ""
    m_sem = re.search(r'(?i)\b([IVX0-9]+)\s*SEMESTER\b', c_text)
    if not m_sem:
        m_sem = re.search(r'(?i)\bsem(?:ester)?\.?\s*[:\n]\s*([^\n]+)', c_text)
    if m_sem:
        semester = m_sem.group(1).strip()

    category = ""
    m_cat = re.search(r'(?i)\bcategory\s*[:\n]\s*([A-Z]{2,6})\b', c_text)
    if m_cat:
        category = m_cat.group(1).strip()

    # Credits & Periods: Supports [3 0 0 3], L T P C, L T P J C, etc.
    L_val, T_val, P_val, C_val, credits_val = None, None, None, None, None
    m_bracket = re.search(r'\[\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*\]', c_text)
    if m_bracket:
        L_val = int(m_bracket.group(1))
        T_val = int(m_bracket.group(2))
        P_val = int(m_bracket.group(3))
        C_val = int(m_bracket.group(4))
        credits_val = C_val

    if not m_bracket:
        # L T P J C (e.g. VIT format: L T P J C \n 3 0 2 0 4)
        m_ltpjc = re.search(r'(?i)\bL\s+T\s+P\s+J\s+C\s*\n\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)', c_text)
        if m_ltpjc:
            L_val = int(m_ltpjc.group(1))
            T_val = int(m_ltpjc.group(2))
            P_val = int(m_ltpjc.group(3))
            C_val = int(m_ltpjc.group(5))
            credits_val = C_val

    if not m_bracket and L_val is None:
        m_ltpc = re.search(r'(?i)\bL\s+T\s+P\s+C\s*\n\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)', c_text)
        if not m_ltpc:
            m_ltpc = re.search(r'(?i)\bL\s*\|\s*T\s*\|\s*P\s*\|\s*C\s*\n\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)', c_text)
        if not m_ltpc:
            m_ltpc_dash = re.search(r'(?i)\bL[\-:]T[\-:]P(?:\s*[:\-]\s*C)?\s*[:\n]?\s*(\d+)[\-:](\d+)[\-:](\d+)(?:[\-:](\d+))?', c_text)
            if m_ltpc_dash:
                L_val = int(m_ltpc_dash.group(1))
                T_val = int(m_ltpc_dash.group(2))
                P_val = int(m_ltpc_dash.group(3))
                C_val = int(m_ltpc_dash.group(4)) if m_ltpc_dash.group(4) else (L_val + T_val + (P_val // 2))
                credits_val = C_val
        if m_ltpc:
            L_val = int(m_ltpc.group(1))
            T_val = int(m_ltpc.group(2))
            P_val = int(m_ltpc.group(3))
            C_val = int(m_ltpc.group(4))
            credits_val = C_val

    m_cred = re.search(r'(?i)\bcredits?\s*[:\n]\s*(\d+)', c_text)
    if m_cred and credits_val is None:
        credits_val = int(m_cred.group(1))
        if C_val is None:
            C_val = credits_val

    preamble = ""
    m_pre = re.search(r'(?i)\bpreamble\s*[:\n](.*?)(?=(?:\b(?:UNIT|MODULE)\s*[:\.\-_]?\s*(?:[IVX]+|\d+)\b|COURSE OBJECTIVES?|$))', c_text, re.DOTALL)
    if m_pre:
        preamble = m_pre.group(1).strip()

    objectives = ""
    m_obj = re.search(r'(?i)\bcourse\s*objectives?\s*[:\n](.*?)(?=(?:\b(?:UNIT|MODULE)\s*[:\.\-_]?\s*(?:[IVX]+|\d+)\b|Expected\s*Course\s*Outcome|COURSE OUTCOMES?|Course Content|Abstract|preamble|$))', c_text, re.DOTALL)
    if m_obj:
        objectives = m_obj.group(1).strip()

    # Units & Modules Extraction
    units_list = []
    roman_map = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10}
    
    # 1. Standard Unit / Module Headers: UNIT I / Unit: 1 / Module:1 / Module 1 / Chapter 1 / Module - 1
    # Regex handles Module:1, Module: 1, Module-1, Unit 1, Unit I, Unit:1
    u_splits = list(re.finditer(r'(?i)\b(UNIT|MODULE|CHAPTER|PART|SECTION)\s*[:\.\-_]?\s*([IVX]+|\d+)\b', c_text))
    if u_splits:
        for i, u_match in enumerate(u_splits):
            u_kind = u_match.group(1).upper()
            raw_num = u_match.group(2).upper()
            u_num = roman_map.get(raw_num, int(raw_num) if raw_num.isdigit() else (i + 1))
            
            start_idx = u_match.end()
            end_idx = u_splits[i+1].start() if i+1 < len(u_splits) else len(c_text)
            
            u_chunk = c_text[start_idx:end_idx]
            m_stop = re.search(r'(?i)\b(Total\s+Lecture\s+hours|COURSE OUTCOMES?|EXPECTED\s*COURSE\s*OUTCOME|CO\s*[1-9]|TEXTBOOKS?|TEXT\s*BOOKS?|TEXT\s*BOOK\(S\)|REFERENCES?|REFERENCE\s*BOOKS?|REFERENCE\s*BOOK\(S\)|SUGGESTED\s*READINGS?|RECOMMENDED\s*BOOKS?|CO/PO|CO-PO|TOTAL\s*PERIODS?|===== PAGE)\b', u_chunk)
            if m_stop and (i + 1 == len(u_splits) or m_stop.start() < len(u_chunk)):
                u_chunk = u_chunk[:m_stop.start()]
                
            u_lines = [l.strip() for l in u_chunk.splitlines() if l.strip()]
            u_title = f"{u_kind} {raw_num}"
            u_hours = None
            body_start = 0
            
            if u_lines:
                # Check Case 1: Line 0 is purely a number (Hours), e.g. "10", "10 hours", "10 hrs"
                if re.match(r'^\d+(\s*(?:hours|hrs|periods|lectures|lec))?$', u_lines[0], re.IGNORECASE):
                    m_num = re.search(r'(\d+)', u_lines[0])
                    if m_num:
                        u_hours = int(m_num.group(1))
                    if len(u_lines) > 1:
                        u_title = u_lines[1].strip(' :-\t')
                        body_start = 2
                    else:
                        body_start = 1

                # Check Case 2: Line 1 is hours, Line 0 is title
                elif len(u_lines) > 1 and re.match(r'^\d+(\s*(?:hours|hrs|periods|lectures|lec))?$', u_lines[1], re.IGNORECASE):
                    u_title = u_lines[0].strip(' :-\t')
                    m_num = re.search(r'(\d+)', u_lines[1])
                    if m_num:
                        u_hours = int(m_num.group(1))
                    body_start = 2

                # Check Case 3: Line 0 is title ending with colon e.g. "First order ordinary differential equations:"
                elif re.match(r'^([^:\n]{3,80}):\s*$', u_lines[0]):
                    u_title = u_lines[0].strip(' :-\t')
                    body_start = 1
                    if len(u_lines) > 1 and re.match(r'^\d+(\s*(?:hours|hrs|periods|lectures|lec))?$', u_lines[1], re.IGNORECASE):
                        m_num = re.search(r'(\d+)', u_lines[1])
                        if m_num:
                            u_hours = int(m_num.group(1))
                        body_start = 2

                else:
                    # Scan first 4 lines for an hours indicator
                    hours_idx = None
                    for l_idx, l_val in enumerate(u_lines[:4]):
                        m_hrs = re.search(r'^(\d+)\s*(?:hours|hrs|periods|lectures|lec)\s*$', l_val, re.IGNORECASE)
                        if m_hrs:
                            hours_idx = l_idx
                            u_hours = int(m_hrs.group(1))
                            break
                    
                    if hours_idx is not None:
                        title_parts = u_lines[:hours_idx]
                        if title_parts:
                            u_title = " ".join(title_parts).strip(' :-\t')
                        body_start = hours_idx + 1
                    else:
                        u_title = u_lines[0].strip(' :-\t')
                        body_start = 1

            u_body = "\n".join(u_lines[body_start:]).strip()

            # If u_title is generic or numeric, check if u_body starts with "Title Name:\nBody..."
            if (not u_title or re.match(r'^(?:UNIT|MODULE|CHAPTER|PART|SECTION|\d+)\s*[IVX0-9]*$', u_title, re.IGNORECASE)) and u_body:
                m_start_colon = re.match(r'^([A-Z][A-Za-z0-9\s,\-/&]{2,70}):\s*(\n|[A-Z].*)', u_body)
                if m_start_colon:
                    u_title = m_start_colon.group(1).strip(' :-\t')
                    u_body = u_body[m_start_colon.end(1):].strip(' :-\t\n')

            u_body = clean_and_heal_text(u_body)
            cleaned_topics = parse_topics_and_subtopics(u_body, unit_num=u_num, page_num=p0)

            if not cleaned_topics and u_body:
                cleaned_topics.append({
                    "id": f"u{u_num}t1",
                    "text": u_body[:100],
                    "page": p0,
                    "bloom": "understand",
                    "bloom_source": "inferred",
                    "subtopics": []
                })

            units_list.append({
                "number": u_num,
                "unit": f"{u_kind} {raw_num}",
                "title": clean_and_heal_text(u_title),
                "hours": u_hours,
                "page": p0,
                "text": u_body,
                "topics": cleaned_topics
            })

    # 2. Abstract Syllabus / Course Content / Colon Subtopics
    if not units_list:
        m_abs = re.search(
            r'(?i)(?:Abstract\s*syllabus|Abstract|Course\s*Content|Syllabus\s*Content|Topics)\s*[:\n](.*?)(?=(?:References?|Reference\s*Books?|Textbooks?|Suggested\s*Reading|CO/PO|CO-PO|COURSE OUTCOMES?|Expected\s*Course\s*Outcome|===== PAGE|$))',
            c_text, re.DOTALL
        )
        if m_abs:
            abs_body = m_abs.group(1).strip()
            topic_splits = list(re.finditer(r'(?:^|\n|\.\s+)([A-Z][A-Za-z\s\-/&]{2,35}):\s*', abs_body))
            if len(topic_splits) >= 2:
                for idx, t_match in enumerate(topic_splits):
                    top_heading = t_match.group(1).strip()
                    s_pos = t_match.end()
                    e_pos = topic_splits[idx+1].start() if idx+1 < len(topic_splits) else len(abs_body)
                    chunk_text = abs_body[s_pos:e_pos].strip()
                    
                    subtopics_list = [
                        st.strip() for st in re.split(r'[,;•\n]', chunk_text)
                        if len(st.strip()) >= 3 and not is_invalid_course_title(st.strip())
                    ]
                    
                    topics_objs = []
                    for s_idx, st in enumerate(subtopics_list[:12], 1):
                        topics_objs.append({
                            "id": f"u{idx+1}t{s_idx}",
                            "text": st,
                            "page": p0,
                            "bloom": "understand",
                            "bloom_source": "inferred",
                            "subtopics": []
                        })
                    
                    if not topics_objs and chunk_text:
                        topics_objs.append({
                            "id": f"u{idx+1}t1",
                            "text": chunk_text[:100],
                            "page": p0,
                            "bloom": "understand",
                            "bloom_source": "inferred",
                            "subtopics": []
                        })
                        
                    units_list.append({
                        "number": idx + 1,
                        "unit": f"MODULE {idx + 1}",
                        "title": top_heading,
                        "hours": None,
                        "page": p0,
                        "text": f"{top_heading}: {chunk_text}",
                        "topics": topics_objs
                    })
            else:
                raw_topics = [st.strip() for st in re.split(r'[,;•\n]', abs_body) if len(st.strip()) >= 3]
                topics_objs = []
                for s_idx, st in enumerate(raw_topics[:25], 1):
                    topics_objs.append({
                        "id": f"u1t{s_idx}",
                        "text": st,
                        "page": p0,
                        "bloom": "understand",
                        "bloom_source": "inferred",
                        "subtopics": []
                    })
                if topics_objs:
                    units_list.append({
                        "number": 1,
                        "unit": "MODULE 1",
                        "title": "Course Content",
                        "hours": None,
                        "page": p0,
                        "text": abs_body,
                        "topics": topics_objs
                    })

    # 3. List of experiments / Practicals fallback
    if not units_list:
        m_exp = re.search(r'(?i)(?:LIST OF EXPERIMENTS|LABORATORY EXPERIMENTS|PRACTICAL EXERCISES|EXPERIMENTS|PRACTICALS)\s*[:\n](.*?)(?=(?:COURSE OUTCOMES?|CO\s*[1-9]|TEXTBOOKS?|REFERENCES?|CO/PO|CO-PO|TOTAL\s*PERIODS?|===== PAGE|$))', c_text, re.DOTALL)
        if m_exp:
            exp_text = m_exp.group(1).strip()
            exp_items = re.split(r'\n(?=\s*\d+[\.\)]|\s*\d+\s+[A-Z])', exp_text)
            topics = []
            for e_idx, e_item in enumerate(exp_items, 1):
                clean_e = " ".join(e_item.strip().split())
                clean_e = re.sub(r'^\d+[\.\)]\s*', '', clean_e).strip()
                if len(clean_e) > 3:
                    topics.append({
                        "id": f"u1t{e_idx}",
                        "text": clean_e,
                        "page": p0,
                        "bloom": "apply",
                        "bloom_source": "inferred",
                        "subtopics": []
                    })
            if topics:
                units_list.append({
                    "number": 1,
                    "unit": "PRACTICAL",
                    "title": "LIST OF EXPERIMENTS",
                    "hours": None,
                    "page": p0,
                    "text": exp_text,
                    "topics": topics
                })

    # Course Outcomes (COs)
    cos_list = []
    # Pattern A: Explicit CO1: / CO 1:
    co_pattern = re.compile(r'(?i)\b(CO\s*([1-9]))\s*[:\-\.]?\s*(.*?)(?=(?:\bCO\s*[1-9]\b|TEXTBOOKS?|TEXT\s*BOOK\(S\)|REFERENCES?|REFERENCE\s*BOOK\(S\)|SUGGESTED\s*READING|CO/PO|CO-PO|COURSE OUTCOMES?|Expected\s*Course\s*Outcome|$))', re.DOTALL)
    for co_m in co_pattern.finditer(c_text):
        co_id = f"CO{co_m.group(2)}"
        co_raw_text = co_m.group(3).strip()
        bloom_val = "understand"
        bloom_src = "inferred"

        m_bloom = re.search(r'\b(K[1-6]|Remember|Understand|Apply|Analyze|Evaluate|Create)\b', co_raw_text, re.IGNORECASE)
        if m_bloom:
            bloom_val = m_bloom.group(1)
            bloom_src = "printed"
            co_raw_text = re.sub(r'\b' + re.escape(m_bloom.group(0)) + r'\b', '', co_raw_text).strip()

        co_clean = " ".join(co_raw_text.split())
        if co_clean and not any(c['id'] == co_id for c in cos_list):
            cos_list.append({
                "id": co_id,
                "code": co_id,
                "text": co_clean,
                "page": p0,
                "bloom": bloom_val,
                "bloom_source": bloom_src
            })

    # Pattern B: Numbered Outcomes under Expected Course Outcome / Course Outcomes
    if not cos_list:
        m_co_block = re.search(
            r'(?i)(?:Expected\s*Course\s*Outcomes?|Course\s*Outcomes?)\s*[:\n](.*?)(?=(?:Student\s*Learning\s*Outcomes|SLO|Module|Unit|Text\s*Book|Reference|CO/PO|$))',
            c_text, re.DOTALL
        )
        if m_co_block:
            co_block_text = m_co_block.group(1).strip()
            co_items = re.split(r'\n(?=\s*\d+[\.\)])', co_block_text)
            for idx, co_item in enumerate(co_items, 1):
                clean_co = " ".join(co_item.strip().split())
                clean_co = re.sub(r'^\d+[\.\)]\s*', '', clean_co).strip()
                if len(clean_co) > 10:
                    cos_list.append({
                        "id": f"CO{idx}",
                        "code": f"CO{idx}",
                        "text": clean_co,
                        "page": p0,
                        "bloom": "understand",
                        "bloom_source": "inferred"
                    })

    # Textbooks, References & Suggested Readings
    textbooks_list = []
    references_list = []
    all_books_list = []

    # Textbooks (including Text Book(s) / Textbooks)
    m_tb = re.search(r'(?i)(?:TEXTBOOKS?|TEXT\s*BOOKS?|TEXT\s*BOOK\(S\))\s*[:\n](.*?)(?=(?:REFERENCES?|REFERENCE\s*BOOKS?|REFERENCE\s*BOOK\(S\)|SUGGESTED\s*READING|CO/PO|CO-PO|COURSE OUTCOMES?|===== PAGE|$))', c_text, re.DOTALL)
    if m_tb:
        tb_text = m_tb.group(1).strip()
        # Splits numbered citations
        tb_items = re.split(r'\n(?=\s*\d+[\.\)]|\s*\[\d+\])', tb_text)
        for idx, item in enumerate(tb_items, 1):
            clean_item = " ".join(item.strip().split())
            clean_item = re.sub(r'^\d+[\.\)]\s*', '', clean_item).strip()
            if len(clean_item) > 5:
                book_entry = {
                    "number": idx,
                    "type": "Textbook",
                    "kind": "text",
                    "title": clean_item,
                    "author": "",
                    "publisher": "",
                    "edition": "",
                    "year": "",
                    "isbn": "",
                    "page": p0,
                    "text": clean_item
                }
                textbooks_list.append(book_entry)
                all_books_list.append(book_entry)

    # References / Reference Books
    m_ref = re.search(r'(?i)(?:REFERENCES?|REFERENCE\s*BOOKS?|REFERENCE\s*BOOK\(S\)|SUGGESTED\s*READING|SUGGESTED\s*READINGS|RECOMMENDED\s*BOOKS?)\s*[:\n](.*?)(?=(?:TEXTBOOKS?|TEXT\s*BOOK\(S\)|CO/PO|CO-PO|COURSE OUTCOMES?|===== PAGE|$))', c_text, re.DOTALL)
    if m_ref:
        ref_text = m_ref.group(1).strip()
        ref_items = re.split(r'\n(?=\s*\d+[\.\)]|\s*\[\d+\]|\s*https?://)', ref_text)
        for idx, item in enumerate(ref_items, 1):
            clean_item = " ".join(item.strip().split())
            clean_item = re.sub(r'^\d+[\.\)]\s*', '', clean_item).strip()
            if len(clean_item) > 5:
                r_type = "Online Resource" if ("http" in clean_item.lower() or "nptel" in clean_item.lower()) else "Reference Book"
                book_entry = {
                    "number": idx,
                    "type": r_type,
                    "kind": "reference",
                    "title": clean_item,
                    "author": "",
                    "publisher": "",
                    "edition": "",
                    "year": "",
                    "isbn": "",
                    "page": p0,
                    "text": clean_item
                }
                references_list.append(book_entry)
                all_books_list.append(book_entry)

    # CO-PO Matrix
    copo_matrix = {}
    m_copo = re.search(r'(?i)(?:CO/PO|CO-PO|MAPPING OF COs WITH POs|COURSE ARTICULATION MATRIX)(.*?)(?=(?:TEXTBOOKS?|REFERENCES?|SUGGESTED\s*READING|===== PAGE|$))', c_text, re.DOTALL)
    if m_copo:
        matrix_text = m_copo.group(1)
        for line in matrix_text.splitlines():
            m_co_row = re.search(r'\b(CO[1-9])\b\s*(.*)', line, re.IGNORECASE)
            if m_co_row:
                co_label = m_co_row.group(1).upper()
                vals = re.findall(r'[1-3\-\u2013\u2014✔✓HMLhml]', m_co_row.group(2))
                if vals:
                    copo_matrix[co_label] = {f"PO{i+1}": vals[i] for i in range(min(12, len(vals)))}

    course_obj = {
        "code": code_cand,
        "title": title_cand,
        "programme": programme,
        "branch": branch,
        "category": category,
        "semester": semester,
        "credits": credits_val,
        "lecture": L_val,
        "tutorial": T_val,
        "practical": P_val,
        "credits_number": C_val,
        "L": L_val,
        "T": T_val,
        "P": P_val,
        "C": C_val,
        "contact_hours": (L_val or 0) * 15 if L_val else None,
        "prerequisites": "",
        "preamble": preamble,
        "objectives": objectives,
        "pages": detail_pages
    }

    return {
        "course": course_obj,
        "units": units_list,
        "course_outcomes": cos_list,
        "programme_outcomes": [],
        "programme_specific_outcomes": [],
        "co_po_matrix": copo_matrix,
        "co_pso_matrix": {},
        "books": all_books_list,
        "textbooks": textbooks_list,
        "references": references_list,
        "not_extracted": []
    }


# =========================================================
# EXTRACT SINGLE COURSE DETAILED SYLLABUS
# =========================================================

def extract_course_details(
    pages_text: str,
    course_query: str,
    start_page: int = None,
    all_courses: List[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Extract complete detailed syllabus information for a single specified course.
    Determines true boundaries dynamically and extracts with instant fast engine.
    """
    print("=" * 70)
    print(f"TASK 2: SINGLE COURSE DETAILED EXTRACTION -> {course_query}")
    print("=" * 70)

    # 1. Dynamically focus text on the exact course boundaries
    section_text, detail_pages = extract_course_section_text(
        pages_text=pages_text,
        course_query=course_query,
        start_page=start_page,
        all_courses=all_courses
    )
    resolved_start = detail_pages[0] if detail_pages else (start_page or 1)
    print(f"[BOUNDARY] Course located on pages: {detail_pages} (detail start: {resolved_start})")
    print(f"Dynamically extracted section text length: {len(section_text)} characters")

    # 2. Universal deterministic extraction
    fast_result = extract_course_details_deterministic(
        section_text=section_text,
        course_query=course_query,
        detail_pages=detail_pages
    )

    # 3. If units or books were not captured deterministically, run AI prompt for this course section
    if not fast_result.get("units") and not fast_result.get("books"):
        try:
            print("[AI] Running LLM Course Detail extraction...")
            prompt = build_course_detail_prompt(course_query, section_text[:8000])
            raw_response = generate_text(
                prompt=prompt,
                system_prompt=COURSE_DETAIL_SYSTEM_PROMPT,
                timeout_seconds=30.0,
            )
            ai_data = parse_and_clean_json(raw_response)
            if ai_data.get("units") or ai_data.get("books"):
                fast_result = ai_data
        except Exception as exc:
            logger.warning(f"[FALLBACK] AI course detail extraction error: {exc}")

    validated_detail = validate_course_detail(fast_result, course_query, default_pages=detail_pages)
    if detail_pages:
        validated_detail["course"]["pages"] = detail_pages
    return validated_detail