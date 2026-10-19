"""LLM wrapper: uses local Ollama if reachable, otherwise a deterministic mock.
The active mode is exposed so the UI can label it (never hide the mock)."""
import json, os, re, urllib.request

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")


def ollama_available() -> bool:
    if os.environ.get("SCHEMESATHI_FORCE_MOCK") == "1":
        return False
    try:
        urllib.request.urlopen(OLLAMA_URL + "/api/tags", timeout=1)
        return True
    except Exception:
        return False


def _ollama(prompt: str, as_json=False) -> str:
    body = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
    if as_json:
        body["format"] = "json"
    req = urllib.request.Request(OLLAMA_URL + "/api/generate", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=90).read())["response"]


# ---- task 1: read an eligibility passage and list the TEXTUAL rules in it ----
RULE_PATTERNS = [  # mock interpreter: phrase in text -> profile field it constrains
    (r"must not be receiving any other scholarship", "other_scholarship", "No"),
]


def interpret_rules(text: str, use_llm: bool):
    """Returns list of {rule, field, required}. Falls back to the mock on any failure."""
    if use_llm:
        prompt = ("Read this scholarship eligibility text. List ONLY conditions about whether the student "
                  "already receives another scholarship. Reply JSON: {\"rules\":[{\"rule\":\"<EXACT_QUOTE_FROM_TEXT>\","
                  "\"field\":\"other_scholarship\",\"required\":\"No\"}]}. The 'rule' MUST be an exact copy-pasted sentence from the text. Empty list if none.\n\n" + text)
        try:
            data = json.loads(_ollama(prompt, as_json=True))
            rules = []
            for r in data.get("rules", []):
                if r.get("field") == "other_scholarship":
                    rule_text = str(r.get("rule", ""))
                    
                    # 1. Force the correct string type to prevent boolean matching errors
                    r["required"] = "No" 
                    r["rule"] = rule_text
                    
                    # 2. Drop LLM hallucinations if the rule isn't actually in the source text
                    if len(rule_text) > 10 and rule_text in text:
                        rules.append(r)
            return rules
        except Exception:
            pass
    return [{"rule": m.group(0), "field": f, "required": req}
            for pat, f, req in RULE_PATTERNS if (m := re.search(pat, text, re.I))]


# ---- task 2: explanation ----
def explain(scheme_name: str, status: str, checks: list, use_llm: bool) -> str:
    facts = "; ".join(f"{c['label']}: {c['detail']}" for c in checks)
    if use_llm:
        prompt = (f"In 2 plain sentences for a student, explain why the result for '{scheme_name}' is "
                  f"'{status}'. Use ONLY these facts, add nothing else: {facts}")
        try:
            return _ollama(prompt).strip()
        except Exception:
            pass
    lead = {"Potential match": "All known conditions are satisfied.",
            "Not a match": "At least one known condition is not satisfied.",
            "More information needed": "No condition failed, but some details are missing."}[status]
    return f"{lead} {facts}."


# ---- task 2b (Critic): split an explanation into atomic claims, then judge each against the evidence ----
def extract_claims(explanation: str, use_llm: bool):
    """Explanation text -> list of atomic claim strings. Ollama: JSON list. Mock: sentence / clause split."""
    if use_llm:
        prompt = ("Split this text into short atomic factual claims, one fact each. "
                  "Reply JSON: {\"claims\":[str]}.\n\n" + explanation)
        try:
            cl = json.loads(_ollama(prompt, as_json=True)).get("claims", [])
            if cl and all(isinstance(c, str) for c in cl):
                return cl
        except Exception:
            pass
    return [c.strip() for c in re.split(r"(?<=[.;])\s+|;\s*", explanation) if len(c.strip()) > 3]


def _nums(text: str):
    return {n.replace(",", "") for n in re.findall(r"\d[\d,]*(?:\.\d+)?", text)}


def judge(claim: str, evidence: str, use_llm: bool):
    """Is the SCHEME-SIDE content of the claim supported by the evidence? -> 'supported' | 'unsupported' | 'not_checkable'.
    Ollama: yes/no entailment prompt. Mock: every scheme-limit number in the claim (after 'limit' / 'minimum')
    must appear in the evidence text (string match, NOT real entailment). Student-side facts are not judged here."""
    m = re.search(r"(?:limit|minimum)\s*(?:Rs\s*)?(\d[\d,]*(?:\.\d+)?)", claim, re.I)
    if not m:
        return "not_checkable"
    if use_llm:
        prompt = (f"EVIDENCE: {evidence}\nCLAIM about the scheme's limit: {claim}\n"
                  "Does the evidence support the scheme-side number in the claim? Answer only YES or NO.")
        try:
            return "supported" if _ollama(prompt).strip().upper().startswith("YES") else "unsupported"
        except Exception:
            pass
    return "supported" if m.group(1).replace(",", "") in _nums(evidence) else "unsupported"


# ---- task 3: chat assistant (grounded in pipeline results + retrieved guideline text) ----
CHAT_RULES = (
    "You are SchemeSathi, an assistant that helps Indian students with scholarship schemes. "
    "Answer ONLY from the CONTEXT below. If the answer is not in the context, say you don't know and point to the "
    "official portal. Never promise that the student will receive a scholarship; eligibility results are "
    "'potential' and must be confirmed on the official portal. If the student wants to change their details, tell "
    "them to update the profile and re-run the agents. Be brief and friendly.")


def chat(question: str, history: list, context: str, use_llm: bool):
    """Returns (answer, mode). Without Ollama this is retrieval-only, and says so."""
    if use_llm:
        convo = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in history[-6:])
        prompt = f"{CHAT_RULES}\n\nCONTEXT:\n{context}\n\nCONVERSATION:\n{convo}\nUSER: {question}\nASSISTANT:"
        try:
            return _ollama(prompt).strip(), "Ollama"
        except Exception:
            pass
    return ("⚠️ Mock mode (Ollama not running), so I can't hold a real conversation. "
            "Here is the most relevant information I found:\n\n" + context), "Mock"


# ---- task 4: turn a free-text self-description into profile fields ----
from .geo import STATES, normalize_state, find_state_in_text

COURSE_WORDS = [(r"\bdiploma\b", "Diploma (Engineering)"), (r"\bm\.?\s?tech\b|\bmaster of technology\b", "M.Tech"),
                (r"\bb\.?\s?tech\b|\bb\.\s?e\b|\bbachelor of engineering\b|\bengineering\b", "B.Tech / B.E."),
                (r"\bmba\b|\bpgdm\b", "MBA"), (r"\bm\.?\s?sc\b", "M.Sc"), (r"\bb\.?\s?sc\b", "B.Sc"),
                (r"\bb\.?\s?com\b", "B.Com"), (r"\bmbbs\b", "MBBS"), (r"\bb\.\s?a\b|\bba\b", "B.A.")]
FIELDS = {"name": str, "state": str, "course": str, "year": int, "income": int, "marks": float, "category": str,
          "gender": str, "minority": str, "other_scholarship": str}
ALLOWED = {"category": {"General", "EWS", "OBC", "SC", "ST"}, "gender": {"Female", "Male", "Other"},
           "minority": {"Yes", "No", "Unknown"}, "other_scholarship": {"Yes", "No", "Unknown"},
           "course": {c for _, c in COURSE_WORDS}}
UNITS = {"crore": 1e7, "crores": 1e7, "cr": 1e7, "million": 1e6, "millions": 1e6, "mn": 1e6, "m": 1e6,
         "lakh": 1e5, "lakhs": 1e5, "lac": 1e5, "lacs": 1e5, "lpa": 1e5, "l": 1e5,
         "thousand": 1e3, "k": 1e3}
_NUM = r"(\d[\d,]*(?:\.\d+)?)\s*(crores?|cr|millions?|mn|m|lakhs?|lacs?|lpa|l|thousand|k)?\b(?!\s*(?:%|percent))"
WORD_NUMS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}


def parse_money(value):
    """'1 million', '10 lakh', '10,00,000', '₹1.5 cr', 1000000, '1000000' -> int rupees, else None."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value)
    if not isinstance(value, str):
        return None
    t = value.lower().replace("₹", " ").replace("rs.", " ").replace("rs", " ").replace("inr", " ")
    for w, n in WORD_NUMS.items():                      # "one million" -> "1 million"
        t = re.sub(rf"\b{w}\b(?=\s*(?:crore|million|lakh|lac|thousand))", str(n), t)
    m = re.search(_NUM, t)
    if not m:
        return None
    num = float(m.group(1).replace(",", ""))
    return int(num * UNITS.get(m.group(2) or "", 1))


def _find_income(low: str):
    key = re.search(r"(?:income|earn\w*|salary|ctc|package|lpa)\D{0,40}?(?=\d|(?:one|two|three|four|five|six|seven|eight|nine|ten)\b)", low)
    for m in ([key] if key else []):
        v = parse_money(low[m.end():m.end() + 30])
        if v:
            return v
    for m in re.finditer(_NUM, low):                    # any amount written with a money unit
        if m.group(2) and m.group(2) in UNITS and not re.match(r"\s*(?:st|nd|rd|th)\b", low[m.end():]):
            v = parse_money(m.group(0))
            if v and v >= 1000:
                return v
    return None


def _find_name(text: str):
    m = re.search(r"(?:my name is|name is|name:|i am called|call me|this is)\s+([A-Za-z][A-Za-z.\' -]{1,40}?)"
                  r"(?=\s*(?:[,.;\n]|\band\b|\bfrom\b|\bi\b|\bwho\b|\bstudying\b|$))", text, re.I)
    if not m:
        m = re.search(r"\b(?i:i am|i'm|im)\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)\b(?!\s*(?:years|yrs))", text)
    if not m:                                           # "Kavya from Chennai ..." / "Kavya, 20, ..."
        m = re.match(r"\s*([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)\s*(?:,|\bfrom\b|\bis a\b)", text)
    if m:
        first = m.group(1).lower().split()[0]
        if (first in {"a", "an", "the", "from", "in", "doing", "studying", "female", "male", "general", "obc", "sc",
                      "st", "ews", "muslim", "christian", "my", "hi", "hello"}
                or normalize_state(m.group(1)) or find_state_in_text(m.group(1))
                or any(re.search(pt, m.group(1).lower()) for pt, _ in COURSE_WORDS)):
            m = None
    return " ".join(w.capitalize() for w in m.group(1).split()) if m else None


def _mock_extract(t: str) -> dict:
    o, low = {}, t.lower()
    if n := _find_name(t):
        o["name"] = n
    if (inc := _find_income(low)) is not None:
        o["income"] = inc
    if m := re.search(r"(\d{1,3}(?:\.\d+)?)\s*(?:%|percent|percentage)", low):
        o["marks"] = float(m.group(1))
    elif m := re.search(r"(?:marks|scored|score|percentage)\D{0,12}(\d{2,3}(?:\.\d+)?)\b", low):
        o["marks"] = float(m.group(1))
    if st := find_state_in_text(t):
        o["state"] = st
    for pat, c in COURSE_WORDS:
        if re.search(pat, low):
            o["course"] = c; break
    if re.search(r"\bobc\b", low): o["category"] = "OBC"
    elif re.search(r"\bews\b|economically weaker", low): o["category"] = "EWS"
    elif re.search(r"scheduled caste|(?:category|belong\w*\s+to|from)\s+(?:the\s+)?sc\b|(?<![.\w])sc\s+(?:category|student|candidate|quota)", low):
        o["category"] = "SC"
    elif re.search(r"scheduled tribe|(?:category|belong\w*\s+to|from)\s+(?:the\s+)?st\b|(?<![.\w])st\s+(?:category|student|candidate|quota)", low):
        o["category"] = "ST"
    elif re.search(r"(?<![.\w])SC\b(?!\.)", t):
        o["category"] = "SC"
    elif re.search(r"(?<![.\w])ST\b(?!\.)", t):
        o["category"] = "ST"
    elif re.search(r"general|open category|unreserved", low):
        o["category"] = "General"
    if re.search(r"\bfemale\b|\bgirl\b|\bshe\b|\bwoman\b|\bdaughter\b", low):
        o["gender"] = "Female"
    elif re.search(r"\bmale\b|\bboy\b|\bhe\b|\bman\b|\bson\b", low):
        o["gender"] = "Male"
    if re.search(r"\bnon[- ]?minority\b|\bnot (?:a |from a )?minority\b|\bno minority\b", low):
        o["minority"] = "No"
    elif re.search(r"muslim|christian|sikh|buddhist|jain|parsi|zoroastrian|minority", low):
        o["minority"] = "Yes"
    if m := re.search(r"(\d)(?:st|nd|rd|th)\s*year|year\s*(?:no\.?\s*)?([1-5])\b", low):
        o["year"] = int(m.group(1) or m.group(2))
    elif m := re.search(r"\b(first|second|third|fourth|final)\s*year", low):
        o["year"] = {"first": 1, "second": 2, "third": 3, "fourth": 4, "final": 4}[m.group(1)]
    if re.search(r"no other scholarship|(?:not|n't)\s+(?:receiving|getting|taking|have|having)\s+any\s+scholarship|"
                 r"no scholarship", low):
        o["other_scholarship"] = "No"
    elif re.search(r"(?:already|currently)\s+(?:get|getting|receiv\w*|have|having)\s+(?:a\s+)?scholarship", low):
        o["other_scholarship"] = "Yes"
    return o


def _clean(found: dict):
    """Validate & normalise. Returns (clean, dropped) where dropped = {field: reason}."""
    clean, dropped = {}, {}
    for k, v in (found or {}).items():
        if k not in FIELDS or v in (None, "", "null"):
            continue
        try:
            if k == "income":
                v = parse_money(v)
                if v is None or not (1000 <= v <= 1_000_000_000):
                    raise ValueError("income must be an amount in rupees (1,000 – 100 crore)")
            elif k == "marks":
                v = float(str(v).replace("%", "").strip())
                if not 0 <= v <= 100:
                    raise ValueError("marks must be 0–100 %")
            elif k == "year":
                v = int(float(v))
                if not 1 <= v <= 5:
                    raise ValueError("year must be 1–5")
            elif k == "state":
                n = normalize_state(str(v))
                if not n:
                    raise ValueError(f"'{v}' is not a recognised state or city")
                v = n
            elif k == "name":
                v = re.sub(r"\s+", " ", str(v)).strip(" .,'-")[:40]
                if len(v) < 2:
                    raise ValueError("name too short")
            else:
                v = FIELDS[k](v)
                if isinstance(v, str):
                    v = v.strip()
                    low = {x.lower(): x for x in ALLOWED.get(k, ())}
                    v = low.get(v.lower(), v)
                if k in ALLOWED and v not in ALLOWED[k]:
                    raise ValueError(f"'{v}' is not one of {sorted(ALLOWED[k])}")
        except (TypeError, ValueError) as e:
            dropped[k] = str(e)
            continue
        clean[k] = v
    return clean, dropped


def extract_profile(text: str, use_llm: bool):
    """Returns (fields, mode, notes). Rules parse numbers/places exactly; the LLM (if on) fills what rules missed.
    Anything invalid is dropped and reported in notes, never guessed."""
    rules, _ = _clean(_mock_extract(text))
    mode, llm_found = "Mock", {}
    if use_llm:
        prompt = ("Extract a student profile from the text. Reply JSON with ONLY the keys you are sure about from: "
                  "name (the student's own name), state (Indian state; if only a city is given, give its state), "
                  "course (one of B.Tech / B.E., B.Sc, B.A., B.Com, MBBS, Diploma (Engineering), M.Tech, MBA, M.Sc), "
                  "year (int 1-5), income (annual family income as a plain integer in rupees, e.g. 1 million = 1000000, "
                  "5 lakh = 500000), marks (percent, number), category (General/EWS/OBC/SC/ST), "
                  "gender (Female/Male/Other), minority (Yes/No), other_scholarship (Yes/No). "
                  "Do not guess missing values.\n\nTEXT: " + text)
        try:
            llm_found, mode = json.loads(_ollama(prompt, as_json=True)), "Ollama + rules"
        except Exception:
            llm_found, mode = {}, "Mock"
    llm_clean, dropped = _clean(llm_found) if llm_found else ({}, {})
    # exact rule-based values win for numbers/places; the LLM fills everything the rules did not find
    merged = {**llm_clean, **{k: v for k, v in rules.items() if k in ("income", "marks", "year", "state")}}
    for k, v in rules.items():
        merged.setdefault(k, v)
    notes = [f"Ignored {k}: {why}" for k, why in dropped.items()]
    return merged, mode, notes
