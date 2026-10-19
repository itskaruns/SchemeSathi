"""PDF ingestion with PyMuPDF: scheme guideline PDF -> classified chunks -> data/ingested_chunks.json.

Usage:  python ingest.py data/guidelines/PMS-SC.pdf PMS-SC
        python ingest.py --demo        (creates data/guidelines/PMS-SC.pdf from the existing record, then ingests it)
The PDF must use section headings: Overview / Eligibility / Documents / Benefits / How to apply.
Only these sample PDFs are ingested in this PoC; the other schemes still use hand-written chunks.
"""
import json, re, sys
from pathlib import Path
import pymupdf

DATA = Path(__file__).parent / "data"
HEADS = {"overview": r"overview", "elig": r"eligibility", "docs": r"documents?( required)?",
         "benefit": r"benefits?", "apply": r"how to apply|application"}


def pdf_to_chunks(pdf_path):
    text = "\n".join(page.get_text() for page in pymupdf.open(pdf_path))
    chunks, kind = {}, None
    for line in (l.strip() for l in text.splitlines()):
        if not line:
            continue
        hit = next((k for k, pat in HEADS.items() if len(line) < 30 and re.fullmatch(pat, line, re.I)), None)
        if hit:
            kind = hit; continue
        if kind:
            chunks[kind] = (chunks.get(kind, "") + " " + line).strip()
    return chunks


def ingest(pdf_path, scheme_id):
    f = DATA / "ingested_chunks.json"
    store = json.loads(f.read_text()) if f.exists() else {}
    store[scheme_id] = {"source": Path(pdf_path).name, "chunks": pdf_to_chunks(pdf_path)}
    f.write_text(json.dumps(store, indent=1))
    return store[scheme_id]


def make_demo_pdf(scheme_id="PMS-SC"):
    s = next(x for x in json.loads((DATA / "schemes.json").read_text()) if x["id"] == scheme_id)
    titles = {"overview": "Overview", "elig": "Eligibility", "docs": "Documents", "benefit": "Benefits", "apply": "How to apply"}
    doc = pymupdf.open(); page = doc.new_page(); y = 60
    page.insert_text((50, y), f"{s['name']} - guidelines (sample)", fontsize=14); y += 30
    for k, t in s["chunks"].items():
        page.insert_text((50, y), titles[k], fontsize=12); y += 18
        words, line = t.split(), ""
        for w in words:
            if len(line) + len(w) > 95:
                page.insert_text((50, y), line, fontsize=9); y += 13; line = ""
            line += w + " "
        page.insert_text((50, y), line, fontsize=9); y += 26
    out = DATA / "guidelines" / f"{scheme_id}.pdf"; out.parent.mkdir(exist_ok=True); doc.save(out)
    return out


if __name__ == "__main__":
    if sys.argv[1:] == ["--demo"]:
        p = make_demo_pdf(); print("created", p); print(ingest(p, "PMS-SC")["source"], "ingested")
    else:
        print(json.dumps(ingest(sys.argv[1], sys.argv[2]), indent=1))
