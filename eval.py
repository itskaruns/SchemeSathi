"""Evaluation harness: synthetic students x all schemes (hand-labelled cases) + critic fault-injection."""
import copy, json, sys
from pathlib import Path
from schemesathi import agents, graph

students = json.loads((Path(__file__).parent / "data" / "students.json").read_text())
ok = total = loops = 0
for st in students:
    prof = {k: v for k, v in st.items() if k != "expected"}
    out = graph.run(prof, use_llm=False, today="2026-10-18")
    loops += out["retry"]
    got = {r["scheme_id"]: r["status"] for r in out["assessments"]}
    for sid, exp in st["expected"].items():
        total += 1; ok += got.get(sid) == exp
        if got.get(sid) != exp:
            print(f"MISMATCH {st['name']} {sid}: expected {exp}, got {got.get(sid)}")
print(f"Eligibility accuracy: {ok}/{total} ({len(students)} students x {len(agents.SCHEMES)} schemes)  | critic-triggered retry loops across {len(students)} students: {loops}")

# fault injection: corrupt the executor's claims and confirm the critic catches each one
prof = {k: v for k, v in students[0].items() if k != "expected"}
caught = 0; tests = 0
def inject(mut):
    global caught, tests
    s = {"profile": prof, "use_llm": False, "today": "2026-10-18", "log": [], "excluded": {}}
    s = agents.planner(s); s = agents.research(s); s["retry"] = 2   # full evidence
    s = agents.executor(s); mut(s); s = agents.critic(s)
    tests += 1; caught += any(v["verdict"] in ("reassess", "uncertain") for v in s["verdicts"].values()) or \
        any(r["status"] == "More information needed" for r in s["assessments"])
def flip_result(s):
    for r in s["assessments"]:
        for c in r["claims"]:
            if c["type"] == "criterion" and c["field"] == "income": c["result"] = "fail" if c["result"] == "pass" else "pass"; return
def fake_key(s):
    for r in s["assessments"]:
        for c in r["claims"]:
            if c["type"] == "criterion" and c["key"]: c["key"] = "9,99,999"; return
def fake_doc(s):
    for r in s["assessments"]:
        for c in r["claims"]:
            if c["type"] == "checklist": c["items"].append("Passport"); return
def fake_link(s):
    for r in s["assessments"]:
        for c in r["claims"]:
            if c["type"] == "link": c["url"] = "https://fake.example"; return
for f in (flip_result, fake_key, fake_doc, fake_link): inject(f)
print(f"Critic fault-injection: caught {caught}/{tests} deliberately wrong claims")

# profile-extraction regression tests (inputs that previously failed: city names, "1 million", names)
from schemesathi import llm
EX = [("I'm Riya Sharma from updaipur, family income 1 million, 82%", {"name": "Riya Sharma", "state": "Rajasthan", "income": 1000000, "marks": 82.0}),
      ("girl from Pune, parents earn Rs 10,00,000, B.Sc 2nd year", {"state": "Maharashtra", "income": 1000000, "course": "B.Sc", "year": 2, "gender": "Female"}),
      ("income 1.5 crore, from Chennai", {"income": 15000000, "state": "Tamil Nadu"}),
      ("Aarav, 5 lakh income, SC category student from Bihar", {"name": "Aarav", "income": 500000, "category": "SC", "state": "Bihar"})]
exok = 0
for t, want in EX:
    got = llm.extract_profile(t, False)[0]
    good = all(got.get(k) == v for k, v in want.items()); exok += good
    if not good: print("EXTRACT MISMATCH:", t, got)
print(f"Profile extraction: {exok}/{len(EX)} regression sentences parsed correctly")

# --- new component tests -------------------------------------------------------------------------------------
import json as _j
from schemesathi import retrieval, llm as _llm
import ingest as _ing
res = []
# 1. vector store + re-rank: a documents query must surface the 'docs' chunk first for every scheme
store = retrieval.build_index(agents.SCHEMES)
res.append(("vector store+rerank puts 'docs' chunk first for a documents query",
            all(retrieval.search(store, "documents required certificate", s["id"], 1, s["keys"])[0]["kind"] == "docs" for s in agents.SCHEMES)))
# 2. re-ranker reorders by overlap (final score >= vector score ordering is not required, but scores must be present)
h = retrieval.search(store, "eligibility income", "PMS-SC", 3)
res.append(("re-ranker returns vector_score and final score", all("vector_score" in x and "score" in x for x in h)))
# 3. PDF ingestion round-trip: PyMuPDF text from the sample PDF equals the hand-written chunks
pdf = _ing.make_demo_pdf("PMS-SC"); got = _ing.pdf_to_chunks(pdf)
orig = next(s for s in _j.loads((Path(__file__).parent / "data" / "schemes.json").read_text()) if s["id"] == "PMS-SC")["chunks"]
res.append(("PyMuPDF ingestion reproduces all 5 chunks of the sample PDF", got == orig))
# 4. claim extraction splits an explanation into several atomic claims
cl = _llm.extract_claims("Not a match. income Rs 300,000 vs limit Rs 250,000; category SC vs allowed SC.", False)
res.append(("claim extraction yields >= 2 atomic claims", len(cl) >= 2))
# 5. judge: correct limit supported, wrong limit unsupported, student-only fact not checkable
ev = orig["elig"]
res.append(("judge: true scheme limit supported", _llm.judge("income Rs 3,00,000 vs limit Rs 250,000", ev, False) == "supported"))
res.append(("judge: fabricated scheme limit unsupported", _llm.judge("income Rs 300,000 vs limit Rs 999,999", ev, False) == "unsupported"))
res.append(("judge: student-only fact not checkable", _llm.judge("category SC", ev, False) == "not_checkable"))
# 6. critic catches a fabricated limit inside the explanation text
s = {"profile": prof, "use_llm": False, "today": "2026-10-18", "log": [], "excluded": {}}
s = agents.planner(s); s = agents.research(s); s["retry"] = 2; s = agents.executor(s)
for r in s["assessments"]:
    if any(c["field"] == "income" for c in r["checks"]): r["explanation"] += " Income limit Rs 9,99,999."; break
s = agents.critic(s)
res.append(("critic flags fabricated limit in explanation (LLM-judge path)",
            any("not supported by source" in i for v in s["verdicts"].values() for i in v["issues"])))
for name, passed in res: print(("PASS " if passed else "FAIL ") + name)
print(f"New components: {sum(p for _, p in res)}/{len(res)} tests passed  (mock mode: embeddings are hashed, judge is string-match)")
sys.exit(0 if ok == total and caught == tests and exok == len(EX) and all(p for _, p in res) else 1)
