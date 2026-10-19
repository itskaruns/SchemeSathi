"""The four agents. Each is a plain function: state -> state update, so LangGraph can use them as nodes."""
from datetime import date
from . import llm
from .geo import STATES
from .retrieval import load_schemes, build_index, search, MODE as RETRIEVAL_MODE

MAX_RETRY = 2
TOPK = {0: 5, 1: 5, 2: 5}          # retrieval width grows on each reassess loop
COURSES = {  # course -> (level, technical?)
    "B.Tech / B.E.": ("UG", True), "B.Sc": ("UG", False), "B.A.": ("UG", False), "B.Com": ("UG", False),
    "MBBS": ("UG", False), "Diploma (Engineering)": ("Diploma", True), "M.Tech": ("PG", True),
    "MBA": ("PG", False), "M.Sc": ("PG", False)}
SCHEMES = load_schemes()
SCHEME_BY_ID = {s["id"]: s for s in SCHEMES}
STORE = build_index(SCHEMES)
UNKNOWN = {None, "", "Unknown"}


def log(state, agent, msg):
    state["log"].append({"agent": agent, "msg": msg})


# ------------------------------------------------------------------ 1. PLANNER
def planner(state):
    p = state["profile"]
    problems = [f for f in ("state", "course", "category", "gender") if p.get(f) in UNKNOWN]
    if p.get("state") not in STATES:
        problems.append("state (not a recognised state/UT)")
    if p.get("course") not in COURSES:
        problems.append("course (unsupported)")
    state["tasks"] = [{"id": i, "name": n, "status": "pending"} for i, n in enumerate(
        ["Validate profile", "Retrieve scheme criteria (RAG)", "Assess eligibility",
         "Verify against evidence", "Prepare checklist & report"])]
    state["tasks"][0]["status"] = "done" if not problems else "failed"
    state.update(retry=0, reassess_ids=None, evidence={}, assessments=[], verdicts={})
    log(state, "Planner", "Profile valid." if not problems else f"Profile problems: {problems}")
    log(state, "Planner", "Plan: " + " → ".join(t["name"] for t in state["tasks"][1:]))
    return state


# ------------------------------------------------------------------ 2. RESEARCH
def research(state):
    p, k = state["profile"], TOPK[state["retry"]]
    targets = state["reassess_ids"] or [s["id"] for s in SCHEMES]
    query = "eligibility criteria income marks category course"
    if state["retry"] > 0:
        query += " documents required list certificate apply deadline"
    state["excluded"] = state.get("excluded", {})
    for sid in targets:
        s = SCHEME_BY_ID[sid]
        if "ALL" not in s["criteria"]["states"] and p["state"] not in s["criteria"]["states"]:
            state["excluded"][sid] = f"Scheme is only for {', '.join(s['criteria']['states'])}"
            continue                                            # state pre-filter
        state["evidence"][sid] = search(STORE, query, sid, k, s["keys"])   # vector search + re-rank
    log(state, "Research", f"[{RETRIEVAL_MODE}] vector search + re-rank: top-{k} passages for {len(state['evidence'])} schemes "
                            f"(state pre-filter removed {len(state['excluded'])})"
                            + (f"; re-run for {targets}" if state["reassess_ids"] else ""))
    state["tasks"][1]["status"] = "done"
    return state


# ------------------------------------------------------------------ 3. EXECUTOR
def _check(field, label, ok, detail, key=None, unknown=False):
    return {"field": field, "label": label, "result": "unknown" if unknown else ("pass" if ok else "fail"),
            "detail": detail, "key": key}


def python_checks(p, s):
    """Exact, deterministic checks done in Python (never left to the LLM)."""
    c, keys, out = s["criteria"], s["keys"], []
    level, technical = COURSES[p["course"]]
    inc = p["income"]
    if c["income_max"] is not None:
        out.append(_check("income", "Family income", inc is not None and inc <= c["income_max"],
                          f"income Rs {inc:,} vs limit Rs {c['income_max']:,}" if inc is not None
                          else f"income not provided (limit Rs {c['income_max']:,})",
                          keys.get("income"), unknown=inc is None))
    if c["min_marks"] is not None:
        m = p["marks"]
        out.append(_check("marks", "Marks", m is not None and m >= c["min_marks"],
                          f"marks {m}% vs minimum {c['min_marks']}%" if m is not None
                          else f"marks not provided (minimum {c['min_marks']}%)", keys.get("marks"), unknown=m is None))
    if c["categories"]:
        out.append(_check("category", "Category", p["category"] in c["categories"],
                          f"category {p['category']} vs allowed {', '.join(c['categories'])}", keys.get("category")))
    if c["gender"]:
        out.append(_check("gender", "Gender", p["gender"] == c["gender"],
                          f"scheme is for {c['gender'].lower()} students", keys.get("gender")))
    if c["minority_required"]:
        mv = p["minority"]
        out.append(_check("minority", "Minority community", mv == "Yes",
                          f"minority status: {mv}", keys.get("minority"), unknown=mv in UNKNOWN))
    if c.get("courses"):
        out.append(_check("course", "Course", p["course"] in c["courses"],
                          f"{p['course']} vs allowed {', '.join(c['courses'])}", keys.get("course")))
    out.append(_check("level", "Course level", level in c["levels"],
                      f"{p['course']} is {level}; scheme covers {', '.join(c['levels'])}", keys.get("level")))
    if c["technical_only"]:
        out.append(_check("technical", "Technical course", technical,
                          f"{p['course']} {'is' if technical else 'is not'} a technical course", keys.get("level")))
    return out


def executor(state):
    p, use_llm = state["profile"], state["use_llm"]
    results = []
    for s in SCHEMES:
        sid = s["id"]
        if sid in state["excluded"]:
            res = {"scheme_id": sid, "status": "Not a match", "checks": [
                _check("state", "State", False, state["excluded"][sid] + f"; student is in {p['state']}")],
                "checklist": [], "claims": [], "explanation": state["excluded"][sid] + f"; student is in {p['state']}."}
            results.append(res)
            continue
        if sid not in state["evidence"]:
            continue
        ev = state["evidence"][sid]
        by_kind = {c["kind"]: c for c in ev}
        checks = python_checks(p, s)
        elig = by_kind.get("elig")
        if elig:                                                  # LLM reads the unstructured rule text
            for r in llm.interpret_rules(elig["text"], use_llm):
                v = p.get(r["field"])
                checks.append(_check("other_scholarship", "Other scholarship", v == r["required"],
                                     f"rule '{r['rule']}'; student answered {v}", r["rule"], unknown=v in UNKNOWN))
        status = ("Not a match" if any(c["result"] == "fail" for c in checks) else
                  "More information needed" if any(c["result"] == "unknown" for c in checks) else "Potential match")
        checklist = []
        if status != "Not a match" and "docs" in by_kind:
            raw = by_kind["docs"]["text"].split(":", 1)[1].strip().rstrip(".")
            checklist = [i.strip().capitalize() for i in raw.split(",")]
        claims = [{"type": "criterion", "field": c["field"], "result": c["result"], "key": c["key"],
                   "evidence_id": elig["id"] if elig else None} for c in checks]
        if status != "Not a match":
            claims.append({"type": "checklist", "items": checklist, "evidence_id": f"{sid}-docs"})
        claims.append({"type": "link", "url": s["url"]})
        results.append({"scheme_id": sid, "status": status, "checks": checks, "checklist": checklist,
                        "claims": claims, "explanation": llm.explain(s["name"], status, checks, False if status == "Not a match" else use_llm)})
    state["assessments"] = results
    state["tasks"][2]["status"] = "done"
    log(state, "Executor", "Python checks + " + ("Ollama" if use_llm else "mock") + " rule reading → " +
        ", ".join(f"{r['scheme_id']}: {r['status']}" for r in results))
    return state


# ------------------------------------------------------------------ 4. CRITIC
def _recheck(field, p, c):
    """Independent recomputation used by the critic (does not call the executor's code)."""
    level, tech = COURSES[p["course"]]
    if field == "income":
        return "unknown" if p["income"] is None else ("pass" if p["income"] <= c["income_max"] else "fail")
    if field == "marks":
        return "unknown" if p["marks"] is None else ("pass" if p["marks"] >= c["min_marks"] else "fail")
    if field == "category":
        return "pass" if p["category"] in c["categories"] else "fail"
    if field == "gender":
        return "pass" if p["gender"] == c["gender"] else "fail"
    if field == "minority":
        return "unknown" if p["minority"] in UNKNOWN else ("pass" if p["minority"] == "Yes" else "fail")
    if field == "level":
        return "pass" if level in c["levels"] else "fail"
    if field == "course":
        return "pass" if p["course"] in c["courses"] else "fail"
    if field == "technical":
        return "pass" if tech else "fail"
    if field == "other_scholarship":
        return "unknown" if p["other_scholarship"] in UNKNOWN else ("pass" if p["other_scholarship"] == "No" else "fail")
    return "fail"


def critic(state):
    p, verdicts, reassess = state["profile"], {}, []
    today = date.fromisoformat(state.get("today", date.today().isoformat()))
    for r in state["assessments"]:
        sid, s = r["scheme_id"], SCHEME_BY_ID[r["scheme_id"]]
        retrieved = {c["id"]: c for c in state["evidence"].get(sid, [])}
        issues, level = [], "verified"
        for cl in r["claims"]:
            if cl["type"] == "criterion":
                if cl["field"] == "state":
                    continue
                ev = retrieved.get(cl["evidence_id"])
                if not ev:
                    issues.append(f"No retrieved evidence for '{cl['field']}' claim"); level = "reassess"
                elif cl["field"] != "other_scholarship" and cl["key"] and cl["key"] not in ev["text"]:
                    issues.append(f"'{cl['key']}' not found in source text for '{cl['field']}'"); level = "reassess"
                elif cl["field"] == "other_scholarship" and cl["key"] not in ev["text"]:
                    issues.append("Rule text not found in source"); level = "reassess"
                if _recheck(cl["field"], p, s["criteria"]) != cl["result"]:
                    issues.append(f"Python re-check disagrees on '{cl['field']}'"); level = "reassess"
            elif cl["type"] == "checklist":
                ev = retrieved.get(cl["evidence_id"])
                if not ev or not cl["items"]:
                    issues.append("Document checklist has no supporting evidence"); level = "reassess"
                elif any(i.lower() not in ev["text"].lower() for i in cl["items"]):
                    issues.append("Checklist item not found in source"); level = "reassess"
            elif cl["type"] == "link" and cl["url"] != s["url"]:
                issues.append("Link differs from scheme record"); level = "reassess"
        # LLM-style check: split the explanation into atomic claims and judge each against the eligibility evidence
        elig_ev = next((c for c in retrieved.values() if c["kind"] == "elig"), None)
        if elig_ev and sid not in state["excluded"]:
            for stmt in llm.extract_claims(r["explanation"], False):
                if llm.judge(stmt, elig_ev["text"], False) == "unsupported":
                    issues.append(f"Explanation claim not supported by source: '{stmt[:80]}'")
                    if level == "verified":
                        level = "uncertain"
        if level == "verified":
            age = (today - date.fromisoformat(s["last_verified"])).days
            if age > 365:
                issues.append(f"Scheme record last verified {age} days ago — may be stale"); level = "uncertain"
            if r["status"] == "More information needed":
                missing = [c["label"] for c in r["checks"] if c["result"] == "unknown"]
                issues.append("Missing information: " + ", ".join(missing)); level = "uncertain"
        verdicts[sid] = {"verdict": level, "issues": issues}
        if level == "reassess":
            reassess.append(sid)
    state["verdicts"], state["tasks"][3]["status"] = verdicts, "done"
    if reassess and state["retry"] >= MAX_RETRY:               # retry limit → downgrade, never over-claim
        for r in state["assessments"]:
            if r["scheme_id"] in reassess:
                if r["status"] == "Potential match":
                    r["status"] = "More information needed"
                verdicts[r["scheme_id"]] = {"verdict": "uncertain", "issues": verdicts[r["scheme_id"]]["issues"] +
                                            ["Retry limit reached — result downgraded"]}
        reassess = []
    state["reassess_ids"] = reassess or None
    if reassess:
        state["retry"] += 1
    c = {k: sum(v["verdict"] == k for v in verdicts.values()) for k in ("verified", "uncertain", "reassess")}
    log(state, "Critic", f"Verified {c['verified']}, uncertain {c['uncertain']}, reassess {c['reassess']}"
        + (f" → loop back to Research (retry {state['retry']})" if reassess else ""))
    return state


# ------------------------------------------------------------------ finalize
def finalize(state):
    state["tasks"][4]["status"] = "done"
    order = {"Potential match": 0, "More information needed": 1, "Not a match": 2}
    state["assessments"].sort(key=lambda r: order[r["status"]])
    log(state, "Planner", "All tasks complete — report ready.")
    return state


def route_after_critic(state):
    return "research" if state["reassess_ids"] else "finalize"
