from .agents import SCHEME_BY_ID


def to_markdown(out):
    p = out["profile"]
    L = [f"# SchemeSathi report for {p['name']}", "",
         f"State: {p['state']} | Course: {p['course']} (year {p['year']}) | Category: {p['category']}",
         f"Mode: {out['llm_mode']} | Orchestration: {out['engine']}", "",
         "> Preparation aid only. Not an official decision. Confirm every detail on the official portal.", ""]
    for r in out["assessments"]:
        s, v = SCHEME_BY_ID[r["scheme_id"]], out["verdicts"].get(r["scheme_id"], {})
        L += [f"## {s['name']} — {r['status']}", f"Critic verdict: {v.get('verdict', 'n/a')}", r["explanation"]]
        for i in v.get("issues", []):
            L.append(f"- To confirm: {i}")
        if r["checklist"]:
            L += ["", "Document checklist:"] + [f"- [ ] {i}" for i in r["checklist"]]
        if r["status"] != "Not a match":
            L.append(f"\nApply: {s['url']}")
        L.append("")
    return "\n".join(L)


def to_csv(out):
    import csv, io
    b = io.StringIO(); w = csv.writer(b)
    w.writerow(["scheme", "status", "critic_verdict", "issues", "documents"])
    for r in out["assessments"]:
        v = out["verdicts"].get(r["scheme_id"], {})
        w.writerow([SCHEME_BY_ID[r["scheme_id"]]["name"], r["status"], v.get("verdict"),
                    "; ".join(v.get("issues", [])), "; ".join(r["checklist"])])
    return b.getvalue()
