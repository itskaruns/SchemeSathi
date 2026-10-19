import json
from pathlib import Path
import streamlit as st
from schemesathi import graph, llm, chat
from schemesathi.agents import COURSES, SCHEME_BY_ID
from schemesathi.geo import STATES
from schemesathi.report import to_markdown, to_csv

st.set_page_config(page_title="SchemeSathi", page_icon="🎓", layout="wide")
DATA = Path(__file__).parent / "data"
CUSTOM = DATA / "custom_students.json"
students = json.loads((DATA / "students.json").read_text()) + (json.loads(CUSTOM.read_text()) if CUSTOM.exists() else [])


def inr(n):  # Indian digit grouping: 1000000 -> 10,00,000
    s_ = str(int(n)); head, tail = s_[:-3], s_[-3:]
    return (",".join([head[max(i - 2, 0):i] for i in range(len(head), 0, -2)][::-1]) + "," + tail) if head else tail
BADGE = {"Potential match": "🟢", "Not a match": "🔴", "More information needed": "🟠"}
VERDICT = {"verified": "✅ Verified", "uncertain": "⚠️ Uncertain (flagged)", "reassess": "🔁 Reassess"}

st.title("🎓 SchemeSathi")
st.caption("Multi-agent scholarship discovery, eligibility assessment and application preparation. "
           "Synthetic student data · preparation aid only, not an official decision.")
live = llm.ollama_available()
st.sidebar.markdown(f"**LLM mode:** {'Ollama (local)' if live else 'Mock LLM (rule-based fallback)'}")
use_llm = st.sidebar.checkbox("Use Ollama when available", value=live, disabled=not live)
st.sidebar.markdown("**Agents:** Planner → Research → Executor → Critic (loops back to Research if verification fails)")

# ---------------- A. Student profile
st.header("A. Student profile")
pick = st.selectbox("Load a demo student (built-in or saved by you)", ["(enter manually)"] + [s["name"] for s in students])
BLANK = {"name": "", "state": "Maharashtra", "course": "B.Tech / B.E.", "year": 1, "income": 500000, "marks": 75,
         "category": "General", "gender": "Female", "minority": "No", "other_scholarship": "No"}
ss = st.session_state
ver = ss.get("ver", 0)
kp = f"{pick}{ver}"
if ss.get("prefill_for") != pick:                       # switched student -> drop old text-fill overrides
    ss.update(prefill={}, prefill_msg=None, prefill_json=None, prefill_for=pick)
base = next((s for s in students if s["name"] == pick), BLANK)
d = {**base, **ss.get("prefill", {})}                   # text-fill values ALWAYS override the selected base profile


def current_form():
    """What is in the form right now, so a text-fill only changes the fields it actually found."""
    g = lambda k, dflt: ss.get(f"{k}{kp}", dflt)
    return {"name": g("n", d["name"]), "state": g("s", d["state"]), "course": g("c", d["course"]), "year": int(g("y", d["year"])),
            "income": g("i", d["income"] or 0) if g("ik", d["income"] is not None) else None,
            "marks": g("m", float(d["marks"] or 0)) if g("mk", d["marks"] is not None) else None,
            "category": g("ca", d["category"]), "gender": g("g", d["gender"]), "minority": g("mi", d["minority"]),
            "other_scholarship": g("o", d["other_scholarship"])}


with st.expander("💬 Or describe the student in plain language (the form is filled from your text)"):
    free = st.text_area("Example: My name is Riya, I'm a girl from Udaipur doing B.Tech 2nd year, family income 10 lakh "
                        "(or 1 million), 82%, general category, no other scholarship", key="free_text")
    if st.button("Fill profile from text") and free.strip():
        found, mode, notes = llm.extract_profile(free, use_llm)
        ss["prefill"] = {**current_form(), **found}
        ss["prefill_json"] = {"found": found, "ignored": notes}
        ss["prefill_msg"] = f"Filled by {mode} extractor: {', '.join(found) or 'nothing recognised'}"
        ss["ver"] = ver + 1
        st.rerun()
if ss.get("prefill_msg"):
    st.info(ss["prefill_msg"] + " — check the form below, then run the agents.")
    with st.expander("What the extractor understood (JSON)"):
        st.json(ss.get("prefill_json") or {})

c1, c2, c3 = st.columns(3)
with c1:
    name = st.text_input("Name", d["name"], key=f"n{kp}")
    state = st.selectbox("State / UT (a city is fine in the text box above)", STATES,
                         STATES.index(d["state"]) if d["state"] in STATES else STATES.index("Maharashtra"), key=f"s{kp}")
    course = st.selectbox("Course", list(COURSES), list(COURSES).index(d["course"]), key=f"c{kp}")
    year = st.number_input("Year of study", 1, 5, int(d["year"]), key=f"y{kp}")
with c2:
    inc_known = st.checkbox("Family income known", d["income"] is not None, key=f"ik{kp}")
    income = st.number_input("Annual family income (Rs)", 0, 1_000_000_000, int(d["income"] or 0), 10000, key=f"i{kp}",
                             disabled=not inc_known)
    if inc_known:
        st.caption(f"= ₹{inr(income)}")
    marks_known = st.checkbox("Marks known", d["marks"] is not None, key=f"mk{kp}")
    marks = st.number_input("Latest marks (%)", 0.0, 100.0, float(d["marks"] or 0), key=f"m{kp}",
                            disabled=not marks_known)
with c3:
    cats = ["General", "EWS", "OBC", "SC", "ST"]
    category = st.selectbox("Category", cats, cats.index(d["category"]), key=f"ca{kp}")
    gender = st.selectbox("Gender", ["Female", "Male", "Other"], ["Female", "Male", "Other"].index(d["gender"]), key=f"g{kp}")
    yn = ["Yes", "No", "Unknown"]
    minority = st.selectbox("Notified minority community?", yn, yn.index(d["minority"]), key=f"mi{kp}")
    other = st.selectbox("Receiving another scholarship?", yn, yn.index(d["other_scholarship"]), key=f"o{kp}")

profile = {"name": name or "Student", "state": state, "course": course, "year": int(year),
           "income": int(income) if inc_known else None, "marks": float(marks) if marks_known else None,
           "category": category, "gender": gender, "minority": minority, "other_scholarship": other}

b1, b2, _ = st.columns([1, 1, 3])
if b1.button("Run SchemeSathi agents", type="primary"):
    with st.spinner("Agents working…"):
        ss["out"] = graph.run(profile, use_llm=use_llm)
if b2.button("💾 Save profile to demo list"):
    if not name.strip():
        st.error("Enter a name first.")
    else:
        saved = [x for x in (json.loads(CUSTOM.read_text()) if CUSTOM.exists() else []) if x["name"] != name.strip()]
        CUSTOM.write_text(json.dumps(saved + [{**profile, "name": name.strip()}], indent=1))
        st.success(f"Saved '{name.strip()}' to data/custom_students.json. It now appears in the dropdown.")

out = st.session_state.get("out")
if out:
    st.divider()
    st.caption(f"Orchestration: {out['engine']} · LLM: {out['llm_mode']} · Retrieval: {out['retrieval_mode']} · Results are for: {out['profile']['name']}")
    if out["profile"] != profile:
        st.warning("The form has changed since the agents last ran. Results, downloads and the chat below still "
                   "reflect the OLD profile. Click **Run SchemeSathi agents** to refresh.")
    tabB, tabC, tabD = st.tabs(["B. Agent activity", "C. Scholarship results", "D. Application preparation"])

    with tabB:
        st.subheader("Task tracker (Planner)")
        st.write(" → ".join(f"{'✅' if t['status']=='done' else '❌'} {t['name']}" for t in out["tasks"]))
        st.subheader("Execution log")
        icon = {"Planner": "🧭", "Research": "🔎", "Executor": "⚙️", "Critic": "🕵️"}
        for l in out["log"]:
            st.markdown(f"{icon[l['agent']]} **{l['agent']}** — {l['msg']}")
        if out["retry"]:
            st.info(f"The Critic sent work back to Research {out['retry']} time(s) because the first-pass "
                    "evidence was too thin to support every claim.")

    with tabC:
        for r in out["assessments"]:
            s, v = SCHEME_BY_ID[r["scheme_id"]], out["verdicts"].get(r["scheme_id"], {})
            with st.expander(f"{BADGE[r['status']]} {s['name']} — {r['status']}", expanded=r["status"] != "Not a match"):
                st.write(r["explanation"])
                st.markdown(f"**Critic:** {VERDICT.get(v.get('verdict'), 'n/a')}")
                for i in v.get("issues", []):
                    st.warning(i)
                st.markdown("**Condition checks**")
                st.table([{"Condition": c["label"], "Result": {"pass": "✅ pass", "fail": "❌ fail", "unknown": "❓ unknown"}[c["result"]],
                           "Detail": c["detail"]} for c in r["checks"]])
                ev = out["evidence"].get(r["scheme_id"], [])
                if ev:
                    st.markdown("**Source evidence (retrieved passages)**")
                    for e in ev:
                        st.caption(f"`{e['id']}` (score {e['score']}): {e['text']}")

    with tabD:
        st.markdown("Only schemes that are not ruled out appear here.")
        for r in out["assessments"]:
            if r["status"] == "Not a match":
                continue
            s = SCHEME_BY_ID[r["scheme_id"]]
            st.subheader(f"{BADGE[r['status']]} {s['name']}")
            for i in r["checklist"]:
                st.checkbox(i, key=f"{r['scheme_id']}{i}")
            st.markdown(f"[Official portal]({s['url']}) · deadlines change yearly, confirm on the portal")
        st.download_button("Download report (Markdown)", to_markdown(out), "schemesathi_report.md")
        st.download_button("Download summary (CSV)", to_csv(out), "schemesathi_summary.csv")

# ---------------- E. Chat assistant
st.divider()
st.header("E. Ask SchemeSathi")
st.caption("Ask about schemes, documents, or why a result came out the way it did. "
           + ("Answers come from your local Ollama model, grounded in the agents' results and retrieved guideline text."
              if use_llm else "Ollama is off, so you only get the closest guideline passages (mock mode)."))
hist = st.session_state.setdefault("chat", [])
for m in hist:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
if q := st.chat_input("e.g. Why am I not eligible for the Central Sector scheme? What documents do I need?"):
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            ans, _mode = chat.ask(q, hist, st.session_state.get("out"), use_llm)
        st.markdown(ans)
    hist += [{"role": "user", "content": q}, {"role": "assistant", "content": ans}]
