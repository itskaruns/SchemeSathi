"""Builds the grounded context for the chat assistant from pipeline results + RAG retrieval."""
from . import llm
from .agents import STORE, SCHEME_BY_ID
from .retrieval import search_all


def build_context(question: str, out: dict | None) -> str:
    parts = []
    if out:
        p = out["profile"]
        parts.append(f"STUDENT: {p['name']}, {p['state']}, {p['course']} year {p['year']}, category {p['category']}, "
                     f"gender {p['gender']}, income {p['income']}, marks {p['marks']}, minority {p['minority']}, "
                     f"other scholarship {p['other_scholarship']}.")
        parts.append("AGENT RESULTS (already verified by the Critic):")
        for r in out["assessments"]:
            v = out["verdicts"].get(r["scheme_id"], {})
            
            # Filter out the noise so the LLM only sees exactly what failed
            fails = [c["detail"] for c in r["checks"] if c["result"] == "fail"]
            reasons = f" Failed checks: {'; '.join(fails)}." if fails else " All conditions met."
            
            parts.append(f"- {SCHEME_BY_ID[r['scheme_id']]['name']}: {r['status']} (critic: {v.get('verdict')})."
                         f"{reasons} Critic flags: {'; '.join(v.get('issues', [])) or 'none'}.")
    else:
        parts.append("No student profile has been assessed yet.")
    hits = search_all(STORE, question, 4)
    if hits:
        parts.append("RELEVANT GUIDELINE PASSAGES:")
        parts += [f"- [{h['id']}] {h['text']}" for h in hits]
    return "\n".join(parts)


def ask(question: str, history: list, out: dict | None, use_llm: bool):
    return llm.chat(question, history, build_context(question, out), use_llm)
