"""Orchestration: LangGraph state graph if installed, otherwise an equivalent minimal runner."""
from datetime import date
from typing import Any, TypedDict
from . import agents, llm


class State(TypedDict, total=False):
    profile: dict; use_llm: bool; today: str; tasks: list; log: list; retry: int; reassess_ids: Any
    evidence: dict; excluded: dict; assessments: list; verdicts: dict


def _langgraph():
    from langgraph.graph import StateGraph, END
    g = StateGraph(State)
    for name, fn in [("planner", agents.planner), ("research", agents.research), ("executor", agents.executor),
                     ("critic", agents.critic), ("finalize", agents.finalize)]:
        g.add_node(name, fn)
    g.set_entry_point("planner")
    g.add_edge("planner", "research"); g.add_edge("research", "executor"); g.add_edge("executor", "critic")
    g.add_conditional_edges("critic", agents.route_after_critic, {"research": "research", "finalize": "finalize"})
    g.add_edge("finalize", END)
    return g.compile()


def _fallback(state):
    state = agents.planner(state)
    while True:
        state = agents.critic(agents.executor(agents.research(state)))
        if agents.route_after_critic(state) == "finalize":
            return agents.finalize(state)


def run(profile: dict, use_llm=None, today=None):
    use_llm = llm.ollama_available() if use_llm is None else use_llm
    state = {"profile": profile, "use_llm": use_llm, "today": today or date.today().isoformat(),
             "log": [], "excluded": {}}
    try:
        out = _langgraph().invoke(state, {"recursion_limit": 50}); out["engine"] = "LangGraph"
    except ImportError:
        out = _fallback(state); out["engine"] = "built-in runner (LangGraph not installed)"
    out["retrieval_mode"] = agents.RETRIEVAL_MODE
    out["llm_mode"] = "Ollama" if use_llm else "Mock LLM (rule-based)"
    return out
