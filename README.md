# SchemeSathi – Multi-Agent Scholarship PoC

## Run
```bash
pip install -r requirements.txt
streamlit run app.py          # UI
python eval.py                # evaluation harness (196 labelled cases, critic fault injection, component tests)
```
Optional real LLM: install Ollama, `ollama pull llama3.2`. If Ollama isn't running, the app uses a
rule-based **Mock LLM** and says so in the sidebar. LangGraph is used if installed (fallback runner otherwise).

## Structure
| File | Role |
|---|---|
| `schemesathi/agents.py` | Planner, Research, Executor, Critic, route logic |
| `schemesathi/graph.py` | LangGraph state graph with conditional Critic → Research edge, retry cap |
| `schemesathi/retrieval.py` | RAG: `embed()` → `VectorStore` (cosine top-k) → keyword re-ranker. State pre-filter happens in Research |
| `ingest.py` | PyMuPDF: guideline PDF → classified chunks → `data/ingested_chunks.json` (`python ingest.py --demo`) |
| `schemesathi/llm.py` | Ollama / mock LLM: reads textual rules, writes explanations, `extract_claims()` and `judge()` for the Critic |
| `data/schemes.json` | 14 curated scheme records (central, state and private/CSR) + guideline chunks (verify against official portals before real use) |
| `data/students.json` | 10 synthetic students with hand-labelled expected results |

## LLM features (need Ollama to be truly conversational)
- **Chat assistant (section E):** answers questions grounded in the agents' verified results + retrieved guideline text (`chat.py`, `llm.chat`).
- **Plain-language profile entry:** LLM extracts profile fields from free text; output is validated before filling the form.
- **Rule reading + explanations** inside the Executor.
Without Ollama, extraction uses regexes and chat only returns the closest guideline passages (clearly labelled mock mode).

## Components and what is real vs mock
| Component | Real path | Mock / fallback path (used by default) |
|---|---|---|
| Embeddings + vector store | `SCHEMESATHI_EMBED=st` + `pip install sentence-transformers` → semantic embeddings | Hashed IDF-weighted bag-of-words vectors in an in-memory `VectorStore` (lexical, **not semantic**) |
| Re-ranking | n/a (keyword re-ranker is always the real code) | Vector score + query-term overlap + bonus for scheme key phrases (not a cross-encoder) |
| PDF ingestion | PyMuPDF reads PDFs with the headings Overview/Eligibility/Documents/Benefits/How to apply | Only ONE sample PDF (PMS-SC, generated from the record) is ingested; other 13 schemes use hand-written chunks |
| Claim extraction | Ollama splits the explanation into JSON claims | Sentence/clause split |
| Judge (entailment) | Ollama YES/NO on the scheme-side number | Scheme limit number must appear in the evidence text (string match, not real entailment) |

The UI and reports label the active retrieval and LLM mode.

## Agent flow
Planner validates profile and sets tasks → Research pre-filters by state and retrieves top-k passages
(k=2 first pass, 5 on retry) → Executor does exact checks in Python and has the LLM read textual rules, produces
status + checklist + atomic claims → Critic checks each claim against retrieved text, splits the explanation into atomic claims and judges them against the evidence, re-computes numbers
independently, flags stale/missing info → if evidence is missing it loops back to Research (max 2 retries,
then downgrades to "More information needed").

## What changed in this version
- Plain-language entry now understands names, cities (Udaipur -> Rajasthan, typos tolerated), and money in any form
  (`1 million`, `10 lakh`, `10,00,000`, `1.5 crore`). Rules parse numbers/places exactly; Ollama fills the rest.
  Invalid values are dropped and listed, never guessed. The extracted JSON is shown under the form.
- The form always updates from the text (previously ignored when a demo student was selected) and only overwrites
  the fields the text mentioned.
- State is a dropdown of all states/UTs; income limit raised to Rs 100 crore.
- A warning appears when the form differs from the last agent run (results/chat would be stale).
- "Save profile to demo list" writes to `data/custom_students.json`.
- 9 new schemes (PMS-ST, MCM-MIN, MH-OBC, WB-SVMCM, GJ-MYSY, KOTAK-KANYA, LIC-GJS, ONGC, HDFC-ECSS), new optional
  `courses` criterion, 4 new demo students. `python eval.py` now checks 196 labelled cases + extraction tests.
- New scheme records are approximations written from memory: VERIFY on the official portals before presenting.

## Demo script (5 min)
1. 30s: problem + architecture slide.
2. Load **Asha Patil** → run → Agent activity tab: show the Critic sending a few schemes back to Research
   (first pass retrieved only 2 passages, so no document evidence), then verified. Pragati is flagged stale.
3. Load **Sneha Joshi** (income missing) → "More information needed", nothing guessed.
4. Load **Vikram Singh** → all "Not a match" with reasons.
5. Show downloaded report; mention `eval.py` result (50/50, 4/4 injected errors caught).

## Honest limitations (say these in the viva)
- Scheme rules are simplified from public guidelines (e.g. CSSS percentile → 80% marks). Not official.
- Scheme set is small (14) and all records are simplified approximations; TF-IDF instead of embeddings; mock LLM is rule-based.
- Eval labels are written by the team, so 50/50 shows the logic is consistent, not real-world accuracy.
- Does not submit applications or guarantee any award.

## Future scope
Real embeddings + ChromaDB as the default, ingesting PDFs for all schemes, more states/schemes, deadline tracking, regional languages.
