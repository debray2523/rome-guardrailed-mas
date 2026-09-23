# ROME Guardrailed Multi-Agent System

A Planner → Executor → Reviewer multi-agent workflow with hard execution limits, strict typed outputs and persistent memory. It is built as a response to the **Alibaba ROME autonomous agent incident (March 2026)**, where an agent in an RL tool loop escaped network controls and hijacked GPUs for crypto-mining.

**Stack:** LangGraph (orchestration) · CrewAI (executor) · PydanticAI (schemas) · Mem0 + SQLite + Chroma (memory) · Groq (LLM)

Design rationale: **[DESIGN.md](DESIGN.md)**

---

## Rubric → where to find it

| Criterion | Marks | Implementation | Evidence |
|---|---|---|---|
| 3-node agent graph (Planner → Executor → Reviewer) with LangGraph and CrewAI | 25 | [`mas/graph.py`](mas/graph.py) `build_graph()`; Planner/Reviewer are PydanticAI agents, Executor is a CrewAI crew ([`mas/agents.py`](mas/agents.py)) | [`01_graph.txt`](docs/evidence/01_graph.txt), [`02_live_run_session1.txt`](docs/evidence/02_live_run_session1.txt) |
| `execution_count` in Graph State, with a conditional edge enforcing the cap | 20 | [`mas/state.py`](mas/state.py) `GraphState.execution_count`; [`mas/guardrails.py`](mas/guardrails.py) `loop_guard` + `route_after_guard` | `tests/test_guardrails.py::test_loop_cap_terminates_after_three_executions` |
| Kill-switch / safe fallback when `execution_count > 3` | 15 | `guardrails.fallback` returns a schema-valid `FinalResponse(status="terminated_loop_cap")`; kill-switch via `MAS_KILL_SWITCH=1` or a `KILL_SWITCH` file | `test_kill_switch_*`; live: [`05_loop_cap.txt`](docs/evidence/05_loop_cap.txt), [`06_kill_switch.txt`](docs/evidence/06_kill_switch.txt) |
| Structured output validated against PydanticAI schemas | 20 | [`mas/schemas.py`](mas/schemas.py); `Agent(output_type=PlannerOutput / ReviewerOutput)`; CrewAI `output_pydantic=ExecutorOutput` then strict re-validation | `tests/test_schemas.py`, `test_invalid_planner_output_is_rejected_not_passed_on` |
| Mem0 with local SQLite; persistence across sessions | 10 | [`mas/memory.py`](mas/memory.py): SQLite `mem0_history.db` + Chroma under `data/` (`data/lexical/` with the fallback embedder) | `tests/test_memory_persistence.py`; live: [`03_live_run_session2_memory.txt`](docs/evidence/03_live_run_session2_memory.txt) |
| Design write-up (why N ≤ 3, why this schema design) | 10 | [DESIGN.md](DESIGN.md) | — |

## Architecture

```mermaid
graph TD
  START([start]) --> load_memory
  load_memory --> planner[Planner · PydanticAI]
  planner -->|ok| loop_guard{{loop_guard<br/>execution_count += 1}}
  planner -.->|error| fallback
  loop_guard -->|count ≤ 3| executor[Executor · CrewAI]
  loop_guard -->|count > 3 or kill-switch| fallback[[fallback · safe FinalResponse]]
  executor --> reviewer[Reviewer · PydanticAI]
  executor -.->|error / kill| fallback
  reviewer -->|approve| finalize
  reviewer -->|reject| loop_guard
  finalize --> save_memory
  fallback --> save_memory
  save_memory --> END([end])
```

Only the three agent nodes call an LLM. `loop_guard`, `fallback`, `finalize` and the memory nodes are deterministic Python, so no model output can steer the run past the cap.

## Setup

```bash
git clone https://github.com/debray2523/rome-guardrailed-mas.git
cd rome-guardrailed-mas
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env    # then add your GROQ_API_KEY (free at console.groq.com)
```

Colab: open [`notebooks/ROME_Guardrailed_MAS.ipynb`](notebooks/ROME_Guardrailed_MAS.ipynb) and add `GROQ_API_KEY` as a Colab secret.

## Run

```bash
# Live (Groq)
python -m mas.cli run --user deb "Outline a CRM database migration. I prefer answers in bullet points."
python -m mas.cli run --user deb "What answer format do I prefer?"      # new process = new session
python -m mas.cli memories --user deb

# Loop-cap demo: real Planner/Executor, Reviewer scripted to always reject
python -m mas.cli run --user deb --force-reject "Summarise my preferences"

# Kill-switch
MAS_KILL_SWITCH=1 python -m mas.cli run "anything"         # or: create a file named KILL_SWITCH

# No API key? Add --offline: scripted models, same graph, guardrails, schemas and Mem0 storage
python -m mas.cli run --offline --user deb "I prefer answers in bullet points"

# Tests (no key, no network)
python -m pytest -q          # 22 passed
```

## Demonstration output (live Groq runs)

Captured on 23 Sep 2026 on Windows with Groq `openai/gpt-oss-120b`. The full, unedited outputs are in [`docs/evidence/`](docs/evidence/); only library warnings are omitted below.

| # | What it proves | File |
|---|---|---|
| 0 | 22 tests pass | [`00_tests.txt`](docs/evidence/00_tests.txt) |
| 1 | Graph structure (8 nodes, conditional edges) | [`01_graph.txt`](docs/evidence/01_graph.txt) |
| 2 | Planner → Executor → Reviewer completes | [`02_live_run_session1.txt`](docs/evidence/02_live_run_session1.txt) |
| 3 | A new process recalls session 1's preference | [`03_live_run_session2_memory.txt`](docs/evidence/03_live_run_session2_memory.txt) |
| 4 | Memories persisted on disk (SQLite + Chroma) | [`04_stored_memories.txt`](docs/evidence/04_stored_memories.txt) |
| 5 | Loop cap: 3 executions, then fallback | [`05_loop_cap.txt`](docs/evidence/05_loop_cap.txt) |
| 6 | Kill-switch halts before any LLM call | [`06_kill_switch.txt`](docs/evidence/06_kill_switch.txt) |

### Session 1: full Planner → Executor → Reviewer run

`python -m mas.cli run --user deb "Outline a 3-step plan to migrate a CRM database. I prefer answers in bullet points."`

```
[guardrails] MAX_EXECUTIONS=3 recursion_limit=20 kill_switch_file=KILL_SWITCH
--- trace ---
  load_memory: 0 facts
  planner: 3 steps
  loop_guard: execution_count=1
  executor: attempt 1
  reviewer: approve
  finalize: completed
  save_memory: stored 1 message(s)
--- recalled memories ---
  - (none)
--- FinalResponse (validated) ---
{
  "status": "completed",
  "answer": "- Perform a full backup of the current CRM database and verify the backup integrity.\n- Set up the target environment (new server/DBMS), configure schema, and run a test migration using a copy of the backup to validate data mapping and performance.\n- Schedule a maintenance window, pause writes to the source database, execute the final data transfer, run post\u2011migration validation checks, and switch the CRM application to the new database.",
  "execution_count": 1,
  "reason": "approved by reviewer"
}
```

### Session 2: a new process recalls the stored preference

`python -m mas.cli run --user deb "What answer format do I prefer?"`

```
[guardrails] MAX_EXECUTIONS=3 recursion_limit=20 kill_switch_file=KILL_SWITCH
--- trace ---
  load_memory: 1 facts
  planner: 3 steps
  loop_guard: execution_count=1
  executor: attempt 1
  reviewer: approve
  finalize: completed
  save_memory: stored 1 message(s)
--- recalled memories ---
  - Outline a 3-step plan to migrate a CRM database. I prefer answers in bullet points.
--- FinalResponse (validated) ---
{
  "status": "completed",
  "answer": "- You prefer answers in bullet points.\n- I will use bullet\u2011point format for future responses.",
  "execution_count": 1,
  "reason": "approved by reviewer"
}
```

`python -m mas.cli memories --user deb`

```
[guardrails] MAX_EXECUTIONS=3 recursion_limit=20 kill_switch_file=KILL_SWITCH
2 stored memories for user 'deb':
  - Outline a 3-step plan to migrate a CRM database. I prefer answers in bullet points.
  - What answer format do I prefer?
```

### Loop cap: reviewer forced to reject, real Planner and Executor

`python -m mas.cli run --user deb --force-reject "Summarise my preferences"`

The Executor runs exactly 3 times. The 4th request makes `execution_count` 4, the conditional edge routes to `fallback`, and a schema-valid `FinalResponse` is returned.

```
[guardrails] MAX_EXECUTIONS=3 recursion_limit=20 kill_switch_file=KILL_SWITCH
--- trace ---
  load_memory: 2 facts
  planner: 3 steps
  loop_guard: execution_count=1
  executor: attempt 1
  reviewer: reject
  loop_guard: execution_count=2
  executor: attempt 2
  reviewer: reject
  loop_guard: execution_count=3
  executor: attempt 3
  reviewer: reject
  loop_guard: execution_count=4
  fallback: status=terminated_loop_cap
  save_memory: stored 1 message(s)
--- recalled memories ---
  - What answer format do I prefer?
  - Outline a 3-step plan to migrate a CRM database. I prefer answers in bullet points.
--- FinalResponse (validated) ---
{
  "status": "terminated_loop_cap",
  "answer": "I could not produce a reviewer-approved answer within the safety budget of 3 execution attempts, so execution was stopped. No further actions were taken. Please refine the request or escalate to a human.",
  "execution_count": 4,
  "reason": "execution_count=4 exceeded MAX_EXECUTIONS=3"
}
```

### Kill-switch

`set MAS_KILL_SWITCH=1` then `python -m mas.cli run --no-memory "anything"`

```
[guardrails] MAX_EXECUTIONS=3 recursion_limit=20 kill_switch_file=KILL_SWITCH
--- trace ---
  load_memory: disabled
  planner: kill-switch
  fallback: status=killed
  save_memory: disabled
--- recalled memories ---
  - (none)
--- FinalResponse (validated) ---
{
  "status": "killed",
  "answer": "Execution was halted by the system kill-switch. No further actions were taken.",
  "execution_count": 0,
  "reason": "operator kill-switch engaged"
}
```

Offline versions of all of these (scripted models, no API key) can be reproduced with `--offline`.

## Project layout

```
mas/
  config.py       all hard limits in one place
  schemas.py      PlannerOutput, ExecutorOutput, ReviewerOutput, FinalResponse
  state.py        GraphState (execution_count, halt_reason, trace)
  guardrails.py   loop_guard, conditional edges, kill-switch, fallback
  agents.py       PydanticAI planner/reviewer, CrewAI executor
  memory.py       Mem0 config (SQLite + Chroma + local embeddings)
  graph.py        LangGraph wiring
  offline.py      scripted models for tests and --offline
  cli.py          command line
notebooks/ROME_Guardrailed_MAS.ipynb
tests/            22 deterministic tests
DESIGN.md
```

## Notes

- Memory uses Mem0 OSS. **SQLite** holds the memory history/audit log (`data/mem0_history.db`) and **Chroma** holds the vectors (`data/chroma`). Both are local files. Embeddings are local Hugging Face `all-MiniLM-L6-v2`, or the lexical fallback below (used for the live evidence, because Windows Application Control blocked the scipy DLLs).
- **Embedder fallback.** If the Hugging Face stack can't load (for example a Windows Application Control policy blocks the scipy/torch DLLs), the live store switches automatically to a pure-Python lexical embedder and says so on stderr. Force it with `MAS_EMBEDDER=hashing` in `.env`. SQLite + Chroma persistence is unchanged; recall matches on shared words rather than meaning. Its data lives in `data/lexical/`.
- **Groq free tier (8,000 tokens/minute).** Planner and Reviewer use Groq's native JSON-schema output with `reasoning_effort=low` and `max_tokens=1500`. HTTP 429 gets a bounded retry (2 retries, 20 s wait), never an open loop. Mem0's LLM fact extraction sends an ~8,200-token prompt, over the free-tier limit, so by default memory stores the user's own statements verbatim (`infer=False`); set `MAS_MEM_INFER=1` on a paid tier to have Groq condense facts.
- Mem0 2.x API: `search(query, filters={"user_id": ...}, top_k=...)`.
- Groq model IDs change: `llama-3.3-70b-versatile` was retired on 16 Aug 2026. The default is now `openai/gpt-oss-120b`; set `GROQ_MODEL` in `.env` to any Groq model with JSON-schema output.
