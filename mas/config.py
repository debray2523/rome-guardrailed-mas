"""Central configuration. Every limit that bounds agent behaviour lives here,
so an auditor can read the system's hard ceilings in one place."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")
os.environ.setdefault("MEM0_TELEMETRY", "False")

# --- Execution guardrails -------------------------------------------------
# Hard cap on Executor attempts per task (N <= 3). A 4th attempt request pushes
# execution_count to 4 (> MAX_EXECUTIONS) and the graph routes to the fallback.
MAX_EXECUTIONS: int = 3

# LangGraph super-step backstop. A full worst-case run needs 14 steps
# (load_memory, planner, 3 x [loop_guard, executor, reviewer], loop_guard,
# fallback, save_memory). 20 leaves headroom but still halts any routing bug.
RECURSION_LIMIT: int = 20

# PydanticAI inner validation-retry loop. Kept at 1 so schema repair cannot
# become a hidden, unbounded loop inside a node.
PYDANTIC_AI_RETRIES: int = 1

# Per-agent-call budget (PydanticAI UsageLimits).
REQUEST_LIMIT_PER_CALL: int = 3
TOKEN_LIMIT_PER_CALL: int = 6_000

# LLM output budget. Groq's free tier allows 8,000 tokens/minute per model, so every
# call is kept small: capped output, low reasoning effort for gpt-oss models.
LLM_MAX_OUTPUT_TOKENS: int = 1_500
LLM_REASONING_EFFORT: str = os.getenv("MAS_REASONING_EFFORT", "low")

# Rate-limit handling (HTTP 429): a *bounded* retry, never an open loop.
RATE_LIMIT_RETRIES: int = 2
RATE_LIMIT_WAIT_S: float = float(os.getenv("MAS_RATE_LIMIT_WAIT_S", "20"))

# CrewAI executor ceilings (CrewAI's own default max_iter is 20).
CREW_MAX_ITER: int = 3
CREW_MAX_EXECUTION_TIME_S: int = 60

# External kill-switch: either env var or the presence of this file.
KILL_SWITCH_ENV: str = "MAS_KILL_SWITCH"
KILL_SWITCH_FILE: Path = Path(os.getenv("MAS_KILL_FILE", "KILL_SWITCH"))

# --- Models ---------------------------------------------------------------
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")  # llama-3.3-70b-versatile retired by Groq 2026-08-16

# --- Memory ---------------------------------------------------------------
DATA_DIR: Path = Path(os.getenv("MAS_DATA_DIR", "data"))
MEM0_HISTORY_DB: Path = DATA_DIR / "mem0_history.db"   # SQLite
CHROMA_PATH: Path = DATA_DIR / "chroma"                 # vector store on disk
EMBED_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIMS: int = 384
MEMORY_TOP_K: int = 3
# Mem0 LLM fact extraction (infer=True) sends an ~8,200-token prompt, which exceeds the
# Groq free-tier 8,000 TPM limit. Default: store the user's own statements verbatim
# (infer=False, no LLM call). Set MAS_MEM_INFER=1 on a paid tier to condense facts.
MEM_INFER: bool = os.getenv("MAS_MEM_INFER", "0").strip().lower() in {"1", "true", "yes", "on"}  # only top-k facts are injected, never raw transcripts
