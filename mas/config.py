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

# CrewAI executor ceilings (CrewAI's own default max_iter is 20).
CREW_MAX_ITER: int = 3
CREW_MAX_EXECUTION_TIME_S: int = 60

# External kill-switch: either env var or the presence of this file.
KILL_SWITCH_ENV: str = "MAS_KILL_SWITCH"
KILL_SWITCH_FILE: Path = Path(os.getenv("MAS_KILL_FILE", "KILL_SWITCH"))

# --- Models ---------------------------------------------------------------
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# --- Memory ---------------------------------------------------------------
DATA_DIR: Path = Path(os.getenv("MAS_DATA_DIR", "data"))
MEM0_HISTORY_DB: Path = DATA_DIR / "mem0_history.db"   # SQLite
CHROMA_PATH: Path = DATA_DIR / "chroma"                 # vector store on disk
EMBED_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIMS: int = 384
MEMORY_TOP_K: int = 3  # only top-k facts are injected, never raw transcripts
