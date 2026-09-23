"""Cross-session persistence: two separate Python processes share one
Mem0 store (SQLite history + Chroma vectors on disk)."""
import sqlite3
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(code: str) -> str:
    out = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=ROOT, capture_output=True, text=True, timeout=300,
    )
    assert out.returncode == 0, out.stderr
    return out.stdout


def test_memory_persists_across_processes(tmp_path):
    data = tmp_path.as_posix()

    # Session 1: store a preference, then the process exits.
    _run(f"""
        from mas.memory import MemoryStore
        s = MemoryStore.offline("{data}")
        s.remember_sync("deb", [{{"role": "user", "content": "I prefer answers in bullet points"}}])
    """)

    # SQLite history file exists and recorded the ADD event.
    db = tmp_path / "mem0_history.db"
    assert db.exists()
    with sqlite3.connect(db) as con:
        assert con.execute("select count(*) from history").fetchone()[0] >= 1

    # Session 2: a brand-new process recalls it by semantic search.
    out = _run(f"""
        from mas.memory import MemoryStore
        s = MemoryStore.offline("{data}")
        print(s.recall_sync("deb", "what answer format do I prefer in bullet points", k=3))
        print(s.recall_sync("someone-else", "bullet points", k=3))
    """)
    mine, others = out.strip().splitlines()
    assert "bullet points" in mine
    assert others == "[]"  # memories are isolated per user_id
