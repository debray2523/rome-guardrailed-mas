"""Command-line entry point.

    python -m mas.cli run --user deb "Plan a data migration; I prefer bullet points"
    python -m mas.cli run --user deb "Summarise my preferences" --force-reject
    python -m mas.cli memories --user deb
    python -m mas.cli graph

Add --offline to any command to use scripted models (no API key needed).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

from . import config


def _deps(offline: bool, force_reject: bool, use_memory: bool):
    from .graph import MASDeps
    from .memory import MemoryStore

    if offline:
        from .offline import StubExecutor, stub_planner, stub_reviewer

        planner, executor = stub_planner(), StubExecutor()
        reviewer = stub_reviewer("reject" if force_reject else "approve")
        memory = MemoryStore.offline() if use_memory else None
    else:
        if not os.getenv("GROQ_API_KEY"):
            sys.exit("GROQ_API_KEY is not set. Put it in .env or use --offline.")
        from .agents import CrewAIExecutor, build_planner, build_reviewer
        from .offline import stub_reviewer

        planner, executor = build_planner(), CrewAIExecutor()
        # --force-reject keeps the real Planner/Executor but scripts the Reviewer
        # to always reject, to demonstrate the loop cap against live models.
        reviewer = stub_reviewer("reject") if force_reject else build_reviewer()
        memory = MemoryStore.local() if use_memory else None
    return MASDeps(planner=planner, reviewer=reviewer, executor=executor, memory=memory)


def cmd_run(args) -> int:
    from .graph import build_graph, run_task

    app = build_graph(_deps(args.offline, args.force_reject, not args.no_memory))
    state = asyncio.run(run_task(app, args.user, args.task))
    print("--- trace ---")
    for line in state["trace"]:
        print(" ", line)
    print("--- recalled memories ---")
    for m in state.get("memories", []) or ["(none)"]:
        print("  -", m)
    print("--- FinalResponse (validated) ---")
    print(json.dumps(state["final"].model_dump(), indent=2))
    return 0


def cmd_memories(args) -> int:
    from .memory import MemoryStore

    store = MemoryStore.offline() if args.offline else MemoryStore.local()
    items = store.all(args.user)
    print(f"{len(items)} stored memories for user '{args.user}':")
    for m in items:
        print("  -", m)
    return 0


def cmd_graph(args) -> int:
    from .graph import MASDeps, build_graph
    from .offline import StubExecutor, stub_planner, stub_reviewer

    app = build_graph(MASDeps(stub_planner(), stub_reviewer(), StubExecutor()))
    print(app.get_graph().draw_mermaid())
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="mas", description="ROME guardrailed multi-agent system")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run a task through Planner -> Executor -> Reviewer")
    r.add_argument("task")
    r.add_argument("--user", default="demo-user")
    r.add_argument("--offline", action="store_true")
    r.add_argument("--force-reject", action="store_true", help="reviewer always rejects (loop-cap demo)")
    r.add_argument("--no-memory", action="store_true")
    r.set_defaults(func=cmd_run)

    m = sub.add_parser("memories", help="list persisted memories for a user")
    m.add_argument("--user", default="demo-user")
    m.add_argument("--offline", action="store_true")
    m.set_defaults(func=cmd_memories)

    g = sub.add_parser("graph", help="print the graph as Mermaid")
    g.set_defaults(func=cmd_graph)

    args = p.parse_args(argv)
    print(f"[guardrails] MAX_EXECUTIONS={config.MAX_EXECUTIONS} "
          f"recursion_limit={config.RECURSION_LIMIT} kill_switch_file={config.KILL_SWITCH_FILE}")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
