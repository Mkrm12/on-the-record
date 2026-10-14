"""Pre-register a measurement before it runs; verify the chain (`reports/preregistration.jsonl`)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eval import ledger as L  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("register")
    r.add_argument("--kind", required=True, choices=sorted(L.KINDS))
    r.add_argument("--label", required=True)
    r.add_argument("--plan", required=True)
    r.add_argument("--bars", required=True, help="JSON object of the bars the run will be judged against")
    sub.add_parser("verify")
    sub.add_parser("show")
    args = p.parse_args(argv)
    if args.command == "register":
        entry = L.register(L.PATH, kind=args.kind, label=args.label, plan=args.plan, bars=json.loads(args.bars),
                           **L.current())  # fmt: skip
        print(f"registered #{entry['seq']} {entry['hash'][:12]} ({entry['kind']}: {entry['label']})")
        return 0
    problems = L.verify(L.PATH)
    if args.command == "verify":
        print("the chain verifies" if not problems else "BROKEN:\n  " + "\n  ".join(problems))
        return 1 if problems else 0
    for entry in L._entries(L.PATH):
        print(f"#{entry['seq']} {entry['hash'][:12]} {entry['created']} {entry['kind']}: {entry['label']} "
              f"(gold {entry['gold'].get('content_hash')}, rubric {entry['rubric']}, code {entry['code']})")
    if problems:
        print("BROKEN:\n  " + "\n  ".join(problems))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
