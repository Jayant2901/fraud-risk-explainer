"""
CLI for the audit trail: `python -m src.audit verify`.

Walks every entry in the audit log (data/audit_log.jsonl by default, or
Redis when REDIS_URL is set — see src/audit_log.py) and reports whether
the hash chain checks out end to end, or exactly where it doesn't.
"""
import os
import sys

sys.path.append(os.path.dirname(__file__))

from audit_log import create_audit_log, verify_chain
from redis_utils import get_redis_client


def verify() -> int:
    log = create_audit_log(get_redis_client())
    entries = log.entries()
    if not entries:
        print("Audit log is empty - nothing to verify.")
        return 0

    result = verify_chain(entries)
    if result["ok"]:
        print(f"OK - {result['entries_verified']} entries verified, chain intact.")
        return 0

    print(
        f"TAMPERED - chain breaks at entry {result['broken_at']} "
        f"(verdict_id={result['verdict_id']!r}): {result['reason']}"
    )
    return 1


def main(argv: list[str]) -> int:
    if len(argv) != 1 or argv[0] != "verify":
        print("Usage: python -m src.audit verify", file=sys.stderr)
        return 2
    return verify()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
