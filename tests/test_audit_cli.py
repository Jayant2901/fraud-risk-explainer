"""
CLI tests for `python -m src.audit verify` — exercised as the module's
verify()/main() functions directly (fast, in-process) rather than
spawning a real subprocess; audit_log.py's own tests already cover the
hash-chain logic verify() calls into.
"""
import json

import audit
from audit_log import AuditLog


class TestVerify:
    def test_an_empty_log_reports_ok(self, tmp_path, monkeypatch, capsys):
        monkeypatch.delenv("REDIS_URL", raising=False)
        monkeypatch.setattr(audit, "create_audit_log", lambda _: AuditLog(None, str(tmp_path / "a.jsonl")))

        assert audit.verify() == 0
        assert "empty" in capsys.readouterr().out.lower()

    def test_an_intact_log_reports_ok(self, tmp_path, monkeypatch, capsys):
        log_path = str(tmp_path / "a.jsonl")
        log = AuditLog(None, log_path)
        log.append({"verdict_id": "v1", "risk_score": 50.0})
        log.append({"verdict_id": "v2", "risk_score": 90.0})
        monkeypatch.setattr(audit, "create_audit_log", lambda _: AuditLog(None, log_path))

        assert audit.verify() == 0
        assert "OK" in capsys.readouterr().out

    def test_a_tampered_log_reports_failure_and_the_break_point(self, tmp_path, monkeypatch, capsys):
        log_path = tmp_path / "a.jsonl"
        log = AuditLog(None, str(log_path))
        log.append({"verdict_id": "v1", "risk_score": 50.0})
        log.append({"verdict_id": "v2", "risk_score": 90.0})
        lines = log_path.read_text(encoding="utf-8").splitlines()
        first_entry = json.loads(lines[0])
        first_entry["risk_score"] = 5.0  # tamper, without recomputing the hash
        lines[0] = json.dumps(first_entry)
        log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        monkeypatch.setattr(audit, "create_audit_log", lambda _: AuditLog(None, str(log_path)))

        assert audit.verify() == 1
        out = capsys.readouterr().out
        assert "TAMPERED" in out
        assert "v1" in out


class TestMain:
    def test_verify_argument_runs_verification(self, tmp_path, monkeypatch):
        monkeypatch.setattr(audit, "create_audit_log", lambda _: AuditLog(None, str(tmp_path / "a.jsonl")))

        assert audit.main(["verify"]) == 0

    def test_missing_argument_is_a_usage_error(self):
        assert audit.main([]) == 2

    def test_unknown_argument_is_a_usage_error(self):
        assert audit.main(["bogus"]) == 2
