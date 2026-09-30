"""監査 G-11: API キーらしき文字列をソースに置かない。SSL 検証を全体で無効化しない。"""
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".py", ".yaml", ".yml", ".toml", ".cfg", ".ini", ".env", ".txt", ".md"}
KEY_ASSIGN = re.compile(r"""(?i)(api[_-]?key|secret|token)\s*[:=]\s*["'][0-9A-Za-z_\-]{16,}["']""")
HEX32 = re.compile(r"\b[0-9a-f]{32}\b")


def _tracked_text_files():
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    for rel in out.splitlines():
        p = ROOT / rel
        if p.suffix in TEXT_SUFFIXES and p.is_file():
            yield p


def test_no_secrets_in_repo():
    hits = []
    for p in _tracked_text_files():
        text = p.read_text(encoding="utf-8", errors="ignore")
        for i, line in enumerate(text.splitlines(), 1):
            if KEY_ASSIGN.search(line) or (p.suffix == ".py" and HEX32.search(line)):
                hits.append(f"{p.relative_to(ROOT)}:{i}")
    assert not hits, f"API キーらしき文字列: {hits}"


def test_ssl_verification_not_disabled():
    hits = [str(p.relative_to(ROOT)) for p in _tracked_text_files()
            if p.suffix == ".py" and "_create_unverified_context" in p.read_text(encoding="utf-8", errors="ignore")
            and p.name != Path(__file__).name]
    assert not hits, f"SSL 検証の無効化: {hits}"
