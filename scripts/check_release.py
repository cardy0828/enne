"""公開前の検査。ソースと README・docs・Rust クレートの版、除外対象拡張子、実験機情報や機密情報の残りを確かめる。

    python -X utf8 scripts/check_release.py

ソースを更新したら、pyproject.toml の version、enne-rs/Cargo.toml の version、
README の「現在の到達点（vX.Y）」、docs/source.md の「vX.Y（…）時点」を同じ版にそろえてから実行する。
"""
from __future__ import annotations

import ast
import re
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GENERIC = [
    r"/home/[a-z]",
    r"[A-Za-z]:[\\/]Users[\\/]",
    r"GPU-[0-9a-f]{8}-[0-9a-f]{4}",
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----",
    r"\b(?:ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9]{20,})\b",
    r"[A-Za-z0-9._%+-]+@(?!users\.noreply\.github\.com|anthropic\.com)[A-Za-z0-9-]+\.(?:com|net|org|io|jp|co\.jp|ac\.jp)\b",
]
LOCAL = ROOT / ".leak-patterns"  # 追跡しないファイル。ホスト名など、公開したくない固有の文字列を1行ずつ書く
FORBIDDEN_SUFFIXES = {".bin", ".wav", ".pcm", ".pt", ".pkl", ".eng", ".sqlite", ".so", ".pyc"}
BINARY_TEXT_SKIP = {".png", ".pdf"}


def iter_candidate_files() -> list[Path]:
    seen: set[Path] = set()
    files: list[Path] = []
    for args in (
        ["git", "-c", "core.quotepath=false", "ls-files"],
        ["git", "-c", "core.quotepath=false", "ls-files", "--others", "--exclude-standard"],
    ):
        out = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=True).stdout
        for rel in out.splitlines():
            p = ROOT / rel
            if p.is_file() and p not in seen:
                seen.add(p)
                files.append(p)
    return sorted(files)


def main() -> int:
    errors = []
    version = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    short = "v" + ".".join(version.split(".")[:2])

    cargo_path = ROOT / "enne-rs" / "Cargo.toml"
    if not cargo_path.is_file():
        errors.append("enne-rs/Cargo.toml が存在しない")
    else:
        cargo_ver = tomllib.loads(cargo_path.read_text(encoding="utf-8"))["package"]["version"]
        if cargo_ver != version:
            errors.append(f"enne-rs/Cargo.toml の version ({cargo_ver}) が pyproject.toml ({version}) と一致しない")

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    if f"## 現在の到達点（{short}）" not in readme:
        errors.append(f"README の「現在の到達点」が {short} になっていない")
    if f"v0.1から{short}まで" not in readme:
        errors.append(f"README 冒頭の開発範囲が {short} になっていない")
    if f"{short} 時点" not in readme:
        errors.append(f"README のソースコードの節が {short} 時点になっていない")
    if f"{short}（" not in (ROOT / "docs" / "source.md").read_text(encoding="utf-8"):
        errors.append(f"docs/source.md が {short} 時点になっていない")
    if f"| {short} |" not in (ROOT / "docs" / "history.md").read_text(encoding="utf-8"):
        errors.append(f"docs/history.md に {short} の行がない")

    for path in sorted((ROOT / "src" / "neurograph").glob("*.py")):
        if ast.get_docstring(ast.parse(path.read_text(encoding="utf-8"))) is None:
            errors.append(f"{path.name} に冒頭の説明がない")

    corpus_dir = ROOT / "scripts" / "corpus"
    if corpus_dir.is_dir():
        for path in sorted(corpus_dir.glob("*.py")):
            if ast.get_docstring(ast.parse(path.read_text(encoding="utf-8"))) is None:
                errors.append(f"scripts/corpus/{path.name} に冒頭の説明がない")

    enne_rs_src = ROOT / "enne-rs" / "src"
    if enne_rs_src.is_dir():
        for path in sorted(enne_rs_src.glob("*.rs")):
            text = path.read_text(encoding="utf-8").lstrip()
            if not text.startswith("//!"):
                errors.append(f"enne-rs/src/{path.name} に冒頭の //! 説明がない")

    patterns = GENERIC + (
        [re.escape(l.strip()) for l in LOCAL.read_text(encoding="utf-8").splitlines() if l.strip()]
        if LOCAL.is_file()
        else []
    )
    leaks = re.compile("|".join(patterns))
    for path in iter_candidate_files():
        rel = path.relative_to(ROOT).as_posix()
        if path.name == ".leak-patterns":
            errors.append(".leak-patterns が公開対象に含まれている")
            continue
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"{rel} は公開対象外のファイル形式 ({path.suffix})")
            continue
        if path.suffix.lower() in BINARY_TEXT_SKIP:
            continue
        if leaks.search(path.read_text(encoding="utf-8", errors="ignore")):
            errors.append(f"{rel} に実験機または機密情報が残っている")

    for e in errors:
        print("NG", e)
    print("OK" if not errors else f"{len(errors)} 件の問題", f"（{short}）")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
