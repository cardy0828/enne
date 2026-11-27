"""公開前の検査。ソースと README・docs の版がそろっているか、実験機の情報が残っていないかを確かめる。

    python -X utf8 scripts/check_release.py

ソースを更新したら、pyproject.toml の version、README の「現在の到達点（vX.Y）」、
docs/source.md の「vX.Y（…）時点」を同じ版にそろえてから実行する。
"""
from __future__ import annotations

import ast
import re
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GENERIC = [r"/home/[a-z]", r"GPU-[0-9a-f]{8}-[0-9a-f]{4}", r"[A-Za-z0-9._%+-]+@(?!users\.noreply\.github\.com|anthropic\.com)[A-Za-z0-9-]+\.(?:com|net|org|io|jp|co\.jp|ac\.jp)\b"]
LOCAL = ROOT / ".leak-patterns"  # 追跡しないファイル。ホスト名など、公開したくない固有の文字列を1行ずつ書く


def main() -> int:
    errors = []
    version = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    short = "v" + ".".join(version.split(".")[:2])

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

    patterns = GENERIC + ([re.escape(l.strip()) for l in LOCAL.read_text(encoding="utf-8").splitlines() if l.strip()]
                          if LOCAL.is_file() else [])
    leaks = re.compile("|".join(patterns))
    tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    for name in tracked:
        path = ROOT / name
        if path.suffix in {".png", ".pdf"} or not path.is_file():
            continue
        if leaks.search(path.read_text(encoding="utf-8", errors="ignore")):
            errors.append(f"{name} に実験機の情報が残っている")

    for e in errors:
        print("NG", e)
    print("OK" if not errors else f"{len(errors)} 件の問題", f"（{short}）")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
