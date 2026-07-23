#!/usr/bin/env python3
"""build_research_index.py — собрать research/fetched_index.md инстанса.

Запускается ИЗ КОРНЯ РЕПЫ ИНСТАНСА (лид, после ресёрч-сессии):
    uv run --project <форк> python <...>/scripts/rogii/build_research_index.py

Провенанс ресёрча в git-виде: сырьё (data/*.db, research/repos/) в .gitignore,
а этот индекс — текстовый, коммитится. Отвечает на «что уже зафетчено/склонировано»
(лиду в новой сессии — не тянуть повторно; напарнику — ре-фетчить по индексу).

Источники:
  - data/discussions.db, data/kernels.db (кэш скила; PROJECT_ROOT=корень инстанса);
  - research/repos/* (клоны из врайтапов — URL+commit через git).
Схема БД читается интроспекцией (pragma) — устойчиво к дрейфу upstream при merge.
Врайтапы не индексируем: скил сохраняет их как .md, они и так коммитятся.
"""

from __future__ import annotations

import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # соседний common
import common  # noqa: E402

INDEX_NAME = "fetched_index.md"


def _rows(db_path: Path, table: str, want: list[str]) -> list[dict]:
    """Строки таблицы по колонкам из want, которые реально существуют.

    Нет файла/таблицы → пустой список (ничего не фетчили — валидное состояние,
    не ошибка). Пересечение want с фактическими колонками = устойчивость к схеме.
    """
    if not db_path.exists():
        return []
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        tables = {r[0] for r in con.execute(
            "select name from sqlite_master where type='table'")}
        if table not in tables:
            return []
        have = {r[1] for r in con.execute(f"pragma table_info({table})")}
        cols = [c for c in want if c in have]
        if not cols:
            return []
        sel = ", ".join(cols)
        return [dict(r) for r in con.execute(f"select {sel} from {table}")]
    finally:
        con.close()


def _md_cell(value) -> str:
    s = "" if value is None else str(value)
    return s.replace("|", "\\|").replace("\n", " ").strip()


def _table(rows: list[dict], cols: list[str]) -> list[str]:
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(_md_cell(r.get(c)) for c in cols) + " |")
    return out


def _git(repo: Path, *args: str) -> str:
    res = subprocess.run(["git", "-C", str(repo), *args],
                         capture_output=True, text=True)
    return res.stdout.strip() if res.returncode == 0 else "?"


def repos_rows(root: Path) -> list[dict]:
    base = root / "research" / "repos"
    if not base.is_dir():
        return []
    rows = []
    for d in sorted(p for p in base.iterdir() if p.is_dir()):
        rows.append({
            "path": f"research/repos/{d.name}",
            "url": _git(d, "remote", "get-url", "origin"),
            "commit": _git(d, "rev-parse", "--short", "HEAD"),
        })
    return rows


def main() -> None:
    root = common.repo_root_from_cwd()
    data = root / "data"

    discussions = _rows(data / "discussions.db", "discussions",
                        ["competition_id", "discussion_id", "title",
                         "author_username", "votes", "url"])
    kernels = _rows(data / "kernels.db", "kernels",
                    ["competition_id", "kernel_ref", "title", "author",
                     "total_votes"])
    repos = repos_rows(root)

    now = datetime.now(timezone.utc).isoformat(timespec="minutes")
    lines = [
        "# Индекс ресёрча — сгенерировано build_research_index.py, руками не править",
        f"\nОбновлён: {now}. Сырьё в .gitignore (data/, research/repos/) — здесь провенанс.",
        f"\nДискашены: {len(discussions)} · кернелы: {len(kernels)} · репы: {len(repos)}",
    ]
    if discussions:
        lines.append("\n## Discussions\n")
        lines += _table(discussions,
                        ["competition_id", "discussion_id", "title",
                         "author_username", "votes", "url"])
    if kernels:
        lines.append("\n## Kernels\n")
        lines += _table(kernels,
                        ["competition_id", "kernel_ref", "title", "author",
                         "total_votes"])
    if repos:
        lines.append("\n## Repos (клоны из врайтапов; код лифтится в src/)\n")
        lines += _table(repos, ["path", "url", "commit"])
    if not (discussions or kernels or repos):
        lines.append("\nПока ничего не зафетчено.")

    out = root / "research" / INDEX_NAME
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[research-index] {out.relative_to(root)}: "
          f"дискашены {len(discussions)}, кернелы {len(kernels)}, репы {len(repos)}")


if __name__ == "__main__":
    main()
