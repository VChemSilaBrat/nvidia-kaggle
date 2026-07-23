#!/usr/bin/env python3
"""fetch_lb_scores.py — дозаписать LB-скоры в record.json инферов.

Запускается ИЗ КОРНЯ РЕПЫ ИНСТАНСА, в любой момент и сколько угодно раз:
    uv run --project <форк> python <...>/scripts/rogii/fetch_lb_scores.py

Механика (эмпирика в research/kaggle_facts.md, проба D):
  - сабмит уходит с `--message <infer_id>` (kaggle_run) → на кагле message
    лежит в колонке description списка сабмитов НАВСЕГДА;
  - скрипт читает `kaggle competitions submissions --csv` (структурированные
    колонки: ref, fileName, date, description, status, publicScore,
    privateScore — БЕЗ регулярок) и джойнит description == infer_id;
  - каждому record.json с недостающим скором дописывает lb_public
    (и lb_private — он появляется только после дедлайна соревнования).

Записи с обоими скорами не трогаются. Инфер без сабмита на LB — честно
репортится как pending (это нормально: не каждый инфер сабмитится).
"""

from __future__ import annotations

import csv
import io
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # соседние rogii-модули
import common  # noqa: E402


def lb_by_message(comp_slug: str) -> dict[str, dict]:
    """Все видимые сабмиты соревнования: {description: строка-словарь csv}.

    При дублях description берётся первая строка (kaggle отдаёт список
    новейшие-первыми) — повторный сабмит с тем же infer_id невозможен
    по построению (infer_id уникален на запуск), так что дубль означает
    ручной сабмит с совпавшим message и честно перекрывается новейшим.
    """
    res = subprocess.run(
        [common.kaggle_cli(), "competitions", "submissions", "-c", comp_slug, "--csv"],
        capture_output=True, text=True,
    )
    if res.returncode != 0:
        raise SystemExit(
            f"[lb] не удалось получить сабмиты {comp_slug}:\n"
            f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
        )
    rows = list(csv.DictReader(io.StringIO(res.stdout)))
    table: dict[str, dict] = {}
    for row in rows:
        desc = row.get("description", "")
        if desc and desc not in table:
            table[desc] = row
    return table


def _score(row: dict, column: str) -> float | None:
    value = (row.get(column) or "").strip()
    return float(value) if value else None


def main() -> None:
    root = common.repo_root_from_cwd()
    sys.path.insert(0, str(root))
    from utils import run as runlib  # noqa: E402 — схема record'а владеет run.py

    cfg = common.load_comp_config(root)
    submissions = lb_by_message(cfg["competition"])

    updated, pending, done = [], [], []
    for rec_path in sorted(root.glob("exps/*/infers/*/" + runlib.RECORD_NAME)):
        rec_dir = rec_path.parent
        rec = runlib.read_infer_record(rec_dir)
        if rec.lb_public is not None and rec.lb_private is not None:
            done.append(rec.infer_id)
            continue
        row = submissions.get(rec.infer_id)
        if row is None:
            pending.append(f"{rec.infer_id} (сабмита на LB нет)")
            continue
        if "complete" not in (row.get("status") or "").lower():
            pending.append(f"{rec.infer_id} (статус: {row.get('status')})")
            continue
        new_public = _score(row, "publicScore")
        new_private = _score(row, "privateScore")
        if new_public == rec.lb_public and new_private == rec.lb_private:
            done.append(rec.infer_id)
            continue
        rec.lb_public = new_public
        rec.lb_private = new_private
        runlib.write_infer_record(rec_dir, rec)
        updated.append(
            f"{rec.infer_id}: public={new_public}"
            + (f", private={new_private}" if new_private is not None else "")
        )

    for line in updated:
        print(f"[lb] дописан {line}")
    if pending:
        print(f"[lb] без скора ({len(pending)}): " + "; ".join(pending))
    print(f"[lb] итог: обновлено {len(updated)}, заполнено ранее {len(done)}, "
          f"ожидает {len(pending)}")


if __name__ == "__main__":
    main()
