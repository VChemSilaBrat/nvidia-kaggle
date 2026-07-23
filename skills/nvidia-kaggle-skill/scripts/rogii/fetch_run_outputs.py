#!/usr/bin/env python3
"""fetch_run_outputs.py — стянуть артефакты кернела в реестр репы (БЕЗ весов).

Запускается ИЗ КОРНЯ РЕПЫ ИНСТАНСА:
    uv run --project <форк> python <...>/scripts/rogii/fetch_run_outputs.py \\
        --kernel owner/train-exp-0001-<run_id>

Что делает:
  1. Качает output кернела (kaggle kernels output — файлы + лог) во временную папку.
  2. Куда класть — определяет по САМООПИСАНИЮ артефактов: manifest.json (train)
     или record.json (infer) несут exp и id. Для debug (манифеста нет) стадия
     и эксп берутся разбором слага кернела <stage>-<exp>-<id>.
  3. Кладёт в exps/<exp>/runs/<run_id>/ | infers/<infer_id>/ | debug/:
     - все файлы output, КРОМЕ весов (weights/ остаётся жить в output кернела —
       infer аттачит его напрямую) и служебного мусора кагла
       (__script__*, __results__.html, __output__.json, custom.css);
     - лог кернела: json-массив {stream_name, time, data} → читаемый log.txt.
  4. При ERROR кернела output может быть пуст — тогда тянем хотя бы лог
     (главный артефакт для диагностики падения).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # соседние rogii-модули
import common  # noqa: E402
import naming  # noqa: E402

# Служебные файлы, которые кагл добавляет в output кернела — в реестр не тащим.
_SERVICE_PREFIXES = ("__script__", "__results__", "__output__", "custom.css")

# id вида 20260723t080138-680-d078 — ровно 3 dash-куска (см. utils/run.py).
_ID_PARTS = 3


def parse_slug(slug: str) -> tuple[str, str, str]:
    """<stage>-<exp>-<id> → (stage, exp, id). Обратная к naming.kernel_slug.

    stage — один кусок, id — ровно _ID_PARTS кусков с конца, exp — середина.
    """
    parts = slug.split("-")
    if len(parts) < 1 + 1 + _ID_PARTS:
        raise SystemExit(f"[fetch] слаг '{slug}' не похож на <stage>-<exp>-<id>.")
    stage, exp_parts, id_parts = parts[0], parts[1:-_ID_PARTS], parts[-_ID_PARTS:]
    if stage not in ("train", "infer", "debug"):
        raise SystemExit(f"[fetch] неизвестная стадия '{stage}' в слаге '{slug}'.")
    return stage, "-".join(exp_parts), "-".join(id_parts)


def log_to_text(log_path: Path) -> str:
    """Лог кернела (json-массив) → читаемый текст с префиксами потоков."""
    entries = json.loads(log_path.read_text(encoding="utf-8"))
    return "".join(f"[{e['stream_name']}] {e['data']}" for e in entries)


def main() -> None:
    parser = argparse.ArgumentParser(description="стянуть output кернела в реестр")
    parser.add_argument("--kernel", required=True,
                        help="owner/slug кернела (печатает kaggle_run в конце)")
    args = parser.parse_args()

    root = common.repo_root_from_cwd()
    _, _, slug = args.kernel.partition("/")
    stage, exp, the_id = parse_slug(slug)

    with tempfile.TemporaryDirectory() as td:
        res = subprocess.run(
            [common.kaggle_cli(), "kernels", "output", args.kernel, "-p", td],
            capture_output=True, text=True,
        )
        if res.returncode != 0:
            raise SystemExit(
                f"[fetch] не удалось скачать output {args.kernel}:\n"
                f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
            )
        tmp = Path(td)

        # Куда класть: train/infer — сверяем самоописание артефакта со слагом;
        # debug — только слаг (манифеста нет by design).
        if stage == "train":
            man_path = tmp / "manifest.json"
            if man_path.exists():
                man = json.loads(man_path.read_text(encoding="utf-8"))
                if (man["exp"], man["run_id"]) != (exp, the_id):
                    raise SystemExit(
                        f"[fetch] манифест ({man['exp']}, {man['run_id']}) не совпал "
                        f"со слагом ({exp}, {the_id}) — что-то сильно не так."
                    )
            dest = root / "exps" / exp / "runs" / the_id
        elif stage == "infer":
            rec_path = tmp / "record.json"
            if rec_path.exists():
                rec = json.loads(rec_path.read_text(encoding="utf-8"))
                if (rec["exp"], rec["infer_id"]) != (exp, the_id):
                    raise SystemExit(
                        f"[fetch] record ({rec['exp']}, {rec['infer_id']}) не совпал "
                        f"со слагом ({exp}, {the_id}) — что-то сильно не так."
                    )
            dest = root / "exps" / exp / "infers" / the_id
        else:
            dest = root / "exps" / exp / "debug"
        dest.mkdir(parents=True, exist_ok=True)

        copied, skipped = [], []
        for path in sorted(tmp.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(tmp)
            if rel.parts[0] == naming.WEIGHTS_SUBDIR:
                skipped.append(str(rel))  # веса живут в output кернела, не в гите
                continue
            if rel.name.startswith(_SERVICE_PREFIXES) or rel.name in _SERVICE_PREFIXES:
                continue
            if rel.name == f"{slug}.log":
                (dest / "log.txt").write_text(log_to_text(path), encoding="utf-8")
                copied.append("log.txt")
                continue
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            copied.append(str(rel))

    if not copied:
        raise SystemExit(
            f"[fetch] в output {args.kernel} не оказалось ни одного файла "
            "(даже лога) — кернел вообще стартовал?"
        )
    print(f"[fetch] OK | {args.kernel} → {dest.relative_to(root)}/")
    print(f"[fetch] забрал: {', '.join(copied)}")
    if skipped:
        print(f"[fetch] веса оставлены в output кернела: {', '.join(skipped)}")


if __name__ == "__main__":
    main()
