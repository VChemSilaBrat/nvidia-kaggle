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


def log_to_text(log_path: Path) -> str:
    """Лог кернела (json-массив) → читаемый текст с префиксами потоков."""
    entries = json.loads(log_path.read_text(encoding="utf-8"))
    return "".join(f"[{e['stream_name']}] {e['data']}" for e in entries)


def main() -> None:
    parser = argparse.ArgumentParser(description="стянуть output кернела в реестр")
    parser.add_argument("--kernel", required=True,
                        help="owner/slug кернела (печатает kaggle_run в конце)")
    parser.add_argument(
        "--allow-missing", action="store_true",
        help="разрешить отсутствие manifest/record (диагностика упавшего кернела: "
             "забрать хотя бы лог)",
    )
    args = parser.parse_args()

    root = common.repo_root_from_cwd()
    sys.path.insert(0, str(root))
    from utils import run as runlib  # noqa: E402 — реестр/схемы владеет run.py

    _, _, slug = args.kernel.partition("/")
    try:
        stage, exp, the_id = naming.parse_slug(slug)
    except ValueError as e:
        raise SystemExit(f"[fetch] {e}") from None

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
        # train/infer обязаны самоописаться (manifest/record); их отсутствие =
        # кернел упал до записи провенанса → без --allow-missing НЕ регистрируем
        # ран как успешный (иначе тихая дыра в провенансе). debug манифеста
        # не пишет by design.
        if stage == "train":
            man_path = tmp / runlib.MANIFEST_NAME
            if man_path.exists():
                man = runlib.read_manifest(tmp)
                if (man.exp, man.run_id) != (exp, the_id):
                    raise SystemExit(
                        f"[fetch] манифест ({man.exp}, {man.run_id}) не совпал "
                        f"со слагом ({exp}, {the_id}) — что-то сильно не так."
                    )
            elif not args.allow_missing:
                raise SystemExit(
                    f"[fetch] в output {args.kernel} НЕТ manifest.json — train упал "
                    "до write_manifest? Ран не регистрируется. Чтобы забрать лог "
                    "для диагностики: повтори с --allow-missing."
                )
            else:
                print("[fetch] WARNING: manifest.json отсутствует (--allow-missing) — "
                      "это НЕ валидный ран, забираю что есть для диагностики")
            dest = runlib.run_dir(exp, the_id, root)
        elif stage == "infer":
            rec_path = tmp / runlib.RECORD_NAME
            if rec_path.exists():
                rec = runlib.read_infer_record(tmp)
                if (rec.exp, rec.infer_id) != (exp, the_id):
                    raise SystemExit(
                        f"[fetch] record ({rec.exp}, {rec.infer_id}) не совпал "
                        f"со слагом ({exp}, {the_id}) — что-то сильно не так."
                    )
            elif not args.allow_missing:
                raise SystemExit(
                    f"[fetch] в output {args.kernel} НЕТ record.json — infer упал "
                    "до write_infer_record? Не регистрируется. Для лога: --allow-missing."
                )
            else:
                print("[fetch] WARNING: record.json отсутствует (--allow-missing) — "
                      "это НЕ валидный инфер, забираю что есть для диагностики")
            dest = runlib.infer_dir(exp, the_id, root)
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
            if rel.name.startswith(_SERVICE_PREFIXES):
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
