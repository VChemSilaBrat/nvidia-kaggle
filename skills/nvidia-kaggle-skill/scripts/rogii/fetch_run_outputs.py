#!/usr/bin/env python3
"""fetch_run_outputs.py — стянуть артефакты кернела в реестр репы (БЕЗ весов).

Запускается ИЗ КОРНЯ РЕПЫ ИНСТАНСА:
    uv run --project <форк> python <...>/scripts/rogii/fetch_run_outputs.py \\
        --kernel owner/train-exp-0001-<run_id>
    ... --log-only        # только лог (диагностика упавшего кернела)
    ... --allow-missing   # забрать файлы, даже если нет manifest/record

Как качает: НЕ `kaggle kernels output` (он тянет весь output включая веса),
а listing-API кернела (пофайловые подписанные URL) → скачиваются ТОЛЬКО
нужные файлы. Веса (weights/) не скачиваются вовсе — они живут в output
кернела, infer аттачит их напрямую. Служебных файлов кагла в listing нет.

Куда класть — по слагу <stage>-<exp>-<id> (naming.parse_slug) со сверкой
самоописания артефакта (manifest.json / record.json через runlib):
exps/<exp>/runs/<run_id>/ | infers/<infer_id>/ | debug/. Лог кернела
(json-массив {stream_name, time, data}) конвертируется в читаемый log.txt.

train/infer без manifest/record = кернел упал до записи провенанса →
без --allow-missing ран НЕ регистрируется (тихих дыр в провенансе нет).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # соседние rogii-модули
import common  # noqa: E402
import naming  # noqa: E402

# Пояс на случай появления служебных файлов кагла в listing (сейчас их там нет).
_SERVICE_PREFIXES = ("__script__", "__results__", "__output__", "custom.css")


def list_output(kernel_ref: str) -> tuple[list, str]:
    """Пофайловый listing output'а кернела + лог. Пагинация учтена."""
    from kagglesdk import KaggleClient
    from kagglesdk.kernels.types.kernels_api_service import (
        ApiListKernelSessionOutputRequest,
    )

    owner, _, slug = kernel_ref.partition("/")
    files, log = [], ""
    with KaggleClient(api_token=os.environ["KAGGLE_API_TOKEN"]) as client:
        token = None
        while True:
            req = ApiListKernelSessionOutputRequest()
            req.user_name = owner
            req.kernel_slug = slug
            if token:
                req.page_token = token
            out = client.kernels.kernels_api_client.list_kernel_session_output(req)
            files.extend(out.files or [])
            log = out.log or log
            token = out.next_page_token
            if not token:
                break
    return files, log


def write_log(log_text: str, dest: Path) -> None:
    """Лог кернела (json-массив) → читаемый log.txt; непарсибельный — как есть."""
    try:
        entries = json.loads(log_text)
        text = "".join(f"[{e['stream_name']}] {e['data']}" for e in entries)
    except (json.JSONDecodeError, KeyError, TypeError):
        print("[fetch] лог не в ожидаемом json-формате — пишу как есть")
        text = log_text
    (dest / "log.txt").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="стянуть output кернела в реестр")
    parser.add_argument("--kernel", required=True,
                        help="owner/slug кернела (печатает kaggle_run в конце)")
    parser.add_argument(
        "--log-only", action="store_true",
        help="скачать только лог кернела (диагностика падения, ноль лишнего трафика)",
    )
    parser.add_argument(
        "--allow-missing", action="store_true",
        help="разрешить отсутствие manifest/record (кернел упал до их записи)",
    )
    args = parser.parse_args()

    if not os.environ.get("KAGGLE_API_TOKEN"):
        raise SystemExit("[fetch] KAGGLE_API_TOKEN не выставлен в окружении.")

    root = common.repo_root_from_cwd()
    sys.path.insert(0, str(root))
    from utils import run as runlib  # noqa: E402 — реестр/схемы владеет run.py

    _, _, slug = args.kernel.partition("/")
    try:
        stage, exp, the_id = naming.parse_slug(slug)
    except ValueError as e:
        raise SystemExit(f"[fetch] {e}") from None

    files, log_text = list_output(args.kernel)

    wanted, skipped = [], []
    for f in files:
        name = f.file_name
        if name.split("/", 1)[0] == naming.WEIGHTS_SUBDIR:
            skipped.append(name)  # веса живут в output кернела — не скачиваем
            continue
        if Path(name).name.startswith(_SERVICE_PREFIXES):
            continue
        wanted.append(f)

    if args.log_only:
        # только лог: дест по слагу, самоописание не проверяем (файлы не качаем)
        if stage == "train":
            dest = runlib.run_dir(exp, the_id, root)
        elif stage == "infer":
            dest = runlib.infer_dir(exp, the_id, root)
        else:
            dest = root / "exps" / exp / "debug"
            dest.mkdir(parents=True, exist_ok=True)
        write_log(log_text, dest)
        print(f"[fetch] OK (--log-only) | {args.kernel} → {dest.relative_to(root)}/log.txt")
        return

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for f in wanted:
            target = tmp / f.file_name
            target.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(f.url) as resp, target.open("wb") as fh:
                while chunk := resp.read(1 << 20):
                    fh.write(chunk)

        # train/infer обязаны самоописаться; отсутствие = кернел упал до записи
        # провенанса → без --allow-missing НЕ регистрируем (см. докстринг).
        if stage == "train":
            if (tmp / runlib.MANIFEST_NAME).exists():
                man = runlib.read_manifest(tmp)
                if (man.exp, man.run_id) != (exp, the_id):
                    raise SystemExit(
                        f"[fetch] манифест ({man.exp}, {man.run_id}) не совпал "
                        f"со слагом ({exp}, {the_id}) — что-то сильно не так."
                    )
            elif not args.allow_missing:
                raise SystemExit(
                    f"[fetch] в output {args.kernel} НЕТ manifest.json — train упал "
                    "до write_manifest? Ран не регистрируется. Для диагностики: "
                    "--allow-missing или --log-only."
                )
            else:
                print("[fetch] WARNING: manifest.json отсутствует (--allow-missing) — "
                      "это НЕ валидный ран, забираю что есть для диагностики")
            dest = runlib.run_dir(exp, the_id, root)
        elif stage == "infer":
            if (tmp / runlib.RECORD_NAME).exists():
                rec = runlib.read_infer_record(tmp)
                if (rec.exp, rec.infer_id) != (exp, the_id):
                    raise SystemExit(
                        f"[fetch] record ({rec.exp}, {rec.infer_id}) не совпал "
                        f"со слагом ({exp}, {the_id}) — что-то сильно не так."
                    )
            elif not args.allow_missing:
                raise SystemExit(
                    f"[fetch] в output {args.kernel} НЕТ record.json — infer упал "
                    "до write_infer_record? Не регистрируется. Для диагностики: "
                    "--allow-missing или --log-only."
                )
            else:
                print("[fetch] WARNING: record.json отсутствует (--allow-missing) — "
                      "это НЕ валидный инфер, забираю что есть для диагностики")
            dest = runlib.infer_dir(exp, the_id, root)
        else:
            dest = root / "exps" / exp / "debug"
        dest.mkdir(parents=True, exist_ok=True)

        copied = []
        for path in sorted(tmp.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(tmp)
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
            copied.append(str(rel))
        write_log(log_text, dest)
        copied.append("log.txt")

    print(f"[fetch] OK | {args.kernel} → {dest.relative_to(root)}/")
    print(f"[fetch] забрал: {', '.join(copied)}")
    if skipped:
        print(f"[fetch] веса НЕ скачивались (живут в output кернела): {', '.join(skipped)}")


if __name__ == "__main__":
    main()
