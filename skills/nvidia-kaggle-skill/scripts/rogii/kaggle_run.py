#!/usr/bin/env python3
"""kaggle_run.py — атомарный build+push стадии экспа на Kaggle.

Запускается ИЗ КОРНЯ РЕПЫ ИНСТАНСА (корень определяется по cwd через git):
    uv run --project <путь-к-форку> python <...>/scripts/rogii/kaggle_run.py \\
        --exp exp-0001 --stage train --accelerator t4
    ... --stage infer --run-ids <id1>,<id2>          # бленд: несколько id
    ... --build-only                                  # собрать, не пушить

Фазы (атомарно, 1 запуск = 1 id = 1 кернел):
  0. pre-flight: git-корень, comp_config.json, KAGGLE_API_TOKEN, D2
     (assert_committed), сверка _stamp.json sources-датасета с локальным
     хешем (всегда, без обхода — ловит «забыл push_sources» ДО траты очереди)
  1. mint id (train → run_id, infer → infer_id) + git_sha + EXPECTED_SRC_HASH
  2. build: exps/<exp>/.build/<id>/ = инжектированный <stage>.py
     + kernel-metadata.json (слаг/маунты — из naming.py, единственный источник)
  3. push: их submit_kernel.py сабпроцессом (train/debug — без --file;
     infer — --file submission.csv --message <infer_id>); поллит до конца
После завершения кернела артефакты тащит fetch_run_outputs.py (отдельно).

Утилиты инстанса (utils/stamp.py, utils/run.py) импортируются ИЗ РЕПЫ —
хеш считает тот же код, что проверяет его в кернеле. Дублей логики нет.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # соседний naming.py
import naming  # noqa: E402

MARKER_PREFIX = "# === injected by"
KAGGLE_WORKING = "/kaggle/working"

# P100 (sm_60) может не поддерживаться торчем текущего кагл-образа
# («no kernel image is available») — поэтому дефолт GPU = t4. Не менять
# на дефолт API (там P100!). См. research/kaggle_facts.md инстанса.
GPU_SHAPES = {"t4": "NvidiaTeslaT4", "p100": "NvidiaTeslaP100"}


def die(msg: str) -> None:
    raise SystemExit(f"[kaggle_run] ОШИБКА: {msg}")


def sh(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    """Запуск команды; упавшая команда → громкая ошибка с полным выводом."""
    res = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if res.returncode != 0:
        die(
            f"команда упала: {' '.join(cmd)}\n"
            f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
        )
    return res


def repo_root_from_cwd() -> Path:
    res = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True
    )
    if res.returncode != 0:
        die("запускай из репы инстанса (cwd не в git-репозитории).")
    return Path(res.stdout.strip())


def load_comp_config(root: Path) -> dict:
    path = root / "comp_config.json"
    if not path.exists():
        die("нет comp_config.json — скопируй comp_config.example.json и заполни.")
    cfg = json.loads(path.read_text(encoding="utf-8"))
    for key in ("competition", "kaggle_username", "sources_dataset"):
        value = cfg.get(key, "")
        # кириллица/пробелы = незаполненный плейсхолдер из example
        if not value or any(ord(ch) > 127 or ch.isspace() for ch in value):
            die(f"поле '{key}' в comp_config.json не заполнено (сейчас: '{value}').")
    if "/" not in cfg["sources_dataset"]:
        die("sources_dataset должен быть 'owner/slug'.")
    return cfg


def kaggle_cli() -> str:
    path = shutil.which("kaggle")
    if not path:
        die(
            "kaggle CLI не найден в PATH. Запускай через uv:\n"
            "  uv run --project <путь-к-форку-nvidia-kaggle> python ... kaggle_run.py ..."
        )
    return path


def preflight_sources_fresh(cfg: dict, expected_hash: str) -> None:
    """Сверить _stamp.json ПОСЛЕДНЕЙ версии sources-датасета с локальным хешем.

    Ловит «правил src, забыл push_sources» до пуша кернела. Вторая линия —
    stamp.verify внутри кернела (пиннинга версий у кагла нет, см. kaggle_facts).
    """
    with tempfile.TemporaryDirectory() as td:
        res = subprocess.run(
            [kaggle_cli(), "datasets", "download", cfg["sources_dataset"],
             "-f", "_stamp.json", "-p", td, "--force"],
            capture_output=True, text=True,
        )
        if res.returncode != 0:
            die(
                f"не удалось скачать _stamp.json из {cfg['sources_dataset']}.\n"
                "Датасет ещё не создан или недоступен → запусти push_sources.py.\n"
                f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
            )
        stamp_file = Path(td) / "_stamp.json"
        zipped = Path(td) / "_stamp.json.zip"
        if not stamp_file.exists() and zipped.exists():
            with zipfile.ZipFile(zipped) as zf:
                zf.extract("_stamp.json", td)
        if not stamp_file.exists():
            die("в sources-датасете нет _stamp.json — он запушен не push_sources'ом.")
        attached = json.loads(stamp_file.read_text(encoding="utf-8"))["src_hash"]
    if attached != expected_hash:
        die(
            "sources-датасет отстал от локального src:\n"
            f"  локальный payload: {expected_hash}\n"
            f"  в датасете:        {attached}\n"
            "Запусти push_sources.py и повтори."
        )


def find_run_dir(root: Path, run_id: str) -> Path:
    """Папка рана по id среди всех экспов (id глобально уникален).

    (Крошечный дубль такого же глоба в local_run.py инстанса — сознательно:
    тащить ради 4 строк зависимость скила от ещё одного модуля не стоит.)
    """
    matches = list(root.glob(f"exps/*/runs/{run_id}"))
    if not matches:
        die(f"ран {run_id} не найден в exps/*/runs/ — опечатка в id?")
    return matches[0]


def build_header(script: str, exp: str, stage: str, values: dict) -> str:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines = [f"{MARKER_PREFIX} {script} @ {now} | {exp} | {stage} ==="]
    for name, value in values.items():
        lines.append(f"{name} = {value!r}")
    lines.append("# === end injected header ===")
    return "\n".join(lines) + "\n\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="build+push стадии экспа на Kaggle")
    parser.add_argument("--exp", required=True, help="имя экспа: exp-0001 (только [a-z0-9-])")
    parser.add_argument("--stage", required=True, choices=["train", "infer", "debug"])
    parser.add_argument("--run-ids", default="", help="infer: run_id весов через запятую")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--accelerator", choices=["none", "t4", "p100"], default="none",
        help="t4 — дефолт для GPU-тренировок; p100 может не работать с torch",
    )
    parser.add_argument("--file", default="submission.csv",
                        help="infer: имя файла сабмишна в /kaggle/working")
    parser.add_argument("--build-only", action="store_true",
                        help="собрать .build/ и выйти (не пушить)")
    args = parser.parse_args()

    # ---- фаза 0: pre-flight ----
    root = repo_root_from_cwd()
    sys.path.insert(0, str(root))
    from utils import run as runlib  # noqa: E402 — utils ИНСТАНСА
    from utils import stamp  # noqa: E402

    cfg = load_comp_config(root)
    if not os.environ.get("KAGGLE_API_TOKEN"):
        die("KAGGLE_API_TOKEN не выставлен в окружении.")

    stage_file = root / "exps" / args.exp / f"{args.stage}.py"
    if not stage_file.exists():
        die(f"нет файла стадии: {stage_file}")
    body = stage_file.read_text(encoding="utf-8")
    if MARKER_PREFIX in body:
        die(f"{stage_file} уже содержит инжектированную шапку — инжектить дважды нельзя.")

    rel_stage = str(stage_file.relative_to(root))
    stamp.assert_committed(["src", "utils", rel_stage], root)

    git_sha = stamp.current_git_sha(root)
    expected = stamp.compute_src_hash(root)
    preflight_sources_fresh(cfg, expected)

    user = cfg["kaggle_username"]

    # ---- фаза 1: mint + шапка ----
    values = {
        "EXP": args.exp,
        "SOURCES_DIR": naming.dataset_mount(cfg["sources_dataset"]),
        "INPUT_DIR": naming.competition_mount(cfg["competition"]),
        "OUTPUT_DIR": KAGGLE_WORKING,
        "GIT_SHA": git_sha,
        "EXPECTED_SRC_HASH": expected,
        "SEED": args.seed,
    }
    dataset_sources = [cfg["sources_dataset"]]
    kernel_sources: list[str] = []

    if args.stage == "train":
        the_id = runlib.new_run_id()
        runlib.run_dir(args.exp, the_id, root)  # место для будущего фетча
        values["RUN_ID"] = the_id
        values["WEIGHTS_DIR"] = f"{KAGGLE_WORKING}/{naming.WEIGHTS_SUBDIR}"
        values["WEIGHTS_REF"] = "kernel:" + naming.kernel_ref(user, "train", args.exp, the_id)
    elif args.stage == "infer":
        the_id = runlib.new_infer_id()
        runlib.infer_dir(args.exp, the_id, root)
        run_ids = [rid for rid in args.run_ids.split(",") if rid]
        weights_dirs, weights_hashes = {}, {}
        for rid in run_ids:
            manifest = runlib.read_manifest(find_run_dir(root, rid))
            if manifest.weights is None:
                die(f"у рана {rid} в манифесте нет весов.")
            ref = manifest.weights.ref
            if ref.startswith("kernel:"):
                kref = ref.removeprefix("kernel:")  # 'owner/slug' — владелец из манифеста!
                kernel_sources.append(kref)
                weights_dirs[rid] = naming.train_weights_mount(kref)
            elif ref.startswith("dataset:"):
                dref = ref.removeprefix("dataset:")
                dataset_sources.append(dref)
                weights_dirs[rid] = naming.dataset_mount(dref)
            elif ref == "local":
                die(
                    f"веса рана {rid} локальные — на кагл они не заливались. "
                    "Сначала push_weights (мост local→kaggle), потом infer."
                )
            else:
                die(f"неизвестный weights.ref '{ref}' у рана {rid}.")
            weights_hashes[rid] = manifest.weights.hash
        values["INFER_ID"] = the_id
        values["RUN_IDS"] = run_ids
        values["WEIGHTS_DIRS"] = weights_dirs
        values["WEIGHTS_HASHES"] = weights_hashes
    else:  # debug: id только для уникальности слага, реестр не трогаем
        the_id = runlib.new_run_id()

    slug = naming.kernel_slug(args.stage, args.exp, the_id)

    # ---- фаза 2: build ----
    staging = root / "exps" / args.exp / ".build" / the_id
    staging.mkdir(parents=True, exist_ok=True)
    code_file = f"{args.stage}.py"
    (staging / code_file).write_text(
        build_header("kaggle_run.py", args.exp, args.stage, values) + body,
        encoding="utf-8",
    )
    metadata = {
        "id": f"{user}/{slug}",
        "title": slug,
        "code_file": code_file,
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true" if args.accelerator != "none" else "false",
        "enable_internet": "false",
        "dataset_sources": dataset_sources,
        "competition_sources": [cfg["competition"]],
        "kernel_sources": kernel_sources,
    }
    if args.accelerator != "none":
        metadata["machine_shape"] = GPU_SHAPES[args.accelerator]
    (staging / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    print(f"[kaggle_run] сборка: {staging.relative_to(root)}")
    print(f"[kaggle_run] кернел: {user}/{slug} | accelerator={args.accelerator}")

    submit = Path(__file__).resolve().parents[1] / "submit_kernel.py"
    submit_cmd = [sys.executable, str(submit), str(staging)]
    if args.stage == "infer":
        submit_cmd += ["--file", args.file, "--message", the_id]

    if args.build_only:
        print("[kaggle_run] --build-only; пуш вручную (ровно ЭТОЙ сборки):")
        print("  PYTHONUNBUFFERED=1", " ".join(submit_cmd))
        print("[kaggle_run] повторный kaggle_run НЕ подойдёт — он заминтит новый id.")
        return

    # ---- фаза 3: push (их скрипт; поллит статус до терминального) ----
    print(f"[kaggle_run] пуш через submit_kernel.py ...")
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    proc = subprocess.Popen(
        submit_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, env=env, cwd=root,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
    proc.wait()

    where = f"https://www.kaggle.com/code/{user}/{slug}"
    if proc.returncode == 0:
        print(f"[kaggle_run] OK | {args.stage} {the_id} | {where}")
        print(f"[kaggle_run] дальше: fetch_run_outputs.py --kernel {user}/{slug}")
    else:
        print(f"[kaggle_run] FAILED (exit {proc.returncode}) | {where}")
        print("[kaggle_run] лог причины: fetch_run_outputs.py тем же вызовом (тянет .log).")
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
