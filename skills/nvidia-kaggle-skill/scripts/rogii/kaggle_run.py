#!/usr/bin/env python3
"""kaggle_run.py — атомарный build+push стадии экспа на Kaggle.

Запускается ИЗ КОРНЯ РЕПЫ ИНСТАНСА (корень определяется по cwd через git):
    uv run --project <путь-к-форку> python <...>/scripts/rogii/kaggle_run.py \\
        --exp exp-0001 --stage train --accelerator t4
    ... --stage infer --run-ids <id1>,<id2>          # бленд: несколько id
    ... --build-only                                  # собрать, не пушить

Фазы (атомарно, 1 запуск = 1 id = 1 кернел):
  0. pre-flight: git-корень, comp_config.json (персональный, в .gitignore),
     kaggle_username == юзер токена, D2 (assert_committed), сверка _stamp.json
     sources-датасета с локальным хешем — всегда, без обхода: ловит «забыл
     push_sources» ДО траты очереди
  1. mint id (train → run_id, infer → infer_id) + git_sha + EXPECTED_SRC_HASH
  2. build: exps/<exp>/.build/<id>/ = инжектированный <stage>.py
     + kernel-metadata.json (слаг/маунты — из naming.py, единственный источник)
  3. push: их submit_kernel.py сабпроцессом (train/debug — без --file;
     infer — --file submission.csv --message <infer_id>); поллит до конца
После завершения кернела артефакты тащит fetch_run_outputs.py (отдельно).

Утилиты инстанса (utils/stamp.py, utils/run.py) импортируются ИЗ РЕПЫ —
хеш считает тот же код, что проверяет его в кернеле. Дублей логики нет.

Веса: train получает WEIGHTS_REL ("weights/<run_id>/") и пишет его в манифест
как weights.path — путь ОТНОСИТЕЛЬНО КОРНЯ ИСТОЧНИКА. infer получает
WEIGHTS_PATHS = {run_id: <корень-маунта-по-ref> + "/" + path} — полные пути,
собранные из манифестов; никакого знания о под-папках в резолвере нет.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # соседние rogii-модули
import common  # noqa: E402
import naming  # noqa: E402

KAGGLE_WORKING = "/kaggle/working"

# P100 (sm_60) может не поддерживаться торчем текущего кагл-образа
# («no kernel image is available») — поэтому дефолт GPU = t4. Не менять
# на дефолт API (там P100!). См. research/kaggle_facts.md инстанса.
GPU_SHAPES = {"t4": "NvidiaTeslaT4", "p100": "NvidiaTeslaP100"}


def preflight_sources_fresh(cfg: dict, expected_hash: str) -> None:
    """Сверить стемп ПОСЛЕДНЕЙ версии sources-датасета с локальным хешем.

    Ловит «правил src, забыл push_sources» до пуша кернела. Вторая линия —
    stamp.verify внутри кернела (пиннинга версий у кагла нет, см. kaggle_facts):
    она ловит stale-attach между пушем и стартом, эта — экономит очередь.

    Имя/схема стемп-файла принадлежат utils/stamp.py (STAMP_NAME, read_stamp
    рядом с писателем stamp_payload) — здесь литералов нет.
    """
    from utils import stamp  # utils ИНСТАНСА (sys.path выставлен в main)

    with tempfile.TemporaryDirectory() as td:
        res = subprocess.run(
            [common.kaggle_cli(), "datasets", "download", cfg["sources_dataset"],
             "-f", stamp.STAMP_NAME, "-p", td, "--force"],
            capture_output=True, text=True,
        )
        if res.returncode != 0:
            raise SystemExit(
                f"[kaggle_run] не удалось скачать {stamp.STAMP_NAME} из "
                f"{cfg['sources_dataset']}.\n"
                "Датасет ещё не создан или недоступен → запусти push_sources.py.\n"
                f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
            )
        zipped = Path(td) / f"{stamp.STAMP_NAME}.zip"
        # Текущий CLI отдаёт мелкие файлы как есть (проверено эмпирически);
        # zip-вариант задокументирован для CLI — ветка оставлена, чтобы при
        # его срабатывании диагноз не врал.
        if not (Path(td) / stamp.STAMP_NAME).exists() and zipped.exists():
            print(f"[kaggle_run] {stamp.STAMP_NAME} пришёл zip'ом — распаковываю")
            with zipfile.ZipFile(zipped) as zf:
                zf.extract(stamp.STAMP_NAME, td)
        try:
            attached = stamp.read_stamp(td)["src_hash"]
        except FileNotFoundError:
            raise SystemExit(
                f"[kaggle_run] в sources-датасете нет {stamp.STAMP_NAME} — "
                "он запушен не push_sources'ом."
            ) from None
    if attached != expected_hash:
        raise SystemExit(
            "[kaggle_run] sources-датасет отстал от локального src:\n"
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
        raise SystemExit(f"[kaggle_run] ран {run_id} не найден в exps/*/runs/ — опечатка в id?")
    return matches[0]


def weights_source(ref: str, path: str, rid: str) -> tuple[str, str | None, str | None]:
    """По weights.ref и weights.path манифеста → (полный путь весов в кернеле,
    kernel_source для аттача, dataset_source для аттача).

    Корень источника выбирается по типу ref, полный путь = корень + path
    (path записан продюсером рана — резолвер не знает про под-папки).
    """
    if ref.startswith("kernel:"):
        kref = ref.removeprefix("kernel:")  # 'owner/slug' — владелец из манифеста!
        return f"{naming.notebook_mount(kref)}/{path}", kref, None
    if ref.startswith("dataset:"):
        dref = ref.removeprefix("dataset:")
        return f"{naming.dataset_mount(dref)}/{path}", None, dref
    if ref == "local":
        raise SystemExit(
            f"[kaggle_run] веса рана {rid} локальные — на кагл они не заливались. "
            "Сначала push_weights (мост local→kaggle), потом infer."
        )
    raise SystemExit(f"[kaggle_run] неизвестный weights.ref '{ref}' у рана {rid}.")


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

    # ---- фаза 0: локальный pre-flight (без сети — чтобы --build-only
    #      работал оффлайн и до первого push_sources) ----
    root = common.repo_root_from_cwd()
    sys.path.insert(0, str(root))
    from utils import run as runlib  # noqa: E402 — utils ИНСТАНСА
    from utils import stamp  # noqa: E402
    from utils.inject import MARKER_PREFIX, build_header  # noqa: E402 — общий формат шапки

    cfg = common.load_comp_config(root)
    user = cfg["kaggle_username"]

    stage_file = root / "exps" / args.exp / f"{args.stage}.py"
    if not stage_file.exists():
        raise SystemExit(f"[kaggle_run] нет файла стадии: {stage_file}")
    body = stage_file.read_text(encoding="utf-8")
    if MARKER_PREFIX in body:
        raise SystemExit(
            f"[kaggle_run] {stage_file} уже содержит инжектированную шапку — "
            "инжектить дважды нельзя."
        )

    # D2: гейт совпадает с тем, что хешируется (PAYLOAD_DIRS) + файл стадии.
    rel_stage = str(stage_file.relative_to(root))
    stamp.assert_committed([*stamp.PAYLOAD_DIRS, rel_stage], root)

    git_sha = stamp.current_git_sha(root)
    expected = stamp.compute_src_hash(root)

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
        values["WEIGHTS_REL"] = f"{naming.WEIGHTS_SUBDIR}/{the_id}/"
        values["WEIGHTS_REF"] = "kernel:" + naming.kernel_ref(user, "train", args.exp, the_id)
    elif args.stage == "infer":
        the_id = runlib.new_infer_id()
        runlib.infer_dir(args.exp, the_id, root)
        run_ids = [rid.strip() for rid in args.run_ids.split(",") if rid.strip()]
        weights_paths, weights_hashes = {}, {}
        for rid in run_ids:
            rdir = find_run_dir(root, rid)
            if not (rdir / runlib.MANIFEST_NAME).exists():
                raise SystemExit(
                    f"[kaggle_run] у рана {rid} нет manifest.json в {rdir} — "
                    "он ещё не зафетчен. Сначала: fetch_run_outputs.py --kernel "
                    f"<owner>/train-<exp>-{rid}"
                )
            manifest = runlib.read_manifest(rdir)
            if manifest.weights is None:
                raise SystemExit(f"[kaggle_run] у рана {rid} в манифесте нет весов.")
            full_path, ksrc, dsrc = weights_source(
                manifest.weights.ref, manifest.weights.path, rid
            )
            weights_paths[rid] = full_path
            weights_hashes[rid] = manifest.weights.hash
            if ksrc:
                kernel_sources.append(ksrc)
            if dsrc:
                dataset_sources.append(dsrc)
        values["INFER_ID"] = the_id
        values["RUN_IDS"] = run_ids
        values["WEIGHTS_PATHS"] = weights_paths
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
        "is_private": True,
        "enable_gpu": args.accelerator != "none",
        "enable_internet": False,
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
        print("[kaggle_run] внимание: сетевые проверки (токен, свежесть sources) "
              "при ручном пуше не выполняются; stale поймает verify в кернеле.")
        return

    # ---- фаза 3: сетевые проверки + push ----
    actual_user = common.token_username()
    if user != actual_user:
        raise SystemExit(
            f"[kaggle_run] kaggle_username в comp_config.json ('{user}') != "
            f"юзер токена ('{actual_user}'). Кернел пушится от юзера токена — "
            "исправь свой персональный конфиг."
        )
    preflight_sources_fresh(cfg, expected)

    print("[kaggle_run] пуш через submit_kernel.py ...")
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
        print("[kaggle_run] причина — в логе кернела: fetch_run_outputs.py --kernel "
              f"{user}/{slug} (тянет .log даже при ERROR).")
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
