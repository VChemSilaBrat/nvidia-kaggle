#!/usr/bin/env python3
"""push_sources.py — залить payload (src/utils/tests) как sources-датасет.

Запускается ИЗ КОРНЯ РЕПЫ ИНСТАНСА, нужен только когда менялся payload:
    uv run --project <форк> python <...>/scripts/rogii/push_sources.py

Поток (детерминированный):
  1. D2: assert_committed(PAYLOAD_DIRS) — грязный payload → крэш.
  2. staging (временная папка) ← копия PAYLOAD_DIRS. Состав задаёт
     ЕДИНСТВЕННАЯ константа stamp.PAYLOAD_DIRS — та же, по которой считается
     хеш. Симметрия hash(репа) == hash(staging) ассертится тут же.
  3. stamp_payload(staging) → _stamp.json {src_hash, pushed_at}
     (информационная метка; проверки живут на пересчёте хеша).
  4. их upload_dataset.py сабпроцессом: create (первый раз) или version.
     Владелец из sources_dataset обязан быть юзером токена: пушить можно
     только в СВОЙ датасет (у каждого члена команды свой; конфиг персональный).

Слаг берётся из comp_config.sources_dataset ('owner/slug'); title передаётся
равным слагу, чтобы их slugify(title) дал ровно его.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # соседние rogii-модули
import common  # noqa: E402


def dataset_exists(ds_ref: str) -> bool:
    """Детерминированная развилка create/version (не фолбэк по тексту ошибки)."""
    res = subprocess.run(
        [common.kaggle_cli(), "datasets", "status", ds_ref],
        capture_output=True, text=True,
    )
    return res.returncode == 0


def main() -> None:
    root = common.repo_root_from_cwd()
    sys.path.insert(0, str(root))
    from utils import stamp  # noqa: E402 — utils ИНСТАНСА (единый хеш-код)

    cfg = common.load_comp_config(root)
    ds_ref = cfg["sources_dataset"]
    owner, _, ds_slug = ds_ref.partition("/")

    user = common.token_username()
    if user != owner:
        raise SystemExit(
            f"[push_sources] владелец датасета '{owner}' != юзер токена '{user}'. "
            "Пушить можно только в свой sources-датасет (у каждого свой)."
        )

    # D2: git_sha, который поедет в манифесты ранов, должен описывать payload.
    stamp.assert_committed(list(stamp.PAYLOAD_DIRS), root)
    git_sha = stamp.current_git_sha(root)

    with tempfile.TemporaryDirectory() as td:
        staging = Path(td) / "staging"
        staging.mkdir()
        ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store", ".git")
        copied = []
        for dir_name in stamp.PAYLOAD_DIRS:
            src_dir = root / dir_name
            if src_dir.is_dir():
                shutil.copytree(src_dir, staging / dir_name, ignore=ignore)
                copied.append(dir_name)
        if not copied:
            raise SystemExit(
                f"[push_sources] в репе нет ни одной payload-папки из {stamp.PAYLOAD_DIRS}."
            )

        src_hash = stamp.stamp_payload(staging)
        local_hash = stamp.compute_src_hash(root)
        if src_hash != local_hash:
            raise SystemExit(
                "[push_sources] staging разошёлся с репой (баг копирования?):\n"
                f"  staging: {src_hash}\n  репа:    {local_hash}"
            )

        notes = f"src {src_hash[:19]} @ git {git_sha[:10]}"
        upload = Path(__file__).resolve().parents[1] / "upload_dataset.py"
        cmd = [sys.executable, str(upload), str(staging), "--title", ds_slug]
        if dataset_exists(ds_ref):
            cmd += ["--version-notes", notes]
            print(f"[push_sources] новая версия {ds_ref}: {notes}")
        else:
            print(f"[push_sources] первый пуш — создаю {ds_ref}")

        print(f"[push_sources] payload: {', '.join(copied)} | {src_hash}")
        proc = subprocess.run(cmd, text=True, capture_output=True)
        print(proc.stdout, end="")
        if proc.returncode != 0:
            raise SystemExit(
                f"[push_sources] upload_dataset.py упал:\n{proc.stdout}\n{proc.stderr}"
            )

    print(f"[push_sources] OK | {ds_ref} | {src_hash}")


if __name__ == "__main__":
    main()
