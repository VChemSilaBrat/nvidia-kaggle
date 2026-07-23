"""common.py — общие функции gap-скриптов (только реально задублированное).

Ошибки — стоковый SystemExit с префиксом вызывающего скрипта, без обёрток.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path


def repo_root_from_cwd() -> Path:
    """Корень репы инстанса = git toplevel от cwd. Скрипты запускаются из репы."""
    res = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True
    )
    if res.returncode != 0:
        raise SystemExit("[rogii] запускай из репы инстанса (cwd не в git-репозитории).")
    return Path(res.stdout.strip())


def load_comp_config(root: Path) -> dict:
    """comp_config.json — ПЕРСОНАЛЬНЫЙ (в .gitignore), у каждого члена команды свой;
    в гите лежит только comp_config.example.json."""
    path = root / "comp_config.json"
    if not path.exists():
        raise SystemExit(
            "[rogii] нет comp_config.json — скопируй comp_config.example.json "
            "и заполни (файл персональный, в .gitignore)."
        )
    cfg = json.loads(path.read_text(encoding="utf-8"))
    for key in ("competition", "kaggle_username", "sources_dataset"):
        value = cfg.get(key, "")
        # кириллица/пробелы = незаполненный плейсхолдер из example
        if not value or any(ord(ch) > 127 or ch.isspace() for ch in value):
            raise SystemExit(
                f"[rogii] поле '{key}' в comp_config.json не заполнено (сейчас: '{value}')."
            )
    if "/" not in cfg["sources_dataset"]:
        raise SystemExit("[rogii] sources_dataset должен быть 'owner/slug'.")
    # Слаг обязан быть kaggle-каноничным ЗАРАНЕЕ: upstream upload_dataset
    # прогоняет title через slugify и МОЛЧА перепишет 'lung_src' в 'lung-src' —
    # датасет создастся под другим ref, чем ссылается весь конвейер.
    _, _, ds_slug = cfg["sources_dataset"].partition("/")
    if not re.fullmatch(r"[a-z0-9-]{6,50}", ds_slug):
        raise SystemExit(
            f"[rogii] слаг sources_dataset '{ds_slug}' не kaggle-каноничен: "
            "нужны только [a-z0-9-], длина 6-50. Переименуй в comp_config.json."
        )
    return cfg


def kaggle_cli() -> str:
    path = shutil.which("kaggle")
    if not path:
        raise SystemExit(
            "[rogii] kaggle CLI не найден в PATH. Запускай через uv:\n"
            "  uv run --project <путь-к-форку-nvidia-kaggle> python ... <скрипт> ..."
        )
    return path


def token_username() -> str:
    """Username владельца KAGGLE_API_TOKEN (сетевая интроспекция).

    Используется как ранняя явная проверка в push_sources (владелец датасета)
    и kaggle_run (kaggle_username конфига): опечатка в персональном конфиге
    иначе всплыла бы поздней невнятной ошибкой пуша.
    """
    token = os.environ.get("KAGGLE_API_TOKEN")
    if not token:
        raise SystemExit("[rogii] KAGGLE_API_TOKEN не выставлен в окружении.")
    from kagglesdk.kaggle_client import KaggleClient
    from kagglesdk.kaggle_creds import KaggleCredentials

    client = KaggleClient(api_token=token)
    return KaggleCredentials(client=client, access_token=token).introspect()
