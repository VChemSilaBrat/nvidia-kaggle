"""naming.py — единственный источник имён кернелов и маунт-путей кагла.

Зачем один модуль: kaggle_run.py по этим функциям НАЗЫВАЕТ кернел, а
инжектор инфера по ним же СТРОИТ маунт-путь весов train-кернела. Две копии
этой логики, разъехавшиеся на символ, = инфер не находит веса. Поэтому
логика живёт здесь и только здесь; fetch-скрипты тоже импортят отсюда.

Маунт-пути — эмпирические факты (см. research/kaggle_facts.md инстанса,
пробы 2026-07-22): раскладка с типовыми префиксами одинакова на CPU/T4/P100:
    датасет:        /kaggle/input/datasets/<owner>/<slug>/
    output кернела: /kaggle/input/notebooks/<owner>/<kernel-slug>/
    соревнование:   /kaggle/input/competitions/<comp-slug>/

Никаких тихих трансформаций имён: имя экспа и id обязаны быть kaggle-safe
([a-z0-9-]) по конвенции репы (папки exps — через дефис: exp-0001; run_id
генерится kaggle-safe в utils/run.py). Не kaggle-safe → громкий ValueError.

Правило валидации: то, что МЫ МИНТИМ (слаги/id кернелов), проверяется строгим
чарсетом (_require_safe); то, что ПРИХОДИТ ИЗВНЕ (ref'ы датасетов/кернелов из
конфига и манифестов, слаг соревнования), проверяется только по форме
(_require_ref / непустота) — их именование задаёт кагл, не мы.

Только stdlib.
"""

from __future__ import annotations

import re

INPUT_ROOT = "/kaggle/input"

# Куда train пишет веса внутри кернела (относительно /kaggle/working);
# в output кернела эта папка появляется под тем же именем.
WEIGHTS_SUBDIR = "weights"

_SAFE_RE = re.compile(r"^[a-z0-9-]+$")

# Лимит длины слага: у датасетов кагла 6-50 (их scripts/constants.py);
# точный лимит для кернелов не проверяли — консервативно держим те же 50.
_MAX_SLUG_LEN = 50


def _require_safe(value: str, what: str) -> str:
    """Упасть громко, если кусок имени не kaggle-safe. Ничего не чинить молча."""
    if not _SAFE_RE.fullmatch(value):
        raise ValueError(
            f"{what} '{value}' не kaggle-safe: допустимы только [a-z0-9-]. "
            "Конвенция репы: папки exps через дефис (exp-0001), id генерятся "
            "kaggle-safe автоматически."
        )
    return value


def kernel_slug(stage: str, exp: str, the_id: str) -> str:
    """Слаг кернела одного запуска: <stage>-<exp>-<id>.

    Пример: train-exp-0001-20260722t143050-123-a1b2
    Один запуск = один кернел = один слаг; никогда не переиспользуется.
    """
    if stage not in ("train", "infer", "debug"):
        raise ValueError(f"Неизвестная стадия '{stage}'")
    slug = f"{stage}-{_require_safe(exp, 'имя экспа')}-{_require_safe(the_id, 'id')}"
    if len(slug) > _MAX_SLUG_LEN:
        raise ValueError(
            f"Слаг '{slug}' длиннее {_MAX_SLUG_LEN} символов — укороти имя экспа."
        )
    return slug


# id из utils/run.py (new_run_id): YYYYmmddtHHMMSS-<ms>-<4hex> — ровно 3
# dash-куска. parse_slug опирается на это; изменится формат id — менять тут.
_ID_PARTS = 3


def parse_slug(slug: str) -> tuple[str, str, str]:
    """<stage>-<exp>-<id> → (stage, exp, id). ОБРАТНАЯ функция к kernel_slug,
    живёт рядом с прямой, чтобы они не разъехались.

    stage — один кусок, id — ровно _ID_PARTS кусков с конца, exp — середина
    (сам содержит дефисы: exp-0001).
    """
    parts = slug.split("-")
    if len(parts) < 1 + 1 + _ID_PARTS:
        raise ValueError(f"слаг '{slug}' не похож на <stage>-<exp>-<id>.")
    stage, exp_parts, id_parts = parts[0], parts[1:-_ID_PARTS], parts[-_ID_PARTS:]
    if stage not in ("train", "infer", "debug"):
        raise ValueError(f"неизвестная стадия '{stage}' в слаге '{slug}'.")
    return stage, "-".join(exp_parts), "-".join(id_parts)


def kernel_ref(owner: str, stage: str, exp: str, the_id: str) -> str:
    """Полная ссылка кернела '<owner>/<slug>' — для kernel-metadata.json id,
    kernel_sources и weights-ref ('kernel:<owner>/<slug>') в манифесте."""
    return f"{_require_safe(owner, 'username')}/{kernel_slug(stage, exp, the_id)}"


def _require_ref(ref: str, what: str) -> str:
    """ref всегда одна строка 'owner/slug' — ровно так он ходит в конфиге,
    манифесте и kernel-metadata. Никаких сплитов у вызывающих."""
    owner, sep, slug = ref.partition("/")
    if not sep or not owner or not slug or "/" in slug:
        raise ValueError(f"{what} '{ref}' должен иметь вид 'owner/slug'.")
    return ref


def dataset_mount(ds_ref: str) -> str:
    """Маунт-путь датасета внутри кернела. ds_ref = 'owner/slug'."""
    return f"{INPUT_ROOT}/datasets/{_require_ref(ds_ref, 'dataset ref')}"


def notebook_mount(kernel_ref_str: str) -> str:
    """Маунт-путь output'а кернела (kernel_sources). kernel_ref_str = 'owner/slug'."""
    return f"{INPUT_ROOT}/notebooks/{_require_ref(kernel_ref_str, 'kernel ref')}"


def competition_mount(comp_slug: str) -> str:
    """Маунт-путь данных соревнования внутри кернела."""
    if not comp_slug or "/" in comp_slug:
        raise ValueError(f"слаг соревнования '{comp_slug}' должен быть непустым, без '/'.")
    return f"{INPUT_ROOT}/competitions/{comp_slug}"
