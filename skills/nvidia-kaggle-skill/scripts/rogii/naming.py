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
    return f"{INPUT_ROOT}/competitions/{comp_slug}"


def train_weights_mount(train_kernel_ref: str) -> str:
    """Корень весов рана, видимый из infer-кернела: маунт output'а
    train-кернела ('owner/train-<exp>-<run_id>') + подпапка weights.
    utils.weights.weights_path(корень, run_id) дальше сам добавит <run_id>/."""
    return f"{notebook_mount(train_kernel_ref)}/{WEIGHTS_SUBDIR}"
