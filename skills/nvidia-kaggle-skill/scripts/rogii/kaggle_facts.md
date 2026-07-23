# Эмпирические факты о Kaggle (пробы 2026-07-22, аккаунт bebra3222, CLI 2.2.3)

Все пробы — крошечные приватные script-кернелы `probe-*`. Это вход для дизайна
`kaggle_run.py` и fetch-скриптов. Перепроверять при смене поведения кагла.

## B. kernel_sources (веса через output кернела) — РАБОТАЕТ

- Поле `kernel_sources` в kernel-metadata.json валидно, output чужого кернела
  маунтится и читается.
- **Маунт-путь: `/kaggle/input/notebooks/<owner>/<kernel-slug>/`**
  (probe: `/kaggle/input/notebooks/bebra3222/probe-train-b1/weights/RID123/model.txt`).
- Кагл докидывает в output служебные файлы: `__script__.py`, `__script__.ipynb`,
  `__results__.html`, `__output__.json`, `custom.css` → fetch_run_outputs должен
  их фильтровать; наши файлы (weights/, manifest.json) лежат рядом нетронутыми.

## C. Маунт-пути и акселераторы

- **Схема маунтов универсальна (проверено на CPU, P100 и T4), с типовыми префиксами:**
  - датасеты:      `/kaggle/input/datasets/<owner>/<ds-slug>/`
  - output кернела: `/kaggle/input/notebooks/<owner>/<kernel-slug>/`
  - соревнование:  `/kaggle/input/competitions/<comp-slug>/`
  (классического плоского `/kaggle/input/<slug>` нет НИ на одном пуле.
  Комментарий старой репы «T4-пул монтирует плоско» — устарел, кагл
  унифицировал раскладку → glob-фолбэки старой репы больше не нужны.)
- **Выбор акселератора — поле `machine_shape`** (найдено в старой репе,
  подтверждено пробой): валидные значения `NvidiaTeslaT4` и `NvidiaTeslaP100`.
  `machine_shape: NvidiaTeslaT4` → **Tesla T4 x2** (nvidia-smi показал две).
  Без machine_shape `enable_gpu: true` → **P100 (дефолт API)**.
- **P100 опасен для torch**: по опыту старой репы, PyTorch текущего кагл-образа
  не поддерживает sm_60 → «no kernel image is available». → В kaggle_run дефолт
  GPU = **T4** (`--accelerator none|t4|p100`), НЕ полагаться на дефолт API.
- Невалидное значение machine_shape НЕ отклоняется — молча даёт дефолт (P100):
  опечатка = не та карта. Использовать только точные строки + рантайм-принт карты.
- Docker-образы CPU и GPU разные (sha в KAGGLE_DOCKER_IMAGE).
- Env кернела: KAGGLE_KERNEL_RUN_TYPE=Batch, KAGGLE_DOCKER_IMAGE, KAGGLE_GCP_ZONE
  и внутренние токены (в логи не печатать).

## A. Версии датасета: лаг

- Старый баг («API выдаёт версию то с +1, то без») — это **stale отображение
  номера версии в API-ответах**, не аттач.
- Замер: `datasets version` (v2, крошечный payload) → кернел, запушенный
  **через ~5 секунд**, уже видел content v2; кернел через +60с — тоже v2.
  Stale-окна на маленьком датасете не поймано.
- Оговорка: на больших датасетах обработка версии дольше → окно возможно.
  **stamp.verify (пересчёт хеша) остаётся обязательной страховкой.**

## E. Пиннинг версии датасета

- `dataset_sources: ["owner/ds/versions/1"]` → **"Invalid dataset specification"**
  при пуше. Пиннинг НЕ поддерживается; кернел получает актуальную версию.
- Следствие: единственная защита от «подъехала чужая/новая версия» — stamp.verify.
  (Для команды некритично: у каждого свой sources-датасет.)

## D. Сабмит с --message (джойн lb-скоров) — РАБОТАЕТ

- Сабмит: `kaggle competitions submit -f ... -m "<infer_id>"` (у submit_kernel.py
  скила — то же через `--message`).
- `competitions submissions --csv` возвращает колонки:
  `ref, fileName, date, description, status, publicScore, privateScore`.
- **message приходит в `description` — точный строковый матч по infer_id работает**:
  проба нашла свою строку: description=probe-msg-..., status=COMPLETE,
  publicScore=0.62200 (нулевой Titanic-бейзлайн, ожидаемо), privateScore пуст
  до конца соревнования.
- → fetch_lb_scores: матч `description == infer_id`, дозапись lb_public
  (и privateScore после дедлайна). Без регулярок.

## Прочее

- Пуш кернела с невалидными полями metadata может проходить МОЛЧА (machine_shape) —
  не полагаться на валидацию пуша.
- Статусы кернела: QUEUED/RUNNING/COMPLETE/ERROR/CANCEL* (kagglesdk enum).
- Лог кернела фетчится `kaggle kernels output` как `<slug>.log`,
  формат: json-массив {stream_name, time, data}.
- Крошечный CPU-кернел проходит очередь и бежит ~15-45с.
