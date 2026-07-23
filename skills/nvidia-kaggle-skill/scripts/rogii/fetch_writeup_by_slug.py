#!/usr/bin/env python3
"""fetch_writeup_by_slug.py — standalone-врайтап по URL/слагу → markdown.

Дополняет upstream fetch_writeup.py, который умеет ТОЛЬКО competition-scoped
врайтапы (`.../competitions/<comp>/writeups/<slug>`, резолвит через лидерборд).
Этот берёт STANDALONE-врайтап (`kaggle.com/writeups/<user>/<slug>`) напрямую.

    uv run --project <форк> python <...>/scripts/rogii/fetch_writeup_by_slug.py \\
        https://www.kaggle.com/writeups/<user>/<slug> [output.md]

Механика (разведка 2026-07-24, см. kaggle_facts.md): внутренний web-сервис
`discussions.WriteUpsService/GetWriteUpBySlug` с payload {userName, slug} отдаёт
{title, topicId, message.rawMarkdown}. Комментарии не тянем (как и upstream).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # соседний runtime
from runtime import kaggle_web_service  # noqa: E402

# kaggle.com/writeups/<user>/<slug> (competition-scoped URL сюда НЕ подходит —
# для него upstream fetch_writeup.py).
URL_RE = re.compile(r"kaggle\.com/writeups/(?P<user>[^/]+)/(?P<slug>[^/?#]+)")


def fetch_writeup_by_slug(url: str) -> str:
    match = URL_RE.search(url)
    if not match:
        raise RuntimeError(
            "URL должен быть standalone-врайтапом вида "
            "https://www.kaggle.com/writeups/<user>/<slug>. Для "
            "competition-врайтапа (/competitions/<comp>/writeups/<slug>) — "
            "upstream fetch_writeup.py."
        )
    client = kaggle_web_service()
    data = client.post(
        "discussions.WriteUpsService/GetWriteUpBySlug",
        {"userName": match.group("user"), "slug": match.group("slug")},
    )
    message = data.get("message") or {}
    markdown = (message.get("rawMarkdown") or message.get("content") or "").strip()
    if not markdown:
        raise RuntimeError(f"Врайтап '{match.group('slug')}' вернул пустой контент.")

    title = data.get("title") or ""
    if title and not markdown.lstrip().startswith("#"):
        markdown = f"# {title}\n\n{markdown}"
    return re.sub(r"\n{3,}", "\n\n", markdown).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="standalone Kaggle writeup → markdown")
    parser.add_argument("url", help="https://www.kaggle.com/writeups/<user>/<slug>")
    parser.add_argument("output_path", nargs="?", help="куда сохранить md (иначе stdout)")
    args = parser.parse_args()

    if not re.match(r"^https?://", args.url):
        parser.error("url должен быть абсолютным http(s) URL врайтапа")

    try:
        content = fetch_writeup_by_slug(args.url)
    except Exception as exc:
        print(f"[writeup] Ошибка: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    if args.output_path:
        out = Path(args.output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(content, encoding="utf-8")
        print(f"[writeup] сохранено: {out} ({len(content)} симв)", file=sys.stderr)
    else:
        sys.stdout.buffer.write(content.encode("utf-8"))


if __name__ == "__main__":
    main()
