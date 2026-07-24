#!/usr/bin/env bash
# Сессия-МЕЙНТЕЙНЕР форка: резолв upstream-PR и фичи в rogii/.
# Модель/эффорт переопределяются env-переменными.
set -euo pipefail
cd "$(dirname "$0")"
exec claude \
    --model "${MAINT_MODEL:-claude-opus-4-8}" \
    --effort "${MAINT_EFFORT:-xhigh}" \
    --append-system-prompt "$(cat .claude/roles/maintainer.md)"
