#!/usr/bin/env bash

source germ_venv/bin/activate

if grep -q 'germ.services.bot.main' <<<"$1"; then
    ./scripts/update-static-files.sh
fi
mkdir -p /var/log/germ

exec gunicorn -c gunicorn_config.py "$1" --bind "0.0.0.0:$2"
