#!/usr/bin/env bash

stop_app() {
  _NOW=$(date +"%Y%m%d-%H%M")
  # Do something
}

trap stop_app SIGTERM

source germ_venv/bin/activate

if grep -q 'germ.services.bot.main' <<<"$1"; then
    ./scripts/update-static-files.sh
    python -m germ.data.wordnet_graph &
fi
mkdir -p /var/log/germ

exec gunicorn -c gunicorn_config.py "$1" --bind "0.0.0.0:$2"
