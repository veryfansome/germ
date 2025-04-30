#!/usr/bin/env bash

TEMPLATED_STATIC_FILES=(
  "index.html"
  "login.html"
  "register.html"
)

stop_app() {
  _NOW=$(date +"%Y%m%d-%H%M")
  # Do something
}

trap stop_app SIGTERM

source germ_venv/bin/activate

# VERSION_SIGNATURE for cache busting
VERSION_SIGNATURE=$(find bot/static -type f -exec sha256sum {} \; | sort -k 2 | sha256sum | awk '{print $1}')
for STATIC_FILE in ${TEMPLATED_STATIC_FILES[*]}; do
    jinja2 bot/templates/${STATIC_FILE}.jinja -D version=${VERSION_SIGNATURE:0:7} -o bot/static/$STATIC_FILE
done

mkdir -p /var/lib/germ/messages
exec gunicorn -c gunicorn_config.py "$1" --bind "0.0.0.0:$2"
