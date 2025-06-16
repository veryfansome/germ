#!/usr/bin/env bash
set -e

TEMPLATED_STATIC_FILES=(
  "index.html"
  "login.html"
  "register.html"
)

source germ_venv/bin/activate

# VERSION_SIGNATURE for cache busting
VERSION_SIGNATURE=$(find germ/services/bot/static -type f -exec sha256sum {} \; | sort -k 2 | sha256sum | awk '{print $1}')
for STATIC_FILE in "${TEMPLATED_STATIC_FILES[@]}"; do
    jinja2 "germ/services/bot/templates/${STATIC_FILE}.jinja" -D "version=${VERSION_SIGNATURE:0:7}" -o "germ/services/bot/static/$STATIC_FILE"
done
