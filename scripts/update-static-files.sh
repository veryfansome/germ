#!/usr/bin/env bash
set -e

TEMPLATED_STATIC_FILES=(
  "index.html"
  "login.html"
  "register.html"
)

if ! [ -d data/oewn2024 ]; then
    curl -L \
        -o data/english-wordnet-2024.zip \
        https://github.com/globalwordnet/english-wordnet/releases/download/2024-edition/english-wordnet-2024.zip
    unzip data/english-wordnet-2024.zip -d data
    rm data/english-wordnet-2024.zip
fi

source germ_venv/bin/activate

# VERSION_SIGNATURE for cache busting
VERSION_SIGNATURE=$(find germ/services/bot/static -type f -exec sha256sum {} \; | sort -k 2 | sha256sum | awk '{print $1}')
for STATIC_FILE in "${TEMPLATED_STATIC_FILES[@]}"; do
    jinja2 "germ/services/bot/templates/${STATIC_FILE}.jinja" -D "version=${VERSION_SIGNATURE:0:7}" -o "germ/services/bot/static/$STATIC_FILE"
done
