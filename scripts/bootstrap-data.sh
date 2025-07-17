#!/usr/bin/env bash
set -e

if ! [ -d data/oewn2024 ]; then
    curl -L \
        -o data/english-wordnet-2024.zip \
        https://github.com/globalwordnet/english-wordnet/releases/download/2024-edition/english-wordnet-2024.zip
    unzip data/english-wordnet-2024.zip -d data
    rm data/english-wordnet-2024.zip
fi

if ! [ -f data/enwiki-latest-categorylinks.sql.gz ]; then
    curl -L \
        -o data/enwiki-latest-categorylinks.sql.gz \
        https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-categorylinks.sql.gz
fi

if ! [ -f data/enwiki-latest-page.sql.gz ]; then
    curl -L \
        -o data/enwiki-latest-page.sql.gz \
        https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-page.sql.gz
fi

if ! [ -f data/enwiki-latest-redirect.sql.gz ]; then
    curl -L \
        -o data/enwiki-latest-redirect.sql.gz \
        https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-redirect.sql.gz
fi

source germ_venv/bin/activate

mkdir -p /var/log/germ
#python -m germ.data.wordnet_graph
