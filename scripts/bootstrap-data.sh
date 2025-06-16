#!/usr/bin/env bash
set -e

if ! [ -d data/oewn2024 ]; then
    curl -L \
        -o data/english-wordnet-2024.zip \
        https://github.com/globalwordnet/english-wordnet/releases/download/2024-edition/english-wordnet-2024.zip
    unzip data/english-wordnet-2024.zip -d data
    rm data/english-wordnet-2024.zip
fi

source germ_venv/bin/activate

mkdir -p /var/log/germ
python -m germ.data.wordnet_graph
