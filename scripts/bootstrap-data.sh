#!/usr/bin/env bash
set -e

if ! [ -d data/oewn2024 ]; then
    curl -L \
        -o data/english-wordnet-2024.zip \
        https://github.com/globalwordnet/english-wordnet/releases/download/2024-edition/english-wordnet-2024.zip
    unzip data/english-wordnet-2024.zip -d data
    rm data/english-wordnet-2024.zip
fi

for TBL in categorylinks page; do
    TBL_DUMP_FILE=data/enwiki-latest-${TBL}.sql.gz
    if ! [ -f "$TBL_DUMP_FILE" ]; then
        curl -L \
            -o "$TBL_DUMP_FILE" \
            https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-${TBL}.sql.gz
    fi
    pigz -d -p 8 "$TBL_DUMP_FILE"
    mysql --user="$MYSQL_USER" --password="$MYSQL_PASSWORD" -h germ-mariadb -D germ < "$TBL_DUMP_FILE"
done

source germ_venv/bin/activate

mkdir -p /var/log/germ
#python -m germ.data.wordnet_graph
