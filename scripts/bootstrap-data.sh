#!/usr/bin/env bash
set -ex

if [ ! -d data/oewn2024 ]; then
    curl -L \
        -o data/english-wordnet-2024.zip \
        https://github.com/globalwordnet/english-wordnet/releases/download/2024-edition/english-wordnet-2024.zip
    unzip data/english-wordnet-2024.zip -d data
    rm data/english-wordnet-2024.zip
fi

for TBL in page page_props; do
    TBL_DUMP_FILE="enwiki-latest-${TBL}.sql"
    TBL_GZIP_FILE="${TBL_DUMP_FILE}.gz"
    if [ ! -f "data/${TBL_DUMP_FILE}" ]; then
        if [ ! -f "data/${TBL_GZIP_FILE}" ]; then
            curl -L \
                -o "data/$TBL_GZIP_FILE" \
                "https://dumps.wikimedia.org/enwiki/latest/${TBL_GZIP_FILE}"
        fi
        pigz -d -p 8 "data/$TBL_DUMP_FILE"
    fi
    #mysql --user="$MYSQL_USER" --password="$MYSQL_PASSWORD" -h germ-mariadb -D germ < "data/$TBL_DUMP_FILE"
done

set +x
source germ_venv/bin/activate

mkdir -p /var/log/germ
#python -m germ.data.wordnet_graph
