#!/usr/bin/env bash
set -e

CLEANUP_CHAT_DATA=0

cd database || exit 1

# Neo4j
neo4j-migrations -p $NEO4J_PASSWORD apply

# PostgreSQL
echo "Loading SQL schema"
psql -U "$POSTGRES_USER" -h "$POSTGRES_HOST" -d germ < sql/germ.ddl

mapfile -t DUMP_DIRS < <(find /src/database_dump/sql/* -type d | sort -n)
DUMP_CNT=${#DUMP_DIRS[@]}
MAX_DUMP_CNT=3
if ((DUMP_CNT)); then
    # Load most recent dump file and retain MAX_DUMP_CNT most recent dumps
    LOADED_FROM_DUMP=0
    for (( i=${#DUMP_DIRS[@]}-1; i>=0 && ! (LOADED_FROM_DUMP); i-- )); do
        DUMP_FILE="${DUMP_DIRS[i]}/data.sql"
        if [ -f "$DUMP_FILE" ]; then
            echo "Loading data from $DUMP_FILE"
            psql -U "$POSTGRES_USER" -h "$POSTGRES_HOST" -d germ < "$DUMP_FILE"
            LOADED_FROM_DUMP=1
        fi
    done
    if ! ((LOADED_FROM_DUMP)); then
        echo "Failed to load any dumps, loading initial data instead"
        psql -U "$POSTGRES_USER" -h "$POSTGRES_HOST" -d germ < sql/germ_data.sql
        CLEANUP_CHAT_DATA=1
    fi
    if ((DUMP_CNT > MAX_DUMP_CNT)); then
        rm -rf "${DUMP_DIRS[@]:0:$((DUMP_CNT - MAX_DUMP_CNT))}"
    fi
else
    # Load initial data if no dump files
    echo "Loading initial data"
    psql -U "$POSTGRES_USER" -h "$POSTGRES_HOST" -d germ < sql/germ_data.sql
    CLEANUP_CHAT_DATA=1
fi

# Application
CHAT_DATA_DIR="${GERM_DATA_DIR:-/opt/germ/data}/chat"
if ((CLEANUP_CHAT_DATA)); then
    rm -rf "${CHAT_DATA_DIR}"
fi
mkdir -p "$CHAT_DATA_DIR"