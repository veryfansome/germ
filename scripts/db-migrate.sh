#!/usr/bin/env bash

cd database || exit 1

# Neo4j
neo4j-migrations -p $NEO4J_PASSWORD apply

# PostgreSQL
# MD5 hash based approach until something more robust.
CURRENT_GERM_HASH=$(md5sum "./sql/germ.ddl" | awk '{print $1}')
CURRENT_GERM_DATA_HASH=$(md5sum "./sql/germ_data.sql" | awk '{print $1}')
CACHED_GERM_HASH_FILE="/tmp/germ.ddl.md5"
CACHED_GERM_DATA_HASH_FILE="/tmp/germ_data.sql.md5"
if [[ -f "$CACHED_GERM_HASH_FILE" ]]; then
    CACHED_GERM_HASH=$(cat "$CACHED_GERM_HASH_FILE")
    if [[ "$CURRENT_GERM_HASH" != "$CACHED_GERM_HASH" ]]; then
        echo "germ.ddl changed"
        psql -U "$POSTGRES_USER" -h "$PG_HOST" -d germ < sql/germ.ddl
    else
        echo "germ.ddl has not changed"
    fi
else
    psql -U "$POSTGRES_USER" -h "$PG_HOST" -d germ < sql/germ.ddl
fi
if [[ -f "$CACHED_GERM_DATA_HASH_FILE" ]]; then
    CACHED_GERM_DATA_HASH=$(cat "$CACHED_GERM_DATA_HASH_FILE")
    if [[ "$CURRENT_GERM_DATA_HASH" != "$CACHED_GERM_DATA_HASH" ]]; then
        echo "germ_data.sql changed"
        psql -U "$POSTGRES_USER" -h "$PG_HOST" -d germ < sql/germ_data.sql
    else
        echo "germ_data.sql has not changed"
    fi
else
    psql -U "$POSTGRES_USER" -h "$PG_HOST" -d germ < sql/germ_data.sql
fi
# Update cached MD5s
echo "$CURRENT_GERM_HASH" > "/tmp/germ.ddl.md5"
echo "$CURRENT_GERM_DATA_HASH" > "/tmp/germ_data.sql.md5"
