#!/usr/bin/env bash

cd database || exit 1

# Neo4j
neo4j-migrations -p $NEO4J_PASSWORD apply

# PostgreSQL
MAX_DUMP_CNT=3
DUMP_CNT=$(find /src/database_dump/sql/* -type d | wc -l)
if ((DUMP_CNT)); then
    # Load the dump file and retain MAX_DUMP_CNT most recent dumps on a rolling basis
    psql -U "$POSTGRES_USER" -h "$PG_HOST" -d germ < sql/germ.ddl
    psql -U "$POSTGRES_USER" -h "$PG_HOST" -d germ < "$(find /src/database_dump/sql/* -type d | sort | tail -1)/dump.sql"
    if ((DUMP_CNT > MAX_DUMP_CNT)); then
        find /src/database_dump/sql/* -type d | sort | head -n $((DUMP_CNT - MAX_DUMP_CNT)) | xargs rm -rf
    fi
else
    psql -U "$POSTGRES_USER" -h "$PG_HOST" -d germ < sql/germ.ddl
    psql -U "$POSTGRES_USER" -h "$PG_HOST" -d germ < sql/germ_data.sql
fi
