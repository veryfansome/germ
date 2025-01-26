#!/usr/bin/env bash

cd database || exit 1

# PostgreSQL
psql -U germ -h germ-db -d germ < /src/database/sql/germ.ddl

# Neo4j
neo4j-migrations -p $NEO4J_PASSWORD apply
