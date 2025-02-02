#!/usr/bin/env bash

cd database || exit 1

# PostgreSQL
cat sql/germ.ddl sql/germ_data.sql | psql -U germ -h germ-pg -d germ

# Neo4j
neo4j-migrations -p $NEO4J_PASSWORD apply
