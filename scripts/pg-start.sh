#!/usr/bin/env bash

stop_pg() {
  _NOW=$(date +"%Y%m%d%H%M")
  mkdir "/dump/$_NOW"
  pg_dump -U "${POSTGRES_USER:-germ}" -f "/dump/$_NOW/schema.sql" --schema-only
  pg_dump -U "${POSTGRES_USER:-germ}" -f "/dump/$_NOW/data.sql" --data-only
  su - postgres -c "$(find /usr/lib/postgresql -type f -name pg_ctl) -D $PGDATA stop"
}

trap stop_pg SIGINT SIGTERM

docker-entrypoint.sh postgres &
wait $!
