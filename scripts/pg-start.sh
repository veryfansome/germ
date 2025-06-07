#!/usr/bin/env bash

stop_pg() {
  _NOW=$(date +"%Y%m%d-%H%M")
  mkdir /dump/$_NOW
  pg_dump -U "${POSTGRES_USER:-germ}" -f "/dump/$_NOW/dump.sql" --data-only \
      -t struct_type \
      -t chat_user \
      -t conversation \
      -t conversation_state \
      -t top_level_domain
  su - postgres -c "$(find /usr/lib/postgresql -type f -name pg_ctl) -D $PGDATA stop"
}

trap stop_pg SIGINT SIGTERM

docker-entrypoint.sh postgres &
wait $!
