#!/usr/bin/env bash
set -x

while ! nc -z "$DB_HOST" 5432; do
  sleep 1
done

python -m db.models