#!/usr/bin/env bash

while ! nc -z "$DB_HOST" 5432; do
  sleep 1
done

. germ/bin/activate
python -m db.models