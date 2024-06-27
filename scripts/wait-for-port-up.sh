#!/usr/bin/env bash
set -e

MARK=$(date +%s)
while ! nc -w 1 -z "$WAIT_FOR_HOST" $WAIT_FOR_PORT; do
    if ! (($(date +%s)-MARK)); then
        sleep 1  # Sleep if time since MARK is less than 1s
    fi
    MARK=$(date +%s)
done
if [ -n "$WAIT_FOR_AFTER_COMMAND" ]; then
    $WAIT_FOR_AFTER_COMMAND
fi