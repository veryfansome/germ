#!/usr/bin/env bash
set -ex

rm -f "/var/run/$WAIT_FOR_DONE_FILE"
while ! nc -z "$WAIT_FOR_HOST" $WAIT_FOR_PORT; do
    sleep 1
done
if [ -n "$WAIT_FOR_AFTER_COMMAND" ]; then
    $WAIT_FOR_AFTER_COMMAND
fi
touch "/var/run/$WAIT_FOR_DONE_FILE"
sleep infinity