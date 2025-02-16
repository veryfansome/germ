#!/usr/bin/env bash

stop_app() {
  _NOW=$(date +"%Y%m%d-%H%M")
  # Do something
}

trap stop_app SIGTERM
source germ_venv/bin/activate
gunicorn -c gunicorn_config.py "$1" --bind "0.0.0.0:$2"
