#!/usr/bin/env bash

if [ ! -e germ_venv ]; then
    python -m venv germ_venv
fi
source germ_venv/bin/activate

pip install --no-cache-dir -r requirements.txt
#opentelemetry-bootstrap -a install
