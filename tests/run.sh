#!/usr/bin/env bash
set -ex

rm -rf /src/bot/static/tests
pytest -vvv \
  -n auto \
  --cov=bot \
  --cov-report=html:/src/bot/static/tests/cov \
  --junitxml=/src/bot/static/tests/report.xml tests