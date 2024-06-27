#!/usr/bin/env bash
set -ex

rm -f /src/tests/report.*
pytest --junitxml=/src/tests/report.xml tests

if [ -z "$CI" ]; then  # Set by Github Actions
    sleep infinity
fi