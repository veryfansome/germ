#!/usr/bin/env bash
set -ex

rm -rf /src/bot/static/tests
if [ "$SKIP_TESTS" == 'false' ]; then
    pytest -vvv \
        -n auto \
        --cov=bot \
        --cov-report=html:/src/bot/static/tests/cov \
        --junitxml=/src/bot/static/tests/report.xml tests
fi