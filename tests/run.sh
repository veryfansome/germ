#!/usr/bin/env bash
set -ex

rm -rf /src/bot/static/tests
if [ "$SKIP_TESTS" == 'false' ]; then
    source germ_venv/bin/activate

    pytest -vvv \
        -n auto \
        --cov=bot \
        --cov-report=html:/src/bot/static/tests/cov \
        --junitxml=/src/bot/static/tests/report.xml \
        tests/test_unit_*

    if [ "$SKIP_INTEGRATION_TESTS" == 'false' ]; then
        # Wait for PG
        export WAIT_FOR_HOST=$PG_HOST WAIT_FOR_PORT=5432
        bash /src/scripts/wait-for-port-up.sh
        # Wait for Neo4j
        export WAIT_FOR_HOST=$NEO4J_HOST WAIT_FOR_PORT=7474
        bash /src/scripts/wait-for-port-up.sh

        pytest -vvv \
            -n auto \
            --cov=bot \
            --cov-report=html:/src/bot/static/tests/cov \
            --junitxml=/src/bot/static/tests/report.xml \
            tests/test_int_*
    fi
fi