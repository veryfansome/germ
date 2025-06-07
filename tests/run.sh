#!/usr/bin/env bash

PYTEST_OPTS=(
    "--cov=germ"
    "--cov-report=html:/src/germ/services/bot/static/tests/cov"
    "--junitxml=/src/germ/services/bot/static/tests/report.xml"
)

rm -rf /src/germ/bot/static/tests
if [ "$SKIP_TESTS" == 'false' ] || [ "$SKIP_INTEGRATION_TESTS" == 'false' ]; then
    echo "Starting test suite"
    source germ_venv/bin/activate
    set -e

    if [ "$SKIP_INTEGRATION_TESTS" == 'false' ]; then
        echo "Running all test cases"
        # Wait for PG
        export WAIT_FOR_HOST=$PG_HOST WAIT_FOR_PORT=5432
        bash /src/scripts/wait-for-port-up.sh
        # Wait for Neo4j
        export WAIT_FOR_HOST=$NEO4J_HOST WAIT_FOR_PORT=7474
        bash /src/scripts/wait-for-port-up.sh

        ./scripts/update-static-files.sh
        mkdir -p /var/log/germ

        pytest -vvv -n auto "${PYTEST_OPTS[@]}" tests
    elif [ "$SKIP_TESTS" == 'false' ]; then
        echo "Running unit test cases"
        pytest -vvv -n auto "${PYTEST_OPTS[@]}" tests/test_unit_*
    fi
fi
