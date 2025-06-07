#!/usr/bin/env bash

stop_neo4j() {
  _NOW=$(date +"%Y%m%d%H%M")
  mkdir "/dump/$_NOW"
  /var/lib/neo4j/bin/neo4j-admin server stop
  /var/lib/neo4j/bin/neo4j-admin database dump neo4j --to-path=/dump/$_NOW
}

trap stop_neo4j SIGINT SIGTERM

MAX_DUMP_CNT=3
DUMP_CNT=$(find /dump/* -type d | wc -l)
if ((DUMP_CNT)); then
    # Load the dump file and retain MAX_DUMP_CNT most recent dumps on a rolling basis
    /var/lib/neo4j/bin/neo4j-admin database load neo4j \
        --from-path="$(find /dump/* -type d | sort | tail -1)" \
        --overwrite-destination=true
    if ((DUMP_CNT > MAX_DUMP_CNT)); then
        find /dump/* -type d | sort | head -n $((DUMP_CNT - MAX_DUMP_CNT)) | xargs rm -rf
    fi
fi

/startup/docker-entrypoint.sh neo4j
