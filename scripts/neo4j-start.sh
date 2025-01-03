#!/usr/bin/env bash

stop_neo4j() {
  /var/lib/neo4j/bin/neo4j-admin server stop
  /var/lib/neo4j/bin/neo4j-admin database dump neo4j --to-path=/dump
}

trap stop_neo4j SIGTERM
/startup/docker-entrypoint.sh neo4j
