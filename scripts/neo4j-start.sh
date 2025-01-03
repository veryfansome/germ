#!/usr/bin/env bash

stop_neo4j() {
  _NOW=$(date +"%Y%m%M%d-%H%M")
  /var/lib/neo4j/bin/neo4j-admin server stop
  mkdir /dump/$_NOW
  /var/lib/neo4j/bin/neo4j-admin database dump neo4j --to-path=/dump/$_NOW
}

trap stop_neo4j SIGTERM
/startup/docker-entrypoint.sh neo4j