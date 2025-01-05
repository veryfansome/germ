#!/usr/bin/env bash

cd database
neo4j-migrations -p $NEO4J_PASSWORD apply
