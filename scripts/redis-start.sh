#!/usr/bin/env bash

stop_redis() {
  _NOW=$(date +"%Y%m%d%H%M")
  rm /data/dump.rdb  # Remove the link
  mkdir "/data/$_NOW"
  redis-cli save
  redis-cli shutdown
  mv /data/dump.rdb "/data/$_NOW"
}

trap stop_redis SIGINT SIGTERM

mapfile -t DUMP_DIRS < <(find /data/* -type d | sort -n)
DUMP_CNT=${#DUMP_DIRS[@]}
MAX_DUMP_CNT=3
if ((DUMP_CNT)); then
    # Link most recent dump file and retain MAX_DUMP_CNT most recent dumps
    LINK_CREATED=0
    for (( i=${#DUMP_DIRS[@]}-1; i>=0 && ! (LINK_CREATED); i-- )); do
        DUMP_FILE="${DUMP_DIRS[i]}/dump.rdb"
        if [ -f "$DUMP_FILE" ]; then
            echo "Linking data from $DUMP_FILE"
            ln -s "$DUMP_FILE" /data
            LINK_CREATED=1
        fi
    done
    if ((DUMP_CNT > MAX_DUMP_CNT)); then
        rm -rf "${DUMP_DIRS[@]:0:$((DUMP_CNT - MAX_DUMP_CNT))}"
    fi
fi

docker-entrypoint.sh redis-server &
wait $!
