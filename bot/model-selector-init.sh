#!/usr/bin/env bash
set -ex

python -m bot.model_selector

touch /var/run/done.model_selector
sleep infinity