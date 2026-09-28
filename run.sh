#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python prepare.py
python validate.py
python submit.py
