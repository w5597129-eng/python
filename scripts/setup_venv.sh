#!/usr/bin/env bash
set -euo pipefail

# Create a venv in .venv, activate it, upgrade pip, and install requirements
# Usage: bash scripts/setup_venv.sh

VENV_PATH=".venv"
REQUIREMENTS="requirements.txt"

if [ ! -f "$REQUIREMENTS" ]; then
  echo "requirements.txt not found in project root"
  exit 1
fi

python3 -m venv "$VENV_PATH"
# shellcheck disable=SC1091
source "$VENV_PATH/bin/activate"
python -m pip install --upgrade pip setuptools wheel
pip install -r "$REQUIREMENTS"

echo "Done. To activate the venv, run: source $VENV_PATH/bin/activate"
