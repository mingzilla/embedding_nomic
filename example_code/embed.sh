#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${PROJECT_ROOT}/scripts_common/_project_common_util.sh"

# --- Configuration ---
CONFIG_DIR="${SCRIPT_DIR}"
CONFIG_FILE="${SCRIPT_DIR}/embedding_config.json"
CONFIG_NAME="nomic-embed-text-v1.5"

scripts_sh::embed "${CONFIG_DIR}" "${CONFIG_FILE}" "${CONFIG_NAME}"