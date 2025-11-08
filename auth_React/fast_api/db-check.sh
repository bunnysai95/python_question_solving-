#!/usr/bin/env bash
set -euo pipefail

# db-check.sh - helper to inspect and backup the sqlite DB used by the backend
# Usage:
#   ./db-check.sh           # show DB path and user count
#   ./db-check.sh backup    # create backups/db.sqlite3.TIMESTAMP

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="$ROOT_DIR/.env"

# load DB_URL from .env if present
DB_URL_DEFAULT="sqlite:///./db.sqlite3"
DB_URL="$DB_URL_DEFAULT"
if [ -f "$ENV_FILE" ]; then
  # shellcheck disable=SC1090
  set -a
  # load only simple assignments
  . "$ENV_FILE"
  set +a
  # if DB_URL exported in env file, use it
  DB_URL="${DB_URL:-$DB_URL_DEFAULT}"
fi

# convert sqlite URI to path: sqlite:///./db.sqlite3 -> ./db.sqlite3
if [[ "$DB_URL" =~ ^sqlite:(///?)(.*)$ ]]; then
  DB_PATH_PART="${BASH_REMATCH[2]}"
  # if DB_PATH_PART is absolute (starts with /) use as-is, else relativize to ROOT_DIR
  if [[ "$DB_PATH_PART" = /* ]]; then
    DB_PATH="$DB_PATH_PART"
  else
    DB_PATH="$ROOT_DIR/$DB_PATH_PART"
  fi
else
  echo "Unsupported DB_URL: $DB_URL" >&2
  exit 1
fi

echo "DB path: $DB_PATH"

if [ ! -f "$DB_PATH" ]; then
  echo "Database file not found: $DB_PATH" >&2
  exit 1
fi

echo -n "User count: "
sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM users;" || echo "(query failed)"

if [ "${1:-}" = "backup" ]; then
  mkdir -p "$ROOT_DIR/backups"
  TS=$(date +%Y%m%d_%H%M%S)
  DEST="$ROOT_DIR/backups/db.sqlite3.$TS"
  cp -v "$DB_PATH" "$DEST"
  echo "Backup created: $DEST"
fi

echo "Done."
