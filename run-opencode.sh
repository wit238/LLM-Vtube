#!/usr/bin/env bash
# load .env into the environment then launch opencode
# - skips comments and empty values so it never blanks env-injected secrets
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ ! -f "$DIR/.env" ]; then
    echo "warning: no .env in $DIR" >&2
else
    while IFS= read -r line; do
        line="${line%"${line##*[![:space:]]}"}"
        case "$line" in
            '' | \#*) continue ;;
        esac
        key="${line%%=*}"
        value="${line#*=}"
        value="${value%"${value##*[![:space:]]}"}"
        case "$value" in
            \"*\" | \'*\') value="${value:1:${#value}-2}" ;;
        esac
        if [ -n "$value" ]; then
            export "$key=$value"
        fi
    done < "$DIR/.env"
fi
exec opencode "$@"