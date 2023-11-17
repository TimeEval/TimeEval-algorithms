#!/usr/bin/env bash

set -e

folder="${1:-}"
SEMVER_REGEX="^(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)(\\-[0-9A-Za-z-]+(\\.[0-9A-Za-z-]+)*)?(\\+[0-9A-Za-z-]+(\\.[0-9A-Za-z-]+)*)?$"

trim-and-validate() {
    local var="$*"
    # remove leading whitespace characters
    var="${var#"${var%%[![:space:]]*}"}"
    # remove trailing whitespace characters
    var="${var%"${var##*[![:space:]]}"}"

    # validate semver version string
    if [[ "$var" =~ $SEMVER_REGEX ]]; then
      printf '%s' "$var"
    else
      echo "Version $var is not a proper version string according to SemVer 'X.Y.Z(-PRERELEASE)(+BUILD)'!" >&2
      exit 1
    fi
}

if [[ -f "$folder/version.txt" ]]; then
  trim-and-validate "$( cat "$folder/version.txt" )"
elif [[ -f "$folder/manifest.json" ]]; then
  trim-and-validate "$( jq -r '.version' "$folder/manifest.json" )"
else
  echo "No version.txt or manifest.json present. Cannot determine Docker image version!" >&2
  exit 1
fi
