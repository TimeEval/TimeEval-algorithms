#!/usr/bin/env bash

set -e

# Check for algorithm main file
if [[ -z "$ALGORITHM_MAIN" ]]; then
    echo 'No algorithm main file specified. Environment variable $ALGORITHM_MAIN is empty!' >&2
    echo "Add the following to your Dockerfile: 'ENV ALGORITHM_MAIN=/app/<your algorithm main file>'" >&2
    exit 1
fi

# Create user with the supplied user ID and group ID to use the correct privileges on the mounted volumes
#   -- more information at the end of the file.
Z_UID=0
Z_GID=0
if [[ ! -z "$LOCAL_UID" ]] && [[ ! -z "$LOCAL_GID" ]]; then
    addgroup --quiet --gid "${LOCAL_GID}" group
    adduser --quiet --gid "${LOCAL_GID}" --uid "${LOCAL_UID}" --no-create-home --disabled-password --disabled-login --gecos "" user
    Z_UID=$LOCAL_UID
    Z_GID=$LOCAL_GID

    # Reset owner:group of working directory
    chown -R "$Z_UID:$Z_GID" .
fi

# Either run algorithm or the supplied executable
if [[ "$1" = "execute-algorithm" ]]; then
    shift
    exec setpriv --reuid=$Z_UID --regid=$Z_GID --init-groups -- java -jar "$ALGORITHM_MAIN" "$@"
else
    exec setpriv --reuid=$Z_UID --regid=$Z_GID --init-groups -- "$@"
fi

# To forward the GID of the group "akita" and the UID of the current user to the docker container run:
# LOCAL_GID=$(getent group akita | cut -d ':' -f 3)
# LOCAL_UID=$(id -u)
# docker run -e LOCAL_GID -e LOCAL_UID this-image:latest id
