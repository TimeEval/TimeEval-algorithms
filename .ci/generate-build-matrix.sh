#!/usr/bin/env bash

set -e

default_branch=main
folder="${1:-.}"
ignore_pattern="0-base-images|1-intermediate-images|2-scripts|data|results|Dockerfile|README.md|\..*|.*\.py|.*\.yml|.*\.sh|.*\.png"
changes_in_basedir=""

function echoerr () {
  echo "$@" >&2
}

# GITHUB_EVENT_NAME=pull_request
# GITHUB_BASE_REF=PR target branch (probably default branch)
# GITHUB_HEAD_REF=PR source branch
# GITHUB_REF=refs/pull/<pr_number>/merge
# GITHUB_REF_TYPE=tag or branch
# RUNNER_ARCH=X86, X64, ARM, or ARM64
# RUNNER_OD=Linux, Windows, or macOS

# if this is a workflow for a PR targeting the default branch
if [[ "$GITHUB_EVENT_NAME" == "pull_request" ]] && [[ "$GITHUB_BASE_REF" == "$default_branch" ]]; then
  # build diff to main
  echoerr "Detected pipeline for a non-default branch (assuming pull request with target $GITHUB_BASE_REF)"
  git fetch origin || echoerr "Could not update remote 'origin'! Repository might be out of date."
  changes_in_basedir=$( git diff --name-only "refs/remotes/origin/$GITHUB_BASE_REF..HEAD" -- "$folder" | sed "s#${folder//\./\\.}/##" | cut -d '/' -f 1 )
  #changes_in_basedir=$( git diff --name-only "$GITHUB_BASE_REF..HEAD" | cut -d '/' -f 1 )

# if this is a workflow for the default branch
elif [[ "$GITHUB_EVENT_NAME" == "push" ]] && [[ "$GITHUB_BASE_REF" == "$default_branch" ]]; then
  # build latest commit for the default branch
  echoerr "Detected pipeline for default branch"
  #changes_in_basedir=$( git diff --name-only "$CI_COMMIT_BEFORE_SHA..$CI_COMMIT_SHA" )
  changes_in_basedir=$( git diff --name-only HEAD~1..HEAD -- "$folder" | sed "s#${folder//\./\\.}/##" | cut -d '/' -f 1 )

# if this is a tag-workflow: build all algorithm images
elif [[ "$GITHUB_EVENT_NAME" == "push" ]] && [[ "$GITHUB_REF_TYPE" == "tag" ]]; then
  echoerr "Detected pipeline for a tag"
  changes_in_basedir=$( ls -1 )

else
  echoerr "Cannot determine algorithm images to build! Please check the environment variables:"
  env | grep "GITHUB" >&2 && true
  echoerr ""
fi

# filter changes: remove non-algorithm-files/-folders and allow grep to find nothing (exit code 1)
changed_algos=$( echo "$changes_in_basedir" | sort | uniq | grep -x -v -E "${ignore_pattern}" || [[ $? == 1 ]] )
# filter changes: remove non-existing algos (e.g. when branch is not up-to-date with default branch or an algorithm was removed)
changed_algos=$( echo "$changed_algos" | while read -r f; do [[ -d "$folder/$f" ]] && echo "$f" || true; done )

if [[ -z "$changed_algos" ]]; then
  echoerr "No algorithm changed!"
fi

echoerr "Generating pipeline for algorithms: $(xargs <<<$changed_algos)"
(jq -Rc '[.]' | jq -sc '{"algorithm_name": add}') <<<"${changed_algos}"
