#!/usr/bin/env bash

set -e

default_branch=main
ignore_pattern="0-base-images|1-data|2-results|3-scripts|Dockerfile|README.md|\..*|.*\.py|.*\.yml|.*\.sh|.*\.png"
changes_in_basedir=""

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
  echo "Detected pipeline for a non-default branch (assuming pull request with target $GITHUB_BASE_REF)" >&2
  git fetch origin
  changes_in_basedir=$( git diff --name-only "refs/remotes/origin/$GITHUB_BASE_REF..HEAD" | cut -d '/' -f 1 )
  #changes_in_basedir=$( git diff --name-only "$GITHUB_BASE_REF..HEAD" | cut -d '/' -f 1 )

# if this is a workflow for the default branch
elif [[ "$GITHUB_EVENT_NAME" == "push" ]] && [[ "$GITHUB_BASE_REF" == "$default_branch" ]]; then
  # build latest commit for the default branch
  echo "Detected pipeline for default branch" >&2
  #changes_in_basedir=$( git diff --name-only "$CI_COMMIT_BEFORE_SHA..$CI_COMMIT_SHA" )
  changes_in_basedir=$( git diff --name-only HEAD~1..HEAD | cut -d '/' -f 1 )

# if this is a tag-workflow: build everything
elif [[ "$GITHUB_EVENT_NAME" == "push" ]] && [[ "$GITHUB_REF_TYPE" == "tag" ]]; then
    echo "Detected pipeline for a tag" >&2
    changes_in_basedir=$( ls -1 )
fi

# filter changes: remove non-algorithm-files/-folders and allow grep to find nothing (exit code 1)
changed_algos=$( echo "$changes_in_basedir" | sort | uniq | grep -x -v -E "${ignore_pattern}" || [[ $? == 1 ]] )
# filter changes: remove non-existing algos (e.g. when branch is not up-to-date with CI_DEFAULT_BRANCH)
changed_algos=$( echo "$changed_algos" | while read -r f; do [[ -d "$f" ]] && echo "$f" || true; done )
# flatten list
changed_algos=$( xargs <<<"${changed_algos}" )

if [[ -z "$changed_algos" ]]; then
  echo "No algorithm changed!" >&2
fi

echo "Generating pipeline for algorithms: $changed_algos" >&2
jq -Rc '{"algorithm_name":[.]}' <<<"${changed_algos}"



# use jq: input = whitespace-separated JSON-values
# jq -c '' <<<"${changed_algos}"
