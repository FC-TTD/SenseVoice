#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
probe=$(mktemp -d .container-context-probe.XXXXXX)
output=$(mktemp -d "${TMPDIR:-/tmp}/sensevoice-context.XXXXXX")
trap 'rm -rf -- "$probe" "$output"' EXIT

# Harmless sentinels exercise Docker's matcher without reading credentials.
mkdir -p "$probe/.git" "$probe/.ssh" "$probe/.aws"
touch "$probe/.git/sentinel" "$probe/.ssh/sentinel" "$probe/.aws/sentinel"
touch "$probe/.env" "$probe/.env.test" "$probe/.netrc" "$probe/.pypirc"
touch "$probe/github_token" "$probe/hf_token" "$probe/runtime-sentinel.txt"

docker buildx build --file - --output "type=local,dest=$output" . <<'DOCKERFILE'
FROM scratch
COPY . /
DOCKERFILE

test ! -e "$output/.git"
for excluded in .git .ssh .aws .env .env.test .netrc .pypirc github_token hf_token; do
    test ! -e "$output/$probe/$excluded"
done
for required in api.py model.py requirements.txt utils/device_env.py "$probe/runtime-sentinel.txt"; do
    test -f "$output/$required"
done
printf '%s\n' 'Docker context exclusions and runtime inputs verified.'
