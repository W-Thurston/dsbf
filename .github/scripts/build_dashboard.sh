#!/usr/bin/env bash
# Build the dashboard and copy it where the wheel picks it up.
#
#   build_dashboard.sh
#
# `uv build` packages everything under src/dsbf, so the built files go to
# src/dsbf/service/dashboard/ (gitignored, never committed). Run it before `uv build`;
# a wheel built without it serves the API and says the dashboard is missing.
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
pnpm --dir "$root/frontend" install --frozen-lockfile
pnpm --dir "$root/frontend" build
pnpm --dir "$root/frontend" check:bundle

target="$root/src/dsbf/service/dashboard"
rm -rf "$target"
cp -r "$root/frontend/dist" "$target"
test -s "$target/index.html"
echo "dashboard copied to $target"
