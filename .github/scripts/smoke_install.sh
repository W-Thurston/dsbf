#!/usr/bin/env bash
# Install the built wheel into a clean venv and run it from a temporary directory.
#
#   smoke_install.sh <wheel>
#
# The base install has no extras and no lock file, so this is what `pip install dsbf`
# gives a user. `dsbf check` runs with a passing and a failing policy to cover the
# exit codes a pipeline step relies on. Then `[server]` is added, the run goes to the
# store, and `dsbf dashboard` must serve the bundled page and the run's summary
# (SMOKE_DASHBOARD=0 skips that part for a wheel built without the frontend).
set -euo pipefail

wheel=$(realpath "$1")
work=$(mktemp -d)
cd "$work"

uv venv --python "${SMOKE_PYTHON:-3.12}" venv
if [ -x venv/Scripts/python.exe ]; then python=venv/Scripts/python.exe; else python=venv/bin/python; fi
uv pip install --python "$python" "$wheel"
dsbf=("$python" -m dsbf.interfaces.cli)
if [ -x venv/Scripts/dsbf.exe ]; then dsbf=(venv/Scripts/dsbf.exe); elif [ -x venv/bin/dsbf ]; then dsbf=(venv/bin/dsbf); fi

printf 'id,score,label\n1,0.5,a\n2,0.7,b\n3,0.2,a\n4,0.9,c\n5,0.1,b\n' > small.csv
cat > pass.yaml <<'Y'
version: 1
rules: []
Y
cat > fail.yaml <<'Y'
version: 1
rules:
  - name: any-finding
    when: { severity: ">=info" }
    then: fail
Y

"${dsbf[@]}" --help > /dev/null
"${dsbf[@]}" version
"${dsbf[@]}" check small.csv --policy pass.yaml --out out-pass
test -s out-pass/run.json

status=0
"${dsbf[@]}" check small.csv --policy fail.yaml --out out-fail || status=$?
if [ "$status" -ne 1 ]; then
  echo "expected exit code 1 from the failing policy, got $status" >&2
  exit 1
fi
test -s out-fail/run.json

if [ "${SMOKE_DASHBOARD:-1}" = 1 ]; then
  # An extra needs a direct reference, and Git Bash does not convert a path that has
  # `[server]` stuck to it, so name the wheel by URL (a Windows path on Windows).
  if command -v cygpath > /dev/null; then url="file:///$(cygpath -m "$wheel")"; else url="file://$wheel"; fi
  uv pip install --python "$python" "dsbf[server] @ $url"
  export DSBF_HOME="$work/home"
  "${dsbf[@]}" check small.csv --sink store --out out-store
  port=${SMOKE_PORT:-8791}
  "${dsbf[@]}" dashboard --no-open --port "$port" &
  server=$!
  trap 'kill "$server" 2>/dev/null || true' EXIT
  "$python" - "$port" <<'P'
import json
import sys
import time
import urllib.request

base = f"http://127.0.0.1:{sys.argv[1]}"


def get(path):
    return urllib.request.urlopen(base + path, timeout=5).read().decode()


for _ in range(60):
    try:
        page = get("/")
        break
    except OSError:
        time.sleep(0.5)
else:
    sys.exit("the dashboard never answered")

assert '<div id="app">' in page, "the root page is not the dashboard"
assert "not installed" not in page, "the wheel has no dashboard files"
datasets = json.loads(get("/v2/datasets"))
dataset = datasets["items"][0]
runs = json.loads(get("/v2/datasets/" + urllib.request.quote(dataset["key"], safe="") + "/runs"))
summary = json.loads(get("/v2/runs/" + runs["items"][0]["run_id"]))
assert summary["header"]["health"], "the run summary has no health"
assert "<title>" in get("/runs/" + runs["items"][0]["run_id"]), "no page fallback"
print("dashboard served the page and a run summary")
P
fi
echo "smoke install passed"
