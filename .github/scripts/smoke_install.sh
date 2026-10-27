#!/usr/bin/env bash
# Install the built wheel into a clean venv and run it from a temporary directory.
#
#   smoke_install.sh <wheel>
#
# The base install has no extras and no lock file, so this is what `pip install dsbf`
# gives a user. `dsbf check` runs with a passing and a failing policy to cover the
# exit codes a pipeline step relies on.
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
echo "smoke install passed"
