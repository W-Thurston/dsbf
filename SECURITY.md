# Security Policy

## Supported versions

DSBF is currently pre-1.0. Security fixes are applied to the latest
version only.

| Version | Supported |
|---|---|
| Latest (`main`) | ✅ |
| Older tags | ❌ |

## Reporting a vulnerability

Please **do not** open a public GitHub issue for security vulnerabilities.

Report vulnerabilities by emailing **WilliamD.Thurston@gmail.com** with:

- A description of the vulnerability and its potential impact
- Steps to reproduce or a minimal proof of concept
- Any suggested remediation if you have one

You can expect an acknowledgement within 48 hours and a status update
within 7 days. If a fix is warranted, a patched release will be made
as soon as reasonably possible and you will be credited in the changelog
unless you prefer otherwise.

## Scope

DSBF is a local profiling tool — it reads datasets and writes reports
to disk. It does not transmit data externally, handle authentication,
or expose a public API by default. The FastAPI backend is intended for
local use only and should not be exposed to the public internet without
additional hardening.

Vulnerabilities in DSBF's dependencies should be reported to those
projects directly.
