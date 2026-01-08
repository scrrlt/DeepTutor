# Dependency Audit and Monitoring

This project runs scheduled dependency audits for Python and Node.

## Automated audits (CI)
- Workflow: `.github/workflows/dependency-audit.yml`
- Schedule: daily at 06:00 UTC + manual dispatch
- Python: `pip-audit` on `requirements.txt`, JSON report uploaded as artifact
- Node: `npm audit --json`, JSON report uploaded as artifact
- Audits are report-only and do not fail the build.
- SBOMs: CycloneDX for Python and Node uploaded with audit artifacts.
- Exceptions: tracked in `security/audit-exceptions.yml` with owner and expiry.

## Local audit commands
- Python: `python -m pip install pip-audit && pip-audit -r requirements.txt`
- Node (from `web/`): `npm audit`

## Known exception policy
- `jspdf`/`dompurify` advisory remains open; no `--force` upgrade applied to avoid breaking changes.

## Remediation workflow
1) Review CI artifacts for findings.
2) Attempt non-breaking upgrades; regenerate lockfiles.
3) For breaking upgrades, stage changes in a feature branch, run full test suite, and document in PR.
4) Track remaining exceptions in this doc or issue tracker.
