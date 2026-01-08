# Quality and Governance

## Branching and releases
- Protected: `main`, `hk-dev` (require PR + green CI).
- Feature work: short-lived branches; rebase onto `hk-dev` before PR.
- Release tags follow semantic versioning; changelog via PR notes (automate later).

## CI gates
- Lint/format: ruff, prettier/eslint (frontend), optional black/isort if enabled.
- Type checks: mypy (Python), ts/next typecheck (frontend).
- Tests: pytest with coverage; frontend tests with coverage.
- Audits: pip-audit and npm audit (report-only workflow); Playwright audit allow-fail lane.
- Optional: OOM harness on schedule.

## Coverage
- Generate coverage artifacts for Python and Node. Enforce thresholds via pytest/nyc config when ready.

## Pre-commit
- Use `.pre-commit-config.yaml` for lint/format/security hooks; run `pre-commit run --all-files` locally.

## Dependencies
- Lockfiles must be regenerated after dependency changes (poetry.lock, package-lock.json).
- Track exceptions in `docs/dependency-audit.md`; avoid `--force` upgrades unless reviewed.

## Secrets and config
- No secrets in repo. Use `.env`/`.env.local`. Validate required env at startup.

## Documentation expectations
- Update docs when changing behavior, configuration, or workflows (configuration, testing, release notes).
