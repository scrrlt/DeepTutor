# Contributing to DeepTutor

Thank you for contributing to DeepTutor. We welcome contributions from all
skill levels and prioritize correctness, safety, and maintainability.

Join the community for discussion and support:

<p align="center">
<a href="https://discord.gg/eRsjPgMU4t"><img src="https://img.shields.io/badge/Discord-Join_Community-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="Discord"></a>&nbsp;
<a href="https://github.com/HKUDS/DeepTutor/issues/78"><img src="https://img.shields.io/badge/WeChat-Join_Group-07C160?style=for-the-badge&logo=wechat&logoColor=white" alt="WeChat"></a>&nbsp;
<a href="./Communication.md"><img src="https://img.shields.io/badge/Feishu-Join_Group-00D4AA?style=for-the-badge&logo=feishu&logoColor=white" alt="Feishu"></a>
</p>

## Table of Contents
- [1. Contribution workflow](#1-contribution-workflow)
- [2. Tooling and quality gates](#2-tooling-and-quality-gates)
- [3. Coding standards](#3-coding-standards)
- [4. Documentation](#4-documentation)
- [5. Testing and coverage](#5-testing-and-coverage)
- [6. Security and data handling](#6-security-and-data-handling)
- [7. Development setup](#7-development-setup)
- [8. Commit message format](#8-commit-message-format)
- [9. Getting started](#9-getting-started)

## 1. Contribution workflow
- All work must branch from `dev`.
- Sync before starting:
   ```bash
   git checkout dev && git pull origin dev
   ```
- Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
- Target pull requests to `dev` (not `main`).
- Run all pre-commit checks before submitting.

## 2. Tooling and quality gates
Tooling is configured in `pyproject.toml` and `.pre-commit-config.yaml`.

Core tools:
- **Ruff** (formatter + linter; preferred).
- **Black** (optional formatter).
- **Prettier** (frontend/config formatting).
- **mypy** or **pyright** (type checking).
- **Bandit** (security linting).
- **pip-audit** and **detect-secrets** (dependency and secret scanning).
- **interrogate** (docstring coverage reporting).

Run the full quality gate locally:
```bash
pre-commit run --all-files
```

## 3. Coding standards
### Python style
- Format with Ruff or Black, using the line lengths in `pyproject.toml`
   (Black 88, Ruff 79).
- Indentation: 4 spaces. No trailing whitespace.
- Imports: absolute only, grouped as StdLib → Third Party → Local.
- No wildcard imports.
- Naming: `snake_case` (vars/funcs), `PascalCase` (classes),
   `UPPER_SNAKE_CASE` (constants).
- `print()` is not allowed. Use structured logging.

### Type hints (Python 3.12+)
- Type hints required for all function signatures.
- Use `|` for unions and built-in generics (`list[str]`, `dict[str, int]`).
- No `Any` in public interfaces; prefer `Protocol` or `object` if unknown.
- Define named constants for magic numbers.
- Prefer `Sequence`, `Iterable`, or `Mapping` over concrete containers in APIs.
- Use Pydantic for external payloads and `TypedDict` for internal structures.

### Async safety
- No blocking I/O inside `async def`.
- Keep strong references to background tasks.
- Use `asyncio.create_subprocess_exec` instead of `subprocess.run`.

## 4. Documentation
- Google-style docstrings with `Args:`, `Returns:`, `Raises:` sections.
- Types belong in signatures, not docstrings.
- Update comments to explain intent (the “why”).

## 5. Testing and coverage
- Framework: `pytest` + `pytest-asyncio`.
- Tests must avoid network and disk unless explicitly marked.
- Coverage targets:
   - 85% overall.
   - 100% branch coverage for critical logic (LLM/RAG pipelines).

Recommended commands:
```bash
pytest
pytest --cov=src --cov-report=term-missing
```

If you change LLM/embedding integrations, run integration tests with valid
provider credentials and set `DISABLE_SSL_VERIFY=true` when required.

## 6. Security and data handling
- Sanitize file uploads and enforce size limits (100MB general, 50MB PDF).
- Always use `shell=False` for subprocesses.
- Prefer `pathlib.Path` for filesystem operations.
- Run `bandit` or `ruff check --select S` for security linting.

## 7. Development setup
Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e ".[all]"
```

Install pre-commit hooks:
```bash
pip install pre-commit
pre-commit install
```

## 8. Commit message format
Use Conventional Commits and keep messages in the imperative mood.

```
<type>: <short description>

[optional body]
```

Allowed types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`.

## 9. Getting started
1. Pick a task from the issue tracker labeled `good first issue`.
2. Comment on the issue to claim it.
3. Submit a PR against `dev` once ready.
