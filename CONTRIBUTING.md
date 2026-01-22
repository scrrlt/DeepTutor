Contributing to DeepTutor 🚀
Thank you for your interest in contributing to DeepTutor! We are committed to building a smooth and robust intelligent learning companion, and we welcome developers of all skill levels to join us.
Join our community for discussion, support, and collaboration:
<p align="center">
<a href="https://discord.gg/eRsjPgMU4t"><img src="https://img.shields.io/badge/Discord-Join_Community-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="Discord"></a>&nbsp;
<a href="https://github.com/HKUDS/DeepTutor/issues/78"><img src="https://img.shields.io/badge/WeChat-Join_Group-07C160?style=for-the-badge&logo=wechat&logoColor=white" alt="WeChat"></a>&nbsp;
<a href="./Communication.md"><img src="https://img.shields.io/badge/Feishu-Join_Group-00D4AA?style=for-the-badge&logo=feishu&logoColor=white" alt="Feishu"></a>
</p>

## Table of Contents
- [Contribution Requirements](#️-contribution-requirements)
- [Code Quality & Security](#-code-quality--security)
- [Security Best Practices](#-security-best-practices)
- [Coding Standards](#-coding-standards)
- [Development Setup](#-development-setup)
- [Commit Message Format](#-commit-message-format)
- [How to Get Started](#-how-to-get-started)

## ⚠️ Contribution Requirements
[!IMPORTANT]
All contributions must be based on the `dev` branch!
1. Fork the repository and clone it locally.
2. **Synchronize**: Always pull from the `dev` branch before starting:
   ```bash
   git checkout dev && git pull origin dev
   ```
3. **Branch**: Create your feature branch from `dev`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **PR Target**: Submit your Pull Request to the `dev` branch (not `main`).
5. **Validation**: Ensure all pre-commit checks pass before submitting.

## 🛠️ Code Quality & Security
We use a suite of automated tools to maintain high standards of code quality and security. These are configured via `pyproject.toml` and `.pre-commit-config.yaml`.

### Our Toolstack
- **Ruff**: High-performance Python linting and formatting.
- **Black** (optional): Python formatter (allowed alternative to Ruff format).
- **Prettier**: Consistent formatting for frontend and configuration files.
- **detect-secrets**: Scans for high-entropy strings and hardcoded secrets.
- **pip-audit**: Scans dependencies for known vulnerabilities (V3.x compatible).
- **Bandit**: Analyzes code for common security issues.
- **MyPy**: Static type checking to ensure code reliability.
- **Interrogate**: Reports docstring coverage. While it won't block your commit, please aim for high coverage.

[!TIP]
Before submitting a PR, you MUST run:
```bash
pre-commit run --all-files
```
While local pre-commit hooks are configured to be lenient and may only show warnings, the CI will perform strict checks and will automatically reject PRs that fail.

## 🔒 Security Best Practices
### File Uploads
- **Size Limits**: General files are capped at 100MB; PDFs are capped at 50MB to prevent resource exhaustion.
- **Validation**: We enforce multi-layer validation (Extension + MIME type + Content sanitization).
- **Sanitization**: All filenames are sanitized to prevent path traversal attacks.

### Development Standards
- **Subprocesses**: Always use `shell=False` when calling subprocesses to prevent command injection.
- **Pathing**: Use `pathlib.Path` for all operations to ensure cross-platform (Windows/Linux/macOS) compatibility.
- **Line Endings**: LF (Unix) line endings are enforced for critical scripts via `.gitattributes`.

## 🧪 Testing & Coverage Requirements

### Coverage Goals
- **Docstring Coverage**: 100% for all exported symbols (functions, classes, modules)
- **Pytest Coverage**: 80-90% overall, with 100% branch coverage for critical business logic (LLM, embedding, RAG pipelines)

### Testing Frameworks
- **Unit Tests**: pytest with asyncio support
- **Integration Tests**: For LLM/embedding providers (requires API keys)
- **Type Checking**: mypy with strict mode
- **Linting**: ruff (replaces flake8, isort, pylint)

### Running Tests

#### Basic Test Suite
```bash
# Run all unit tests (no network calls)
pytest

# Run with coverage report
pytest --cov=src --cov-report=html
```

#### LLM & Embedding Tests (Required for Related Changes)
> [!WARNING]
> If you're working on LLM, embedding, or RAG features, you **MUST** run network tests to ensure providers work correctly.

**Prerequisites**: Populate `.env` with API keys for providers you want to test.


# SSL inspection must be disabled to avoid SSL/TLS errors during tests.
DISABLE_SSL_VERIFY=true  # Set to 'true' to disable SSL verification for LLM network tests.
                        # It is recommended to set this to 'true' only for testing purposes.

# [Optional] For VS Code users: To enable the Python extension to load environment
# variables from this .env file, set `python.terminal.useEnvFile` to 'True' in your
# VS Code settings (settings.json). This ensures the environment is correctly
# configured for network tests. Note that this setting is for the VS Code terminal,
# not for the Python interpreter itself. It is recommended to add this setting to
# your workspace settings (`.vscode/settings.json`) or user settings.

# If unable to set via IDE, you can manually load the .env file in your test setup
# code as follows (PowerShell example):
#
# $envContent = Get-Content .env -Raw
# foreach ($line in ($envContent -split "`n")) {
#     if ($line -match '^([^=]+)=(.*)$') {
#         $key = $Matches[1].Trim()
#         $value = $Matches[2].Trim()
#         [Environment]::SetEnvironmentVariable($key, $value, "Process")
#     }
# }

# If you need to test specific providers, use LLM_BINDING=<provider> with the integration tests.
```

#### Coverage Commands
```bash
# Check docstring coverage
interrogate -v src/

# Check pytest coverage
pytest --cov=src --cov-report=term-missing

# Combined quality check
pre-commit run --all-files
```
## 💻 Coding Standards
To keep the codebase maintainable, please follow these guidelines:
### Python Guidelines
- **Formatter:** Black or Ruff format.
- **Linter/Sort:** Ruff.
- **Line limit:** 100.
- **Indentation:** 4 spaces. No trailing whitespace.
- **Imports:** Absolute only. Grouping: StdLib → Third Party → Local. No wildcards (`from x import *`).
- **Naming:** `snake_case` (vars/funcs), `PascalCase` (classes/types), `UPPER_SNAKE_CASE` (constants), modules `lowercase_with_underscores`.
- Use type hints for all function signatures.
- Prefer f-strings for string formatting.
- Keep functions small and focused on a single responsibility.

### Type Hinting (Python 3.12+ Strict)
- **Tooling:** mypy (strict) or pyright.
- **Coverage:** Mandatory type hints for all function signatures (arguments and returns).
- **Syntax:** Use `|` for unions (e.g., `str | None`) and standard generics (`list[str]`, `dict[str, int]`).
- **No `Any` in APIs:** Do not use `Any` in function/method signatures, public classes, or exported types. Prefer precise types, `object` if truly unknown, or `Protocol` for structural typing.
   - If interfacing with untyped third‑party libraries, use `cast()` or isolated wrapper functions with explicit `# type: ignore[...]` and a short comment explaining why. `Any` may appear inside such wrapper *implementations* (locals/private helpers only), but must not leak into public signatures.
- **No magic numbers:** Define named constants or type aliases.
- **Abstractions:** Prefer `Sequence`, `Iterable`, or `Mapping` over concrete `list`/`dict` for arguments.
- **Data models:** Use Pydantic for external/API payloads; `TypedDict` for internal lightweight structures.

### Documentation
- **Format:** Google Style (preferred over reST).
- **Docstrings:** Triple double quotes (`"""`). Mandatory sections: `Args:`, `Returns:`, `Raises:`.
- **No types in docs:** Types belong in signatures, not text.
- **Coverage:** Aim for 100% docstring coverage; minimum 80% for exported symbols.
- **Comments:** Explain *why*, not *what*. Update comments immediately when code changes.

### Testing & CI/CD
- **Framework:** `pytest` with `pytest-asyncio`.
- **Isolation:** Tests must not hit external networks or disks (use `unittest.mock` or `respx`).
- **Coverage:**
   - **Repo floor:** Minimum 85% overall.
   - **Critical path:** 100% branch coverage required for `src/core` (or `src/logic` if used).
- **Strategy:** Fail fast. Run linting and type checks before tests.

### Architecture & Async Safety
- **Blocking ban:** Never call blocking I/O (`requests`, `time.sleep`) in `async def`. Use `httpx`, `asyncio.sleep`, `aiofiles`.
- **Subprocesses:** Strictly use `asyncio.create_subprocess_exec`. `subprocess.run`/`Popen` are banned.
- **Task safety:** Keep strong references to background tasks (e.g., a `set`) to prevent GC halting execution.
- **Exceptions:** Do not catch generic `Exception` without re‑raising or logging with `logger.exception`. Use custom exception classes inheriting from a project base exception for domain errors.

### Operational & Security
- **Logging:** Structured JSON logging preferred. Include context (trace IDs). `print()` is banned. Use typed logger methods directly (e.g., `logger.info()`, `logger.error()`) instead of string-based level wrappers (e.g., `log("info", msg)` or `logger.log(level_str, msg)`) to keep logging structured and type-safe.
- **Security:** Run `bandit` or `ruff check --select S` for security linting.

## ⚙️ Development Setup
<details>
<summary><b>Setting Up Your Environment (Recommended)</b></summary>

**Step 1: Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**Step 2: Install dependencies**
```bash
pip install -e ".[all]"
```
</details>

<details>
<summary><b>Setting Up Pre-commit (First Time Only)</b></summary>

**Step 1: Install pre-commit**
```bash
# Using pip
pip install pre-commit

# Or using conda
conda install -c conda-forge pre-commit
```

**Step 2: Install Git hooks**
```bash
pre-commit install
```
Hooks will now run automatically on every commit.

**Step 3: Initialize the Secrets Baseline**
If you encounter new "false positive" secrets (like API hash placeholders), update the baseline:
```bash
detect-secrets scan > .secrets.baseline
```
</details>

### Common Commands
| Task              | Command                                        |
|-------------------|------------------------------------------------|
| Check all files   | `pre-commit run --all-files`                   |
| Check quietly     | `pre-commit run --all-files -q`                |
| Update tools      | `pre-commit autoupdate`                        |
| Emergency skip    | `git commit --no-verify -m "message"` (Not recommended) |

## 📋 Commit Message Format
We use Conventional Commits. Squash‑merge only. PR titles must describe *why*, not just *what*.

Format:
```
<type>: <short description>

[optional body]
```
- `feat`: A new feature.
- `fix`: A bug fix.
- `docs`: Documentation only changes.
- `style`: Formatting-only changes.
- `refactor`: Code change that neither fixes a bug nor adds a feature.
- `test`: Adding or correcting tests.
- `chore`: Routine maintenance tasks.

## 💡 How to Get Started
1. Browse our [Issues](https://github.com/HKUDS/DeepTutor/issues) for tasks labeled `good first issue`.
2. Comment on the issue to let others know you're working on it.
3. Follow the process above and submit your PR!

Questions? Reach out on Discord.
Let's build the future of AI tutoring together! 🚀
