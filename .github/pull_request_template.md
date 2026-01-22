<!--
Thank you for contributing to DeepTutor! 🚀
Please ensure your PR is ready for review and follows our contribution guidelines.
For more details, see our [CONTRIBUTING.md](https://github.com/HKUDS/DeepTutor/blob/dev/CONTRIBUTING.md).
-->

### Description
*A clear and concise description of the changes.*

### Related Issues
- Closes #...
- Related to #...

### Module(s) Affected
- [ ] `agents`
- [ ] `api`
- [ ] `config`
- [ ] `core`
- [ ] `knowledge`
- [ ] `logging`
- [ ] `services`
- [ ] `tools`
- [ ] `utils`
- [ ] `web` (Frontend)
- [ ] `docs` (Documentation)
- [ ] `scripts`
- [ ] `tests`
- [ ] Other: `...`

### Checklist
- [ ] I have read and followed the [contribution guidelines](https://github.com/HKUDS/DeepTutor/blob/dev/CONTRIBUTING.md).
- [ ] My code follows the project's coding standards.
- [ ] I have run `pre-commit run --all-files` and fixed any issues.
- [ ] I have added relevant tests for my changes.
- [ ] I have updated the documentation (if necessary).
- [ ] My changes do not introduce any new security vulnerabilities.

### Test Results
*Run the following commands and paste results:*

#### Pytest Coverage
```bash
pytest --cov=src --cov-report=term-missing
# Result: e.g., "50 passed, 2 failed, 10 skipped, 85% coverage"
```

#### Quality Checks
```bash
# Mypy
mypy --config-file pyproject.toml src/
# Result: e.g., "Success: no issues found in X source files"

# Ruff
ruff check src/
# Result: e.g., "Found 0 errors."

# Bandit
bandit -r src/
# Result: e.g., "Run completed successfully. X issues found."
```

### Additional Notes
*Add any other context or screenshots about the pull request here.*
