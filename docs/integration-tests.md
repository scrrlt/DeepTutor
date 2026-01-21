# Integration Tests

This project contains integration tests that exercise real provider APIs (OpenAI, Cohere, Anthropic, Perplexity, etc.). Integration tests are intentionally gated and **do not** run by default in CI or locally unless the required API keys are provided.

How to run integration tests locally

1. Add required keys to your environment (do NOT commit keys to the repository):

   - `OPENAI_API_KEY` (OpenAI)
   - `COHERE_API_KEY` (Cohere)
   - `ANTHROPIC_API_KEY` (Anthropic)
   - `PERPLEXITY_API_KEY` (Perplexity)
   - Optionally: `OPENAI_BASE_URL`, `COHERE_API_URL`, `ANTHROPIC_API_URL`, etc.

2. Run pytest targeting integration tests only:

   pytest -m integration -q

CI behavior

- The GitHub Actions workflow `.github/workflows/ci.yml` runs unit tests by default.
- Integration tests are run in a separate job and only if repository secrets for API keys are present in the Actions secrets. The job condition is expressed as:

  ```yaml
  if: ${{ secrets.OPENAI_API_KEY || secrets.COHERE_API_KEY || secrets.ANTHROPIC_API_KEY || secrets.PERPLEXITY_API_KEY }}
  ```

Security

- Do NOT add API keys to source control. Use GitHub repository secrets or environment variables in your CI runner.

Notes

- Integration tests are marked with `@pytest.mark.integration` and will be skipped automatically if the corresponding environment variables are not set.
- If you want more providers added to integration test coverage, open an issue or add the tests and mark them with `@pytest.mark.integration`.
