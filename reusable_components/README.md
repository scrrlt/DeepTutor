# Reusable Components from Chimera Platform

This folder contains reusable components, logic, and code extracted from the Chimera Platform that could be beneficial for DeepTutor.

## Components

### Python Utilities

- **budget_enforcer.py**: Daily token & cost budget enforcement utility with file-based storage. Useful for managing LLM API usage limits.

- **rate_limiter.py**: Token bucket rate limiter for API calls. Helps prevent rate limit violations on external services.

- **retry_logic.py**: Comprehensive exponential backoff retry logic with error categorization and jitter. Advanced retry mechanism for resilient API calls.

- **code_generator.py**: Comprehensive code generation utility with templates for routes, models, tests, CLI commands, migrations, and Dockerfiles. Useful for scaffolding new features.

- **feature_flags.py**: Feature flag system with environment variable and file-based configuration.

- **schema_validator.py**: Environment variable schema validation tool with type checking and defaults.

- **mock_server.py**: Simple Flask mock server for testing API integrations.

- **smoke_tests.py**: Comprehensive end-to-end smoke testing framework with colored output and multiple test scenarios.

### JavaScript Utilities

- **logger.js**: Comprehensive logging system with structured JSON logging, file streams, performance tracking, and middleware for Express.js applications.

- **error_tracker.js**: Error tracking utility with in-memory storage and statistics.

- **oauth.js**: OAuth 2.0 implementation supporting Google, GitHub, and Microsoft providers with user info normalization.

### Authentication & User Management

- **auth/api_key_auth.py**: API key authentication service using Google Secret Manager. Can be adapted for API key management.

- **user_manager.py**: Complete user management system with Firebase integration, JWT tokens, usage tracking, role-based access control, and quota management.

### Agent Framework

- **agents/base.py**: Abstract base class for research agents.

- **agents/insight_agent.py**: Example agent implementation.

### Deployment & Operations

- **deploy.sh**: Google Cloud Run deployment script for multi-service applications.

## Integration Notes

These components can be integrated into DeepTutor as follows:

1. **Budget Enforcer**: Replace or enhance the existing budget.py in src/services/llm/
2. **Rate Limiter**: Use in LLM client or API routers to prevent rate limit issues
3. **Retry Logic**: Enhance the existing error_handler.py with more sophisticated retry strategies
4. **Logger**: Adapt for the web frontend (Next.js) to replace or supplement existing logging
5. **Code Generator**: Use for quickly scaffolding new API routes, models, and tests
6. **Auth**: Adapt for API key authentication in DeepTutor's API
7. **User Manager**: Implement user accounts, authentication, and usage tracking
8. **OAuth**: Add social login capabilities
9. **Error Tracker**: Enhance error monitoring and reporting
10. **Feature Flags**: Control feature rollout and A/B testing
11. **Schema Validator**: Validate environment configuration
12. **Mock Server**: Test API integrations without external dependencies
13. **Smoke Tests**: Implement comprehensive end-to-end testing
14. **Agent Base**: Use as a foundation for new agent types in DeepTutor

## Source

All components are extracted from the Chimera Platform repository at D:\dev-hub\repo-base\chimera-platform