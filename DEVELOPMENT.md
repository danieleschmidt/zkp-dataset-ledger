# Quick Development Setup

This file provides quick setup instructions. For comprehensive development documentation, see `docs/DEVELOPMENT.md`.

## Quick Start

```bash
# Prerequisites (install once)
make dev-setup

# Development workflow
make dev          # Format, lint, test
make test-coverage # Generate coverage report
make docker-build  # Build container

# Quality checks before commit
make check        # All quality checks
make pre-release  # Full release preparation
```

## Essential Commands

- `make help` - Show all available commands
- `make build` - Build the project
- `make test` - Run all tests
- `make fmt` - Format code
- `make clippy` - Run linter
- `make audit` - Security audit
- `make docs` - Generate documentation

## Environment Setup

1. Install Rust: https://rustup.rs/
2. Run: `make dev-setup`
3. Test: `make dev`

For detailed setup instructions, see `docs/DEVELOPMENT.md`.