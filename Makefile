# ZKP Dataset Ledger Development Makefile

.PHONY: help build test clean fmt clippy audit bench docker docs install dev-setup

# Default target
help: ## Show this help message
	@echo "ZKP Dataset Ledger Development Commands:"
	@echo
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

# Build targets
build: ## Build the project
	cargo build

build-release: ## Build optimized release version
	cargo build --release

install: ## Install the CLI tool locally
	cargo install --path .

# Development targets
dev-setup: ## Set up development environment
	rustup component add rustfmt clippy
	cargo install cargo-audit cargo-tarpaulin cargo-expand
	@echo "Development environment setup complete!"

fmt: ## Format code using rustfmt
	cargo fmt --all

clippy: ## Run clippy linter
	cargo clippy --all-targets --all-features -- -D warnings

# Testing targets
test: ## Run all tests
	cargo test --all-features

test-integration: ## Run integration tests only
	cargo test --test integration_tests

test-coverage: ## Generate test coverage report
	cargo tarpaulin --out Html --output-dir coverage/

# Benchmarking
bench: ## Run all benchmarks
	cargo bench

bench-ci: ## Run benchmarks for CI (compile only)
	cargo bench --no-run

# Security and quality
audit: ## Check for security vulnerabilities
	cargo audit

outdated: ## Check for outdated dependencies
	cargo outdated

# Quality check combination
check: fmt clippy test audit ## Run all quality checks

# Docker targets
docker-build: ## Build Docker image
	docker build -t zkp-dataset-ledger .

docker-run: ## Run Docker container
	docker run --rm -v $(PWD)/test-data:/input zkp-dataset-ledger

docker-compose-up: ## Start all services with docker-compose
	docker-compose up -d

docker-compose-down: ## Stop all services
	docker-compose down

# Documentation
docs: ## Generate documentation
	cargo doc --no-deps --all-features

docs-open: ## Generate and open documentation
	cargo doc --no-deps --all-features --open

# Utility targets
clean: ## Clean build artifacts
	cargo clean
	rm -rf coverage/
	rm -rf target/criterion/

expand: ## Expand macros (useful for debugging)
	cargo expand

# Release preparation
pre-release: clean fmt clippy test audit docs ## Prepare for release
	@echo "Pre-release checks completed successfully!"

# Development workflows
dev: ## Development workflow: format, check, test
	$(MAKE) fmt
	$(MAKE) clippy
	$(MAKE) test

dev-fast: ## Fast development workflow: minimal features build and test
	cargo check --lib --no-default-features --features csv
	cargo test --lib --no-default-features --features csv

ci: ## CI workflow: all checks
	$(MAKE) fmt
	$(MAKE) clippy  
	$(MAKE) test
	$(MAKE) audit
	$(MAKE) bench-ci

# Database setup for postgres feature
db-setup: ## Set up development database
	docker-compose up -d postgres
	sleep 5
	@echo "Database setup complete!"

# Performance profiling
profile: ## Run performance profiling
	cargo build --release
	perf record --call-graph=dwarf ./target/release/zkp-ledger notarize test-data/sample.csv
	perf report

# Generate flame graph (requires cargo-flamegraph)
flamegraph: ## Generate flame graph for benchmarks
	cargo flamegraph --bench proof_generation

# Create test data
test-data: ## Generate test datasets
	mkdir -p test-data
	echo "id,value,category" > test-data/small.csv
	for i in $$(seq 1 1000); do echo "$$i,$$((i*10)),$$(((i-1) % 5))" >> test-data/small.csv; done
	echo "Generated test-data/small.csv with 1000 rows"

# Version management
version: ## Show current version
	@grep '^version = ' Cargo.toml | head -1

bump-patch: ## Bump patch version
	cargo install cargo-edit
	cargo set-version --bump patch

bump-minor: ## Bump minor version  
	cargo install cargo-edit
	cargo set-version --bump minor

bump-major: ## Bump major version
	cargo install cargo-edit
	cargo set-version --bump major